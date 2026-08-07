"""Forced-close agent loop for hex GRPO training.

Two-phase generation:
1. Think phase: generate with stop at </think>, capped at think_budget tokens.
2. Answer scaffold "</think>\n\nMove:" (injected, mask 0 for injected tokens;
   a model-generated </think> keeps mask 1), then a short answer phase (mask 1).

Guarantees every rollout yields a parseable "Move: <cell>" tail regardless of
whether the model's think phase terminates on its own — needed because
Qwen3-1.7B base ~never terminates thinking on open-ended move choice
(see RESEARCH_LOG 2026-08-05).
"""

import json as _json
import logging
import os
import random as _random
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ANSWER_BUDGET = 8


@register("hex_forced_close")
class HexForcedCloseAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.think_budget = self.response_length - ANSWER_BUDGET - 8
        tok = self.tokenizer
        self.think_close_id = tok.convert_tokens_to_ids("</think>")
        assert isinstance(self.think_close_id, int) and self.think_close_id >= 0
        self._nl_ids = tok.encode("\n", add_special_tokens=False)
        # task-dependent answer scaffolds (see _scaffolds)
        self._scaffold_cache = {}

    def _scaffolds(self, answer_word: str):
        if answer_word not in self._scaffold_cache:
            tok = self.tokenizer
            scaffold = tok.encode(f"\n\n{answer_word}:", add_special_tokens=False)
            force = self._nl_ids + [self.think_close_id] + scaffold
            self._scaffold_cache[answer_word] = (scaffold, force)
        return self._scaffold_cache[answer_word]

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)

        # task-aware answer scaffold: judgment prompts ask for "Answer: ...",
        # move prompts for "Move: ..."
        user_text = messages[-1].get("content", "") if messages else ""
        if "Answer: Black|White|Neither" in user_text:
            answer_word, answer_budget = "Answer", 8
        elif ("Answer: <comma-separated list of cells>" in user_text
              or 'a JSON array' in user_text):
            answer_word, answer_budget = "Answer", 48
        elif '"winner"' in user_text:
            answer_word, answer_budget = "Answer", int(
                os.getenv("HEX_WITNESS_ANSWER_BUDGET", "64"))
        else:
            answer_word, answer_budget = "Move", ANSWER_BUDGET
        scaffold_ids, force_ids = self._scaffolds(answer_word)
        # think cap must leave room for THIS task's answer within response_length
        think_budget = self.response_length - answer_budget - 8
        # hard cap on think tokens (e.g. =2 for the minimal-CoT RL probe).
        # In cap mode the deterministic opener '<think>\n' (p=1.0 under the
        # SFT'd policy) is prefilled as context so every capped token is a
        # free content token, and the think phase can use its own sampling
        # temperature/top_k (HEX_THINK_TEMP / HEX_THINK_TOPK) so exploration
        # over the register tokens survives a collapsed policy; the
        # off-policy distortion is bounded by PPO clipping.
        cap = os.getenv("HEX_THINK_CAP_TOKENS")
        think_temp = float(os.getenv("HEX_THINK_TEMP", "0") or 0)
        think_topk = int(os.getenv("HEX_THINK_TOPK", "0") or 0)
        if cap:
            think_budget = min(think_budget, int(cap))
            opener = self.tokenizer.encode("<think>\n", add_special_tokens=False)
            prompt_ids = list(prompt_ids) + opener

        metrics = {}
        # phase 1: think — segmented with a rising logit bias on </think>
        # (HEX_CLOSE_BIAS="512:0,832:6,1088:12" => tokens 0-512 bias 0,
        # 512-832 bias 6, 832-1088 bias 12). Empty/unset = single unbiased
        # segment (legacy behavior). Bias-sampled closes stay mask-1; the
        # off-policy distortion on that single token is handled by PPO clipping.
        schedule = []
        sched_env = os.getenv("HEX_CLOSE_BIAS", "")
        if sched_env:
            for part in sched_env.split(","):
                end, bias = part.split(":")
                schedule.append((min(int(end), think_budget), float(bias)))
        else:
            schedule = [(think_budget, 0.0)]

        think_ids: list[int] = []
        think_lps: list[float] = []
        out1 = None
        for seg_end, bias in schedule:
            remaining = seg_end - len(think_ids)
            if remaining <= 0:
                continue
            sp1 = dict(sampling_params)
            sp1["max_tokens"] = remaining
            sp1["stop_token_ids"] = [self.think_close_id]
            if bias:
                sp1["logit_bias"] = {self.think_close_id: bias}
            if think_temp > 0:
                sp1["temperature"] = think_temp
            if think_topk > 0:
                sp1["top_k"] = think_topk
            out1 = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + think_ids,
                sampling_params=sp1,
                priority=priority,
            )
            think_ids += list(out1.token_ids)
            think_lps += (list(out1.log_probs) if out1.log_probs
                          else [0.0] * len(out1.token_ids))
            if think_ids and think_ids[-1] == self.think_close_id:
                break

        # normalize: strip a model-generated trailing </think> (keep flag)
        natural = bool(think_ids) and think_ids[-1] == self.think_close_id
        if natural:
            close_lp = think_lps[-1] if think_lps else 0.0
            think_ids = think_ids[:-1]
            think_lps = think_lps[:-1]

        response_ids = list(think_ids)
        response_mask = [1] * len(response_ids)
        response_lps = list(think_lps)
        if natural:
            # model closed its own think: closing tag is model-generated
            response_ids += [self.think_close_id]
            response_mask += [1]
            response_lps += [close_lp]
            response_ids += scaffold_ids
            response_mask += [0] * len(scaffold_ids)
            response_lps += [0.0] * len(scaffold_ids)
        else:
            response_ids += force_ids
            response_mask += [0] * len(force_ids)
            response_lps += [0.0] * len(force_ids)

        # phase 2: answer — branch n=A answers per think (one request, shared
        # KV). Think reward = mean over answers (Rao-Blackwellized credit; see
        # RESEARCH_LOG 2026-08-06). One randomly chosen answer is emitted as
        # the trained tokens; reward_score carries the mean.
        n_branch = int(os.getenv("HEX_ANSWER_BRANCH", "1"))
        # run() never sees verl's validate flag; val is distinguishable by
        # val_kwargs.temperature (0.6) vs training (1.0). Keep val unbranched
        # so val metrics stay comparable across branches/arms.
        if float(sampling_params.get("temperature", 1.0)) < 1.0:
            n_branch = 1
        sp2 = dict(sampling_params)
        sp2["max_tokens"] = answer_budget
        branch_reward = None
        gt_str = kwargs.get("reward_model", {}).get("ground_truth") \
            if isinstance(kwargs.get("reward_model"), dict) else None
        # verl's vllm server passes n into SamplingParams but returns only
        # outputs[0] (and TokenOutput.token_ids is a flat list[int]), so n>1
        # in one request silently wastes n-1 answers. Fire n parallel requests
        # instead; prefix cache shares the think KV, answers are 8-64 tokens.
        import asyncio as _asyncio
        outs = await _asyncio.gather(*[
            self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=dict(sp2),
                priority=priority,
            ) for _ in range(max(1, n_branch))
        ])
        out2: TokenOutput = outs[0]
        if n_branch > 1 and gt_str is not None:
            # Setting reward_score below bypasses verl's async reward path, so
            # the reward-fn side channel would go silent for these samples. The
            # loop scores each branch with logging suppressed (sync block — no
            # awaits, so no coroutine can interleave the env mutation) and
            # writes ONE side-channel record itself (controller depends on it).
            all_ids = [list(o.token_ids) for o in outs]
            all_lps = [list(o.log_probs) if o.log_probs else None for o in outs]
            log_path = os.environ.pop("HEX_ROLLOUT_LOG", None)
            try:
                from hexenv import reward_verl
                prefix = self.tokenizer.decode(response_ids)
                scores = []
                for ids in all_ids:
                    r = reward_verl.compute_score(
                        "hex", prefix + self.tokenizer.decode(ids), gt_str)
                    scores.append(r["score"] if isinstance(r, dict) else r)
                branch_reward = sum(scores) / len(scores)
            except Exception:
                logger.exception("answer-branch scoring failed")
                branch_reward = None
            finally:
                if log_path is not None:
                    os.environ["HEX_ROLLOUT_LOG"] = log_path
            pick = _random.randrange(len(all_ids))
            picked_ids = list(all_ids[pick])
            picked_lps = (list(all_lps[pick]) if all_lps[pick]
                          else [0.0] * len(picked_ids))
            if branch_reward is not None and log_path:
                try:
                    gt = _json.loads(gt_str) if isinstance(gt_str, str) else gt_str
                    with open(log_path, "a") as f:
                        f.write(_json.dumps({
                            "gt": gt, "move": None,
                            "kind": "win" if scores[pick] > 0 else "lose",
                            "score": scores[pick], "shaped": branch_reward,
                            "branch_scores": scores, "n_branch": len(all_ids),
                            "response": prefix + self.tokenizer.decode(picked_ids),
                        }) + "\n")
                except OSError:
                    pass
        else:
            picked_ids = list(out2.token_ids)
            picked_lps = (list(out2.log_probs) if out2.log_probs
                          else [0.0] * len(picked_ids))
        response_ids += picked_ids
        response_mask += [1] * len(picked_ids)
        response_lps += picked_lps

        # merge server-stamped weight-version fields across both phases
        extra_fields = dict(out1.extra_fields)
        extra_fields.update(out2.extra_fields)
        mins = [f.get("min_global_steps") for f in (out1.extra_fields, out2.extra_fields)]
        maxs = [f.get("max_global_steps") for f in (out1.extra_fields, out2.extra_fields)]
        mins = [m for m in mins if m is not None]
        maxs = [m for m in maxs if m is not None]
        if mins:
            extra_fields["min_global_steps"] = min(mins)
        if maxs:
            extra_fields["max_global_steps"] = max(maxs)
        extra_fields.update({"turn_scores": [], "tool_rewards": [],
                             "natural_close": natural})

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_lps[: self.response_length],
            reward_score=branch_reward,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
