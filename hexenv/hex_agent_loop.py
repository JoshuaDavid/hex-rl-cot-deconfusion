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

import logging
import os
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
        answer_word = "Answer" if "Answer: Black|White|Neither" in user_text else "Move"
        scaffold_ids, force_ids = self._scaffolds(answer_word)

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
                schedule.append((min(int(end), self.think_budget), float(bias)))
        else:
            schedule = [(self.think_budget, 0.0)]

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

        # phase 2: answer
        sp2 = dict(sampling_params)
        sp2["max_tokens"] = ANSWER_BUDGET
        out2: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids + response_ids,
            sampling_params=sp2,
            priority=priority,
        )
        response_ids += list(out2.token_ids)
        response_mask += [1] * len(out2.token_ids)
        response_lps += (list(out2.log_probs) if out2.log_probs
                         else [0.0] * len(out2.token_ids))

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
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
