"""Arm E R4: single-token task-SELECTION RL agent loop.

Rollout structure (only the ONE selection token is trained):
  [prompt: select_prompt]                                    (prompt, untrained)
  <selected-task>                                            (forced ctx, mask 0)
  X            <- ONE token, sampled from {A,B,D,E}          (TRAINED, mask 1)
  </selected-task>\n<task-X>                                 (forced ctx, mask 0)
  ...model's own helper answer...                            (generated, mask 0)
  </task-X>\n<evaluated-task>                                (forced ctx, mask 0)
  ...model's own answer to Task C...                         (generated, mask 0)
  </evaluated-task>                                          (forced ctx, mask 0)

reward = grade(C) on the model's own evaluated answer. GRPO's group-relative
baseline (group_n rollouts / board) contrasts the C-reward across the different
X's sampled in the group, so the gradient on the single X token pushes selection
toward the instrumentally-useful helper. Everything except X has mask 0, so the
helper/answer generations are environment only — the policy learns to SELECT.

Logs one side-channel record per rollout (selected X + C score) to
HEX_ROLLOUT_LOG for P(select=X) tracking.
"""

import json as _json
import logging
import os
from uuid import uuid4
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op

import sys
sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")
from hexenv.arme import (board_from_gt, grade, extract_tag, parse_answer,
                         SELECTABLE, EVALUATED)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

HELPER_BUDGET = int(os.getenv("ARME_HELPER_BUDGET", "160"))
ANSWER_BUDGET = int(os.getenv("ARME_ANSWER_BUDGET", "120"))
# Decouple exploration from generation: the SELECTION token is sampled at the
# rollout temperature (exploration over A/C/D), but the helper + evaluated answer
# are generated at ARME_GEN_TEMP (default greedy) so a hard-to-generate but
# useful helper (D) is NOT derailed by high-temp noise -> its content advantage
# survives into the reward. Without this, high-temp helper generation erases the
# differential (D's edge is at temp 0). See RESEARCH_LOG 2026-08-10.
GEN_TEMP = float(os.getenv("ARME_GEN_TEMP", "0.0"))


@register("hex_select")
class HexSelectAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        tok = self.tokenizer
        self._sel_open = tok.encode("<selected-task>", add_special_tokens=False)
        # single-token id for each selectable letter (A/B/D/E), no leading space
        self._letter_id = {c: tok.encode(c, add_special_tokens=False)[0]
                           for c in SELECTABLE}
        self._id_letter = {v: k for k, v in self._letter_id.items()}
        self._allowed = list(self._letter_id.values())

    def _enc(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        tok = self.tokenizer

        resp_ids, resp_mask, resp_lps = [], [], []

        def add(ids, trained, lps=None):
            resp_ids.extend(ids)
            resp_mask.extend([1 if trained else 0] * len(ids))
            resp_lps.extend(lps if lps is not None else [0.0] * len(ids))

        # forced <selected-task>
        add(self._sel_open, False)

        # === the ONE trained token: pick X in {A,B,D,E} ===
        sp_sel = dict(sampling_params)
        sp_sel["n"] = 1
        sp_sel["max_tokens"] = 1
        sp_sel["allowed_token_ids"] = self._allowed
        sp_sel.pop("stop", None)
        sp_sel.pop("stop_token_ids", None)
        out_sel = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids + resp_ids,
            sampling_params=sp_sel, priority=priority)
        x_id = list(out_sel.token_ids)[0]
        x_lp = (list(out_sel.log_probs)[0] if out_sel.log_probs else 0.0)
        x = self._id_letter.get(x_id, SELECTABLE[0])
        add([x_id], True, [x_lp])

        # forced </selected-task>\n<task-X>
        add(self._enc(f"</selected-task>\n<task-{x}>"), False)

        # === helper generation (untrained env step) ===
        sp_h = dict(sampling_params)
        sp_h["n"] = 1
        sp_h["temperature"] = GEN_TEMP
        sp_h["max_tokens"] = HELPER_BUDGET
        sp_h["stop"] = [f"</task-{x}>", "</evaluated-task>"]
        sp_h.pop("stop_token_ids", None)
        out_h = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids + resp_ids,
            sampling_params=sp_h, priority=priority)
        add(list(out_h.token_ids), False, None)

        # forced </task-X>\n<evaluated-task>
        add(self._enc(f"</task-{x}>\n<evaluated-task>"), False)

        # === evaluated (Task C) generation (untrained env step) ===
        sp_c = dict(sampling_params)
        sp_c["n"] = 1
        sp_c["temperature"] = GEN_TEMP
        sp_c["max_tokens"] = ANSWER_BUDGET
        sp_c["stop"] = ["</evaluated-task>", "<|im_end|>"]
        sp_c.pop("stop_token_ids", None)
        out_c = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids + resp_ids,
            sampling_params=sp_c, priority=priority)
        c_text = tok.decode(list(out_c.token_ids))
        add(list(out_c.token_ids), False, None)
        add(self._enc("</evaluated-task>"), False)

        # === reward: grade the model's own Task-C answer ===
        gt_str = kwargs.get("reward_model", {}).get("ground_truth") \
            if isinstance(kwargs.get("reward_model"), dict) else None
        c_score, c_perfect = -1.0, 0.0
        try:
            gt = _json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            board = board_from_gt(gt)
            payload = parse_answer(c_text.split("</evaluated-task>")[0], EVALUATED)
            s, p = grade(EVALUATED, board, payload) if payload is not None else (-1.0, False)
            c_score, c_perfect = float(s), float(p)
        except Exception:
            logger.exception("arme reward failed")

        log_path = os.environ.get("HEX_ROLLOUT_LOG")
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(_json.dumps({
                        "selected": x, "score": c_score, "perfect": c_perfect,
                        "kind": "win" if c_perfect else "lose",
                        "helper_text": tok.decode(list(out_h.token_ids))[:200],
                        "c_text": c_text[:200],
                    }) + "\n")
            except OSError:
                pass

        extra_fields = dict(out_sel.extra_fields)
        for o in (out_h, out_c):
            extra_fields.update(o.extra_fields)
        extra_fields.update({"turn_scores": [], "tool_rewards": [],
                             "selected_task": x})

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=resp_ids[: self.response_length],
            response_mask=resp_mask[: self.response_length],
            response_logprobs=resp_lps[: self.response_length],
            reward_score=c_score,
            num_turns=3,
            metrics={},
            extra_fields=extra_fields,
        )
