"""SFT dataset that keeps <think> content in the loss.

MultiTurnSFTDataset renders each turn through the chat template; Qwen3's
template strips <think>...</think> from assistant messages, so the v1
certificate SFT silently trained on answer-only targets (RESEARCH_LOG
2026-08-06 — the think-ablation accident). This subclass tokenizes assistant
turns raw (<|im_start|>assistant\n + content + <|im_end|>\n) so the narration
survives; all other turns use the parent path.

Acceptance test before any training run: decode input_ids[loss_mask] and eyeball
that the think block is present.
"""

import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class CertSFTDataset(MultiTurnSFTDataset):
    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        if message.get("role") != "assistant":
            return super()._process_single_message(
                index, message, full_message, tools=tools, enable_thinking=enable_thinking
            )
        text = f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.ones_like(input_ids)
        loss_mask[: len(self.generation_prompt)] = 0
        return input_ids, loss_mask, attention_mask, {}
