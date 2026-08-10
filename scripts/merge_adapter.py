"""Merge a LoRA adapter into the base model -> a full bf16 HF model for vllm/RL.
Run: /venv/verl/bin/python scripts/merge_adapter.py <adapter_dir> <out_dir> [base]
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-1.7B"
if os.path.exists(os.path.join(out, "config.json")):
    print(f"{out} exists, skipping")
    sys.exit(0)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print(f"merged {adapter} -> {out}")
