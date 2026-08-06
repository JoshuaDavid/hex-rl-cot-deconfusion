"""Export a peft-format LoRA adapter dir from a verl FSDP SFT checkpoint.

verl saves the full peft-wrapped state dict (base weights + lora_A/B with
'.default.' adapter naming); vllm's LoRARequest wants a standalone peft
adapter (adapter_config.json + adapter_model.safetensors).

Run: /venv/verl/bin/python scripts/export_armD_adapter.py <global_step_dir> <out_dir>
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file

ckpt_dir, out_dir = sys.argv[1], sys.argv[2]
sd = torch.load(os.path.join(ckpt_dir, "model_world_size_1_rank_0.pt"),
                map_location="cpu", weights_only=False)
meta = json.load(open(os.path.join(ckpt_dir, "lora_train_meta.json")))

lora = {}
targets = set()
for k, v in sd.items():
    if ".lora_A.default.weight" in k or ".lora_B.default.weight" in k:
        nk = k.replace(".default.weight", ".weight")
        lora[nk] = v.contiguous().to(torch.bfloat16)
        targets.add(k.split(".lora_")[0].rsplit(".", 1)[-1])

assert lora, "no lora keys found"
os.makedirs(out_dir, exist_ok=True)
save_file(lora, os.path.join(out_dir, "adapter_model.safetensors"))
cfg = {
    "peft_type": "LORA",
    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    "task_type": meta.get("task_type", "CAUSAL_LM"),
    "r": meta["r"],
    "lora_alpha": meta["lora_alpha"],
    "lora_dropout": 0.0,
    "bias": "none",
    "fan_in_fan_out": False,
    "target_modules": sorted(targets),
    "modules_to_save": None,
}
json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)
print(f"wrote {len(lora)} lora tensors, targets {sorted(targets)} -> {out_dir}")
