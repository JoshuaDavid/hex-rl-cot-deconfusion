"""Does hosting the CNN destroy Qwen's language? Splice FT'd blocks 0..22 into
the original model (blocks 23..27, norms, embeddings, lm_head untouched) and
compare token-level NLL on a text sample + greedy generations."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

SAMPLE = (
    "The theory of evolution by natural selection explains how species change "
    "over time. Individuals with traits better suited to their environment tend "
    "to survive and reproduce, passing those traits to their offspring. Over many "
    "generations, this process can produce new species. Darwin developed this "
    "idea after observing finches on the Galapagos Islands, noticing that beak "
    "shapes varied with food sources on each island."
)
PROMPTS = ["The capital of France is",
           "def fibonacci(n):",
           "Water boils at"]


def nll(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    return out.loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_containment_r1/best.pt")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    base_nll = nll(model, tok, SAMPLE)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["backbone"]  # truncated backbone (23 blocks), keys like layers.N....
    n_loaded = 0
    with torch.no_grad():
        for k, v in sd.items():
            if k.startswith("layers."):
                obj = model.model
                parts = k.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                getattr(obj, parts[-1]).copy_(v.bfloat16().cuda())
                n_loaded += 1
    print(f"spliced {n_loaded} tensors from FT blocks 0..22")
    ft_nll = nll(model, tok, SAMPLE)
    print(f"NLL original {base_nll:.3f} -> spliced {ft_nll:.3f} "
          f"(ppl {torch.tensor(base_nll).exp():.1f} -> {torch.tensor(ft_nll).exp():.1f})")
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=25, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        print(f"  {p!r} -> {tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
