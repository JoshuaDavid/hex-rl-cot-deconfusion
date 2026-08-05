"""Generation backend: vLLM if available, else batched transformers.

API: gen = Backend(model_name); texts = gen.generate(chat_prompts, n=1, ...)
chat_prompts: list of user-message strings. Returns list of lists (n per prompt).
"""

from __future__ import annotations


class Backend:
    def __init__(self, model_name: str, max_model_len: int = 8192, enable_thinking: bool = True):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        try:
            from vllm import LLM

            self.vllm = LLM(model=model_name, max_model_len=max_model_len,
                            gpu_memory_utilization=0.85)
            self.kind = "vllm"
        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa",
            )
            self.model.eval()
            self.kind = "hf"

    def _render(self, user_msg: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate(self, user_msgs: list[str], n: int = 1, temperature: float = 0.6,
                 top_p: float = 0.95, max_tokens: int = 2048, seed: int = 0,
                 batch_size: int = 16) -> list[list[str]]:
        rendered = [self._render(m) for m in user_msgs]
        if self.kind == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(temperature=temperature, top_p=top_p,
                                max_tokens=max_tokens, n=n, seed=seed)
            outs = self.vllm.generate(rendered, sp)
            return [[o.text for o in out.outputs] for out in outs]

        import torch

        torch.manual_seed(seed)
        flat = [r for r in rendered for _ in range(n)]
        results: list[str] = []
        for i in range(0, len(flat), batch_size):
            chunk = flat[i:i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
            gen = out[:, enc.input_ids.shape[1]:]
            results.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
            print(f"  hf-generate {min(i + batch_size, len(flat))}/{len(flat)}", flush=True)
        return [results[i * n:(i + 1) * n] for i in range(len(user_msgs))]
