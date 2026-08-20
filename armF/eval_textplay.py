"""P72: the P70 text-path model PLAYS hex. Player = spliced polish LM +
P70-trained blocks 23-27 (polish_top_all_dlab), moves emitted as text
(greedy, 6 tokens, parsed), illegal/unparseable -> random legal + counter.

Protocol = eval_stitch_polish play_games_p2 (plays SECOND vs distilled
argmax; fresh random openings, seeds 2000+g) so numbers are comparable to
stitch cuts (P54: k18 3/40 on 4-ply, k0 26/40).
"""
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import eval_stitch_moves as EM  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402

DEV = "cuda"
PAT = re.compile(r"^\n([A-K])(\d{2})")


def load_textpath_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16).cuda().eval()
    ck = torch.load("checkpoints/armF_polish19b/final.pt",
                    map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in ck["backbone"].items():
            if k.startswith("layers."):
                obj = model.model
                for p in k.split(".")[:-1]:
                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                getattr(obj, k.split(".")[-1]).copy_(v.bfloat16().cuda())
    del ck
    ck70 = torch.load("checkpoints/armF_p60/polish_top_all_dlab.pt",
                      map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in ck70["trainable"].items():
            obj = model
            for p in k.split(".")[:-1]:
                obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
            getattr(obj, k.split(".")[-1]).copy_(v.bfloat16().cuda())
    del ck70
    return model


def make_text_player(model, tok, ctr):
    @torch.no_grad()
    def fn(b, moves):
        assert len(moves) % 2 == 1  # plays second: last move is X's
        ctr["moves"] += 1
        txt = SP.d04n_text(list(moves))
        ids = tok(txt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        cont = tok.decode(out[0, ids.shape[1]:])
        m = PAT.match(cont)
        if m:
            mv = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
            if mv in b.legal_moves:
                return mv
        ctr["illegal"] += 1
        return random.choice(sorted(b.legal_moves))
    return fn


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cnn = W.load_model()
    student = R4.load_student(cnn)
    model = load_textpath_model()

    random.seed(0)
    ctr = {"moves": 0, "illegal": 0}
    tp = make_text_player(model, tok, ctr)
    w_rand = SP.play_games_p2(tp, EM.make_random_player(), 20)
    print(f"vs random: {w_rand}/20 (illegal so far "
          f"{ctr['illegal']}/{ctr['moves']})", flush=True)
    w4 = SP.play_games_p2(tp, B.make_bn_player(student, [0]), 40,
                          opening_plies=4)
    print(f"vs distilled 4-ply: {w4}/40 (illegal so far "
          f"{ctr['illegal']}/{ctr['moves']})", flush=True)
    w1 = SP.play_games_p2(tp, B.make_bn_player(student, [0]), 40,
                          opening_plies=1)
    print(f"vs distilled 1-ply: {w1}/40 (illegal so far "
          f"{ctr['illegal']}/{ctr['moves']})", flush=True)

    res = {"vs_random": f"{w_rand}/20", "vs_dist_4ply": f"{w4}/40",
           "vs_dist_1ply": f"{w1}/40", "text_moves": ctr["moves"],
           "illegal_or_unparseable": ctr["illegal"],
           "illegal_rate": ctr["illegal"] / max(ctr["moves"], 1)}
    print(json.dumps(res, indent=1), flush=True)
    Path("armF/results/p72_textplay.json").write_text(json.dumps(res,
                                                                 indent=1))


if __name__ == "__main__":
    main()
