"""Arm F post-r3: how important is attention per layer, and what does it attend to?

Two experiments on the r3ext render-free checkpoint:

1. --mode ablate: per-layer attention ablation, two variants:
   - zero: attn sublayer output -> 0 (removes mixing AND per-token attn write)
   - selfonly: attn output -> o_proj(v_self) (keeps per-token write, kills mixing)
   Metric: mean val R2 (same 60-seq val set as training) vs unablated baseline.

2. --mode patterns: eager attention on N val seqs; from move (color) query
   tokens aggregate attention mass by key class (sink/pre/self/nl/num/cell/
   color), by record recency (own record, prev record, older), and by hex
   adjacency of the attended move's cell to the query move's cell.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402
import train_movesonly_z0 as Z  # noqa: E402
import train_movesfull as F  # noqa: E402
import render_moves as RM  # noqa: E402

DEV = "cuda"
L = 19
NB = 23  # backbone blocks


def load_all(ckpt):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    recs = Z.build_seqs(tok, games, "numbered")
    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:60]
    cnn = W.load_model()
    backbone = T.load_backbone()
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    mu, sd = d["mu"].to(DEV), d["sd"].to(DEV)
    ads = nn.ModuleList([nn.Linear(2048, 7744) for _ in range(L)]).to(DEV)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, _ = backbone.load_state_dict(
        {k: v.float() for k, v in ck["backbone"].items()}, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    ads.load_state_dict({k: v.float() for k, v in ck["ads"].items()})
    print(f"loaded {ckpt} (step {ck.get('step')}, "
          f"ckpt r2 {sum(ck['r2'])/len(ck['r2']):.4f})", flush=True)
    return tok, games, recs, val, cnn, backbone, ads, mu, sd


def zero_hook(module, args, kwargs, output):
    return (torch.zeros_like(output[0]),) + tuple(output[1:])


def selfonly_hook(module, args, kwargs, output):
    h = kwargs.get("hidden_states", args[0] if args else None)
    B, Tn, _ = h.shape
    v = module.v_proj(h).view(B, Tn, -1, module.head_dim)
    nrep = (module.config.num_attention_heads
            // module.config.num_key_value_heads)
    v = v.repeat_interleave(nrep, dim=2).reshape(B, Tn, -1)
    return (module.o_proj(v),) + tuple(output[1:])


def run_ablate(backbone, ads, cnn, games, recs, val, mu, sd, out):
    base = F.evaluate(backbone, ads, cnn, games, recs, val, mu, sd)
    print(f"baseline mean R2 {base.mean().item():.4f}", flush=True)
    res = {"baseline": base.tolist(), "zero": {}, "selfonly": {}}
    for variant, hook in [("zero", zero_hook), ("selfonly", selfonly_hook)]:
        for j in range(NB):
            hnd = backbone.layers[j].self_attn.register_forward_hook(
                hook, with_kwargs=True)
            r2 = F.evaluate(backbone, ads, cnn, games, recs, val, mu, sd)
            hnd.remove()
            res[variant][j] = r2.tolist()
            print(f"{variant} L{j:2d}: mean {r2.mean().item():.4f} "
                  f"(d {r2.mean().item()-base.mean().item():+.4f}) | "
                  f"z0 {r2[0]:.3f} z9 {r2[9]:.3f} z18 {r2[18]:.3f}",
                  flush=True)
        Path(out).write_text(json.dumps(res))
    print(f"wrote {out}")


HEX_NBRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]


def char_labels(moves):
    """Per-char (class, record) labels for the numbered-format text."""
    lab = [("pre", -1)] * len(RM.PREAMBLE_M)
    for t, mv in enumerate(moves):
        cell = RM.move_str(mv)
        color = "X" if t % 2 == 0 else "O"
        for part, cls in [("\n", "nl"), (f"{t+1}.", "num"),
                          (f" {cell}", "cell"), (f" {color}", "color")]:
            lab += [(cls, t)] * len(part)
    return lab


@torch.no_grad()
def run_patterns(backbone, tok, games, recs, val, out, n_seqs=8, min_t=4):
    backbone.config._attn_implementation = "eager"
    backbone.eval()
    CLS = ["sink", "pre", "self", "nl", "num", "cell", "color"]
    DELTA = ["own", "prev", "d2", "d3_5", "d6_10", "d11p"]
    cls_mass = torch.zeros(NB, len(CLS))
    dl_mass = torch.zeros(NB, len(DELTA))
    head_prev = torch.zeros(NB, 16)   # mass on prev record, per head
    adj = torch.zeros(NB, 2)          # per-record mass: adjacent vs not
    adj_n = torch.zeros(2)
    nq = 0
    for i in val[:n_seqs]:
        r = recs[i]
        moves = games[r["gi"]]["moves"].tolist()
        lab = char_labels(moves)
        enc = tok(r["text"], return_offsets_mapping=True,
                  add_special_tokens=False)
        toklab = [lab[e - 1] for (s, e) in enc["offset_mapping"]]
        ids = r["ids"].long()[None].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = backbone(input_ids=ids, output_hidden_states=False,
                         output_attentions=True, use_cache=False)
        att = torch.stack([a[0].float().cpu() for a in o.attentions])
        # (NB, 16, T, T)
        mt = r["mt"].tolist()
        for t in range(min_t, len(mt)):
            q = mt[t]
            aq = att[:, :, q, :]           # (NB, 16, T)
            am = aq.mean(dim=1)            # (NB, T) head-avg
            for k in range(q + 1):
                cls, rec = toklab[k]
                c = ("sink" if k == 0 else
                     "self" if k == q else cls)
                cls_mass[:, CLS.index(c)] += am[:, k]
                if rec >= 0:
                    dt = t - rec
                    d = ("own" if dt == 0 else "prev" if dt == 1 else
                         "d2" if dt == 2 else "d3_5" if dt <= 5 else
                         "d6_10" if dt <= 10 else "d11p")
                    dl_mass[:, DELTA.index(d)] += am[:, k]
                    if dt == 1:
                        head_prev += aq[:, :, k]
            # adjacency: mass per past record, split by hex adjacency
            qx, qy = moves[t]
            for s in range(t):
                sx, sy = moves[s]
                a01 = 0 if (sx - qx, sy - qy) in HEX_NBRS else 1
                ktoks = [k for k in range(q) if toklab[k][1] == s]
                adj[:, a01] += am[:, ktoks].sum(dim=1)
                adj_n[a01] += 1
            nq += 1
    res = {"n_queries": nq, "classes": CLS,
           "cls_mass": (cls_mass / nq).tolist(),
           "delta_bins": DELTA, "delta_mass": (dl_mass / nq).tolist(),
           "head_prev_mass": (head_prev / nq).tolist(),
           "adj_mass_per_record":
               (adj / adj_n[None, :].clamp(min=1)).tolist(),
           "adj_counts": adj_n.tolist()}
    Path(out).write_text(json.dumps(res))
    cm, dm = cls_mass / nq, dl_mass / nq
    print("\nlayer | " + " ".join(f"{c:>6}" for c in CLS)
          + " || " + " ".join(f"{d:>6}" for d in DELTA) + " || adj/rec non/rec")
    ar = adj / adj_n[None, :].clamp(min=1)
    for j in range(NB):
        print(f"L{j:2d}   | " + " ".join(f"{cm[j, c]:.4f}" for c in
                                         range(len(CLS)))
              + " || " + " ".join(f"{dm[j, d]:.4f}" for d in
                                  range(len(DELTA)))
              + f" || {ar[j, 0]:.5f} {ar[j, 1]:.5f}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/armF_movesfull_r3ext/best.pt")
    ap.add_argument("--mode", default="all",
                    choices=["ablate", "patterns", "all"])
    args = ap.parse_args()
    tok, games, recs, val, cnn, backbone, ads, mu, sd = load_all(args.ckpt)
    if args.mode in ("patterns", "all"):
        run_patterns(backbone, tok, games, recs, val,
                     "armF/results/attn_patterns.json")
    if args.mode in ("ablate", "all"):
        run_ablate(backbone, ads, cnn, games, recs, val, mu, sd,
                   "armF/results/attn_ablate.json")


if __name__ == "__main__":
    main()
