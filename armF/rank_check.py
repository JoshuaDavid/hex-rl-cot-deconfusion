"""Effective dimensionality of HexHex activation maps: for each layer, treat the
full map (121*64=7744 dims) as one vector over positions; SVD over N=10000
positions. Reports rank@90%/99% variance, participation ratio, and variance
captured by the top 2048 dims (one-Qwen-token affine readout bound)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

DEV = "cuda"
N = 10000


@torch.no_grad()
def main():
    cnn = W.load_model()
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"][:N]
    X = [torch.empty(N, 7744) for _ in range(19)]
    for i in range(0, N, 250):
        bb = boards[i:i+250].float().to(DEV)
        acts = W.dump_acts(cnn, bb)
        for l, a in enumerate(acts):
            X[l][i:i+len(bb)] = a.reshape(len(bb), -1).cpu()
        if (i + 250) % 2500 == 0:
            print(f"{i+250}/{N}", flush=True)
    out = {}
    for l in range(19):
        x = X[l].to(DEV)
        x = x - x.mean(0)
        ev = torch.linalg.svdvals(x).square().cpu()  # variance spectrum
        tot = ev.sum()
        cum = ev.cumsum(0) / tot
        r90 = int((cum < 0.90).sum().item()) + 1
        r99 = int((cum < 0.99).sum().item()) + 1
        pr = (tot ** 2 / (ev ** 2).sum()).item()
        top2048 = cum[2047].item()
        out[l] = {"rank90": r90, "rank99": r99, "pr": round(pr, 1),
                  "var_top2048": round(top2048, 4)}
        print(f"z{l:2d}: rank90 {r90:5d} rank99 {r99:5d} PR {pr:8.1f} "
              f"var@2048 {top2048:.4f}", flush=True)
    Path("armF/results/rank_check.json").write_text(json.dumps(out, indent=1))
    print("wrote armF/results/rank_check.json")


if __name__ == "__main__":
    main()
