"""PE15: per-cell write-subspace geometry of the trained rank-1024 bottlenecks.

For each layer l, cell i: B_i = rows of dec weight D_l (7744x1024) for cell
i's 64 channel dims -> orthonormalize -> 64-dim subspace of R^1024.
Overlap(i,j) = ||Q_i^T Q_j||_F^2 / 64 = mean squared cosine of principal
angles (1 = identical subspace, 64/1024 = .0625 = random baseline).
Reports per layer: mean overlap over all pairs, hex-adjacent pairs,
non-adjacent pairs, and the same for the PCA-init decoder (control).

Usage: /venv/main/bin/python armF/fingerE_subspaces.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402

DEV = "cuda"


def adjacency():
    adj = torch.zeros(121, 121, dtype=torch.bool)
    for r in range(11):
        for c in range(11):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 11 and 0 <= cc < 11:
                    adj[r * 11 + c, rr * 11 + cc] = True
    return adj


@torch.no_grad()
def layer_stats(D, adj):
    """D: (7744, 1024) decoder weight. Activation layout is (64ch, 11, 11)
    flattened, so cell i's rows are i + 121*ch for ch in 0..63."""
    idx = torch.arange(64)[:, None] * 121 + torch.arange(121)[None, :]  # (64,121)
    Q = torch.empty(121, 1024, 64, device=DEV)
    for i in range(121):
        Bi = D[idx[:, i]].T  # (1024, 64)
        Q[i], _ = torch.linalg.qr(Bi)
    G = torch.einsum("iak,jal->ijkl", Q, Q)  # (121,121,64,64)
    ov = G.pow(2).sum((-1, -2)) / 64
    off = ~torch.eye(121, dtype=torch.bool, device=DEV)
    a = adj.to(DEV)
    return {"mean": ov[off].mean().item(),
            "adjacent": ov[a].mean().item(),
            "nonadjacent": ov[off & ~a].mean().item()}


@torch.no_grad()
def main():
    torch.manual_seed(0)
    cnn = W.load_model()
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    Vs, mus, _ = B.pca_basis(cnn, boards[:B.N_BASIS])

    ck = torch.load("checkpoints/armF_fingerE/bottleneck_anchored.pt",
                    map_location=DEV, weights_only=False)
    sd = ck["state_dict"]
    adj = adjacency()
    out = {}
    n_adj_wins = 0
    for l in range(19):
        trained = layer_stats(sd[f"bns.{l}.dec.weight"].to(DEV), adj)
        init = layer_stats(Vs[l].T.contiguous(), adj)
        out[l] = {"trained": trained, "pca_init": init}
        win = trained["adjacent"] > trained["nonadjacent"]
        n_adj_wins += win
        print(f"z{l:2d} trained: mean {trained['mean']:.4f} adj "
              f"{trained['adjacent']:.4f} nonadj {trained['nonadjacent']:.4f} "
              f"{'ADJ>' if win else 'adj<='} | pca: mean {init['mean']:.4f} "
              f"adj {init['adjacent']:.4f} nonadj {init['nonadjacent']:.4f}",
              flush=True)
    print(f"\nrandom-subspace baseline overlap = 64/1024 = 0.0625")
    print(f"adjacent > nonadjacent in {n_adj_wins}/19 layers (PE15 needs >=15)")
    out["n_adj_wins"] = n_adj_wins
    Path("armF/results/fingerE_subspaces.json").write_text(
        json.dumps(out, indent=1))
    print("wrote armF/results/fingerE_subspaces.json")


if __name__ == "__main__":
    main()
