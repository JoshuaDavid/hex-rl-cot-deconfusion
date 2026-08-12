"""Per-layer PCA basis of the full 7744-dim activation maps (top-128), for the
r2 moves-format PCA-summary supervision stream. Whitened target coords:
v = ((flat_map - mean) @ V.T) / sqrt(eig).  Saves armF/results/pca_basis.pt."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

DEV = "cuda"
N = 10000
M = 128


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
    V = torch.empty(19, M, 7744)
    eig = torch.empty(19, M)
    mean = torch.empty(19, 7744)
    for l in range(19):
        x = X[l].to(DEV)
        mu = x.mean(0)
        x = x - mu
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        V[l] = vh[:M].cpu()
        eig[l] = (s[:M] ** 2 / N).cpu()
        mean[l] = mu.cpu()
        var_frac = (s[:M] ** 2).sum() / (s ** 2).sum()
        print(f"z{l:2d}: top-{M} var frac {var_frac:.4f}", flush=True)
    torch.save({"V": V, "eig": eig, "mean": mean, "m": M},
               "armF/results/pca_basis.pt")
    print("wrote armF/results/pca_basis.pt")


if __name__ == "__main__":
    main()
