"""C3: train logistic-regression probes per (layer, concept) on stored activations.

/venv/main/bin/python scripts/train_probes.py results/probes/<tag>.npz
Reports AUC (5-fold CV) per layer x concept + shuffled-label control.
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
    path = sys.argv[1]
    d = np.load(path, allow_pickle=True)
    acts, labels, keys, layers = d["acts"], d["labels"], list(d["label_keys"]), d["layers"]
    rng = np.random.RandomState(0)

    print(f"{path}: acts {acts.shape}")
    header = ["concept"] + [f"L{int(l)}" for l in layers] + ["shuf(maxL)"]
    print("  ".join(f"{h:>18}" for h in header))
    for ki, key in enumerate(keys):
        y = labels[:, ki]
        mask = ~np.isnan(y)
        y = y[mask]
        # binarize continuous labels at median
        if len(set(y.tolist())) > 2:
            y = (y > np.median(y)).astype(float)
        if y.std() == 0 or min((y == 1).sum(), (y == 0).sum()) < 20:
            continue
        row = [str(key)]
        aucs = []
        for j in range(acts.shape[1]):
            X = acts[mask, j].astype(np.float32)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
            aucs.append(auc)
            row.append(f"{auc:.3f}")
        # shuffled control at best layer
        jbest = int(np.argmax(aucs))
        X = acts[mask, jbest].astype(np.float32)
        ys = y.copy()
        rng.shuffle(ys)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        auc_s = cross_val_score(clf, X, ys, cv=5, scoring="roc_auc").mean()
        row.append(f"{auc_s:.3f}")
        print("  ".join(f"{c:>18}" for c in row))


if __name__ == "__main__":
    main()
