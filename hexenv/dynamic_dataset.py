"""Dynamic curriculum dataset for verl (arm C).

Loaded via data.custom_cls. Presents a virtual-length dataset; each __getitem__
draws a category according to data/curriculum/weights.json (hot-reloaded on
mtime change), then a row uniformly within the category. Categories are the
*.parquet files in data/curriculum/; files appearing mid-run are loaded and
tokenized on the next refresh; categories absent from weights.json (or weight 0)
are never sampled — removal is instant and needs no restart.
"""

from __future__ import annotations

import json
import os
import random

from torch.utils.data import Dataset

from verl.utils.dataset.rl_dataset import RLHFDataset

CURRICULUM_DIR = os.environ.get(
    "HEX_CURRICULUM_DIR", "/workspace/hex-rl-cot-deconfusion/data/curriculum")
WEIGHTS_PATH = os.path.join(CURRICULUM_DIR, "weights.json")
VIRTUAL_LEN = 1_000_000


class DynamicCurriculumDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None,
                 max_samples: int = -1):
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
        # verl instantiates this class for BOTH train and val splits. The
        # dynamic-mixture behavior is train-only; val files delegate to the
        # stock dataset (finite length, fixed contents).
        files = data_files if isinstance(data_files, list) else [data_files]
        if any("val" in os.path.basename(str(f)) for f in files):
            self._delegate = RLHFDataset(data_files, tokenizer, config, processor)
            print(f"[curriculum] val split -> stock dataset "
                  f"({len(self._delegate)} rows)", flush=True)
            return
        self._delegate = None
        self._cats: dict[str, RLHFDataset] = {}
        self._weights: dict[str, float] = {}
        self._weights_mtime = -1.0
        self._known_files: set[str] = set()
        self._refresh(force=True)

    def _refresh(self, force=False):
        try:
            mtime = os.path.getmtime(WEIGHTS_PATH)
        except OSError:
            mtime = -1.0
        if not force and mtime == self._weights_mtime:
            return
        self._weights_mtime = mtime
        # load any new category shards
        for fn in sorted(os.listdir(CURRICULUM_DIR)):
            if not fn.endswith(".parquet"):
                continue
            path = os.path.join(CURRICULUM_DIR, fn)
            if path in self._known_files:
                continue
            cat = fn[:-len(".parquet")]
            try:
                self._cats[cat] = RLHFDataset([path], self.tokenizer,
                                              self.config, self.processor)
                self._known_files.add(path)
                print(f"[curriculum] loaded category '{cat}' "
                      f"({len(self._cats[cat])} rows)", flush=True)
            except Exception as e:  # never kill training on a bad shard
                print(f"[curriculum] FAILED to load {fn}: {e}", flush=True)
        # load weights
        try:
            with open(WEIGHTS_PATH) as f:
                w = json.load(f)
        except Exception:
            w = {}
        weights = {c: float(w.get(c, 0.0)) for c in self._cats}
        if all(v <= 0 for v in weights.values()):
            weights = {c: 1.0 for c in self._cats}  # fail-open: uniform
        changed = weights != self._weights
        self._weights = weights
        if not changed:
            return
        print(f"[curriculum] active weights: "
              f"{ {c: round(v, 3) for c, v in weights.items() if v > 0} }",
              flush=True)

    def __len__(self):
        if self._delegate is not None:
            return len(self._delegate)
        return VIRTUAL_LEN

    def __getitem__(self, item):
        if self._delegate is not None:
            return self._delegate[item]
        self._refresh()
        rng = random.Random(hash(("hexC", item)))
        cats = [c for c, v in self._weights.items() if v > 0 and len(self._cats[c])]
        wts = [self._weights[c] for c in cats]
        cat = rng.choices(cats, weights=wts, k=1)[0]
        ds = self._cats[cat]
        return ds[rng.randrange(len(ds))]

    def resume_dataset_state(self):
        pass
