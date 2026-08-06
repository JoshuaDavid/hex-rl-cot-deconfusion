"""Arm D SFT dataset: precomputed input_ids + float loss-weight mask.

Rows come from scripts/build_armD_witness.py with all tokenization and
token-importance weights already baked in; this class only tensorizes.
Float weights flow through verl's no_padding sft_loss
(sum(logp*mask)/sum(mask)) as weighted cross-entropy.

ARMD_UNIFORM=1 binarizes the weights (uniform-loss control arm).
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class ArmDWitnessSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        self.dataframe = pd.concat([pd.read_parquet(p) for p in parquet_files],
                                   ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.max_length = config.get("max_length", 2048)
        self.uniform = os.environ.get("ARMD_UNIFORM", "0") == "1"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        input_ids = torch.tensor(list(row["input_ids"]), dtype=torch.long)
        loss_mask = torch.tensor(list(row["loss_mask"]), dtype=torch.float32)
        if self.uniform:
            loss_mask = (loss_mask > 0).float()
        input_ids = input_ids[: self.max_length]
        loss_mask = loss_mask[: self.max_length]
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        return {"input_ids": input_ids, "position_ids": position_ids,
                "loss_mask": loss_mask}
