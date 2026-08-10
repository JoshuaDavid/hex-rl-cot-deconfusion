"""R4 RL dataset: select-format prompts for the single-token selection RL.
prompt = select_prompt(board); reward_model.ground_truth = gt (board recon for
in-loop grading). Reward is computed inside hex_select_loop (reward_score set),
so no reward-fn task handler is needed. Boards from pool_train (disjoint from
pool_test used for the differential eval).

Output: data/arme/rl/{train,val}.parquet
Run: /venv/main/bin/python scripts/build_arme_r4_data.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hexenv.arme import board_from_gt, select_prompt

N_TRAIN = 4000
N_VAL = 128


def main():
    os.makedirs("data/arme/rl", exist_ok=True)
    gts = [json.loads(l) for l in open("data/arme/pool_train.jsonl")]
    rows = []
    for gt in gts[: N_TRAIN + N_VAL]:
        b = board_from_gt(gt)
        rows.append({
            "data_source": "hex_arme_select",
            "prompt": [{"role": "user", "content": select_prompt(b)}],
            "ability": "hex_select",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"category": "arme_select", "size": gt["size"]},
        })
    pd.DataFrame(rows[N_VAL:]).to_parquet("data/arme/rl/train.parquet")
    pd.DataFrame(rows[:N_VAL]).to_parquet("data/arme/rl/val.parquet")
    print(f"R4 RL: train {len(rows)-N_VAL} val {N_VAL}")


if __name__ == "__main__":
    main()
