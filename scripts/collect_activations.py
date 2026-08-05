"""C3: collect residual-stream activations at the last prompt token for a fixed
position set, at several layers. Output: npz with acts [n_pos, n_layers, d] +
labels computed programmatically from the board.

/venv/main/bin/python scripts/collect_activations.py --model <dir> \
    --corpus data/probe_positions.jsonl --out results/probes/<tag>.npz
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, EMPTY, cell_name
from hexenv.prompts import move_prompt


def board_from_record(rec):
    b = Board(rec["size"])
    for color, cell in rec["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def bridges(board, color):
    """Count bridge patterns for color: two same-color stones at the two-bridge
    offset with both carrier cells empty."""
    n = board.size
    count = 0
    # bridge offsets: (dx, dy) pairs reachable via two distinct common neighbors
    offs = [(1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)]
    stones = [(x, y) for y in range(n) for x in range(n) if board.grid[y][x] == color]
    sset = set(stones)
    for (x, y) in stones:
        for dx, dy in offs:
            p2 = (x + dx, y + dy)
            if p2 not in sset or p2 < (x, y):
                continue
            common = set(board.neighbors(x, y)) & set(board.neighbors(*p2))
            if len(common) == 2 and all(board.grid[cy][cx] == EMPTY for cx, cy in common):
                count += 1
    return count


def labels_for(rec):
    b = board_from_record(rec)
    mover = b.to_move
    opp = WHITE if mover == BLACK else BLACK
    lab = {
        "to_move_is_black": float(b.to_move == BLACK),
        "stone_count": float(rec["n_stones"]),
        "mover_wins": float(rec["winner"] == rec["to_move"]),
        "bridges_mover": float(bridges(b, mover)),
        "bridges_opp": float(bridges(b, opp)),
        "has_bridge_mover": float(bridges(b, mover) > 0),
        "p_random_win": len(rec["winning_moves"]) / rec["n_legal"],
        "size": float(rec["size"]),
    }
    # validation-tier: stone color at a few fixed cells (exists-for-sure features)
    for cell in ["c3", "d4", "b2"]:
        x, y = ord(cell[0]) - 97, int(cell[1]) - 1
        if x < b.size and y < b.size:
            lab[f"cell_{cell}_black"] = float(b.grid[y][x] == BLACK)
            lab[f"cell_{cell}_empty"] = float(b.grid[y][x] == EMPTY)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--layers", type=str, default="7,14,21,27")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = [json.loads(l) for l in open(args.corpus)][: args.limit]
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    acts = np.zeros((len(recs), len(layers), model.config.hidden_size), dtype=np.float16)
    all_labels = []
    with torch.no_grad():
        for i, rec in enumerate(recs):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": move_prompt(board_from_record(rec))}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model(**enc, output_hidden_states=True)
            for j, L in enumerate(layers):
                acts[i, j] = out.hidden_states[L][0, -1].float().cpu().numpy().astype(np.float16)
            all_labels.append(labels_for(rec))
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(recs)}", flush=True)

    label_keys = sorted({k for l in all_labels for k in l})
    label_mat = np.array([[l.get(k, np.nan) for k in label_keys] for l in all_labels])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, acts=acts, labels=label_mat,
                        label_keys=np.array(label_keys), layers=np.array(layers))
    print(f"saved {args.out}: acts {acts.shape}, labels {label_mat.shape}")


if __name__ == "__main__":
    main()
