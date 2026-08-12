"""Pre-tokenize r2 moves-format sequences. For each game, sample --cuts random
cut plies (uniform over [2, T], distinct); one sequence per (game, cut).
Saves armF/data/tokens_moves.pt with padded ids, move-token idx, cell idx."""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render_moves as RM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]

    recs = []
    for gi, g in enumerate(games):
        T = len(g["moves"])
        cuts = random.sample(range(2, T + 1), min(args.cuts, T - 1))
        for cut in cuts:
            text, spans, cell_off = RM.render_moves(
                g["moves"][:cut], g["boards"][cut - 1].float())
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            mt = RM.move_token_indices(enc["offset_mapping"], spans)
            ct = RM.cell_token_indices(enc["offset_mapping"], cell_off)
            recs.append({"game": gi, "cut": cut,
                         "ids": torch.tensor(enc["input_ids"], dtype=torch.int32),
                         "mt": torch.tensor(mt, dtype=torch.int16),
                         "ct": torch.tensor(ct, dtype=torch.int16)})
        if (gi + 1) % 200 == 0:
            print(f"{gi+1}/{len(games)} games, {len(recs)} seqs", flush=True)

    N = len(recs)
    L = max(len(r["ids"]) for r in recs)
    Tm = max(len(r["mt"]) for r in recs)
    ids = torch.full((N, L), tok.pad_token_id, dtype=torch.int32)
    mt = torch.full((N, Tm), -1, dtype=torch.int16)
    ct = torch.empty(N, 121, dtype=torch.int16)
    lens = torch.empty(N, dtype=torch.int32)
    cuts = torch.empty(N, dtype=torch.int32)
    game_id = torch.empty(N, dtype=torch.int32)
    for i, r in enumerate(recs):
        ids[i, :len(r["ids"])] = r["ids"]
        mt[i, :len(r["mt"])] = r["mt"]
        ct[i] = r["ct"]
        lens[i] = len(r["ids"])
        cuts[i] = r["cut"]
        game_id[i] = r["game"]
    torch.save({"input_ids": ids, "move_tok": mt, "cell_idx": ct, "lens": lens,
                "cuts": cuts, "game_id": game_id}, "armF/data/tokens_moves.pt")
    print(f"wrote armF/data/tokens_moves.pt: {N} seqs, maxlen {L}, maxcut {Tm}")


if __name__ == "__main__":
    main()
