"""r2 pre-launch verification: replay games, check stored boards/cells match,
check tokenization readout points decode to the right things, eyeball samples."""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import render_moves as RM  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402

random.seed(0)
games = torch.load("armF/data/games.pt", weights_only=False)["games"]
toks = torch.load("armF/data/tokens_moves.pt", weights_only=False)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# 1) replay 20 random games: stored boards == replayed board_tensor (uint8),
#    stored cell = canonical index of played stone, present in channel 1
for gi in random.sample(range(len(games)), 20):
    g = games[gi]
    b = Board(11, switch_allowed=False)
    for t, mv in enumerate(g["moves"].tolist()):
        b.set_stone(tuple(mv))
        assert torch.equal(b.board_tensor.to(torch.uint8), g["boards"][t]), (gi, t)
        c = int(g["cells"][t])
        cx, cy = divmod(c, 11)
        assert g["boards"][t][1, cx + 1, cy + 1] == 1, (gi, t, "played stone not in ch1")
print("replay check: 20 games OK (boards, cells, channel semantics)")

# 2) column indexing identity: reshape(M,64,121)[..., c] == a[:, :, cx, cy]
a = torch.randn(3, 64, 11, 11)
c = 5 * 11 + 7
assert torch.equal(a.reshape(3, 64, 121)[:, :, c], a[:, :, 5, 7])
print("column reshape indexing OK")

# 3) PCA round trip: relative recon error ~= 1 - var_frac
pca = torch.load("armF/results/pca_basis.pt", weights_only=False)
bb = torch.stack([games[0]["boards"][i].float() for i in range(8)]).cuda()
cnn = W.load_model()
acts = W.dump_acts(cnn, bb)
for l in (0, 9, 18):
    fm = acts[l].reshape(8, -1).cpu() - pca["mean"][l]
    co = fm @ pca["V"][l].T
    rec = co @ pca["V"][l]
    err = ((fm - rec) ** 2).sum() / (fm ** 2).sum()
    print(f"  z{l}: pca-128 recon residual frac {err:.3f}")

# 4) token readout decoding on 3 random sequences (eyeball)
for i in random.sample(range(len(toks["lens"])), 3):
    gi, cut = int(toks["game_id"][i]), int(toks["cuts"][i])
    g = games[gi]
    ids = toks["input_ids"][i][: toks["lens"][i]].tolist()
    mt = toks["move_tok"][i]
    mt = mt[mt >= 0].tolist()
    assert len(mt) == cut
    decoded_moves = [tok.decode([ids[j]]) for j in mt]
    expect_last = [RM.move_str(mv)[-1] for mv in g["moves"][:cut].tolist()]
    assert [d.strip()[-1] for d in decoded_moves] == expect_last, (i, decoded_moves)
    board = g["boards"][cut - 1]
    syms = [tok.decode([ids[j]]).strip() for j in toks["cell_idx"][i].tolist()]
    for cell in range(121):
        x, y = divmod(cell, 11)
        want = ("X" if board[0, x + 1, y + 1] == 1
                else "O" if board[1, x + 1, y + 1] == 1 else ".")
        assert syms[cell] == want, (i, cell, syms[cell], want)
    print(f"seq {i} (game {gi} cut {cut}): move tokens + 121 render cells OK")
    print(tok.decode(ids))
print("ALL CHECKS PASSED")
