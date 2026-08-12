"""Load the pretrained HexHex 11x11 model and expose per-layer activations.

Capture points (19 total, matching Qwen layers 5..23):
  z_0  = output of the initial 2->64 conv (pre-swish, as in the original forward)
  z_k  = output of skiplayer k-1 for k=1..18 (post-swish residual)
Each z is (B, 64, 11, 11).
"""
import sys
from pathlib import Path

import torch

HEXHEX_ROOT = Path("/workspace/hex-rl-cot-deconfusion/HexHex")
CKPT = HEXHEX_ROOT / "reference_models" / "11_2w4_2000.pt"
sys.path.insert(0, str(HEXHEX_ROOT))

from hexhex.creation.create_model import create_model  # noqa: E402
from hexhex.logic.hexboard import Board  # noqa: E402

BOARD_SIZE = 11
N_LAYERS = 19  # capture points
CHANNELS = 64


def load_model(device="cuda"):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model  # RotationWrapperModel(Conv)


def inner(model):
    return model.internal_model


@torch.no_grad()
def dump_acts(model, x):
    """x: (B, 2, 13, 13) canonical board tensors. Returns list of 19 (B,64,11,11)."""
    m = inner(model)
    acts = []
    h = m.conv(x)
    acts.append(h.clone())
    for sl in m.skiplayers:
        h = sl(h)
        acts.append(h.clone())
    return acts


@torch.no_grad()
def policy_logits(model, x):
    """Full-pipeline logits (rotation-averaged, illegal-masked): (B, 121)."""
    return model(x)


@torch.no_grad()
def stitched_logits(model, z, k):
    """Enter the CNN at capture point k with activations z (B,64,11,11),
    run remaining skiplayers + policy head. k=0 means z is the initial conv
    output and all 18 skiplayers run; k=19 would mean no layers left.
    No rotation averaging, no illegal masking (mask externally)."""
    m = inner(model)
    h = z
    for sl in m.skiplayers[k:] if k > 0 else m.skiplayers:
        h = sl(h)
    return m.policyconv(h).view(-1, BOARD_SIZE ** 2) + m.bias


def empty_board():
    return Board(BOARD_SIZE, switch_allowed=False)


def board_from_moves(moves):
    """moves: list of (x, y) alternating players starting with player 0."""
    b = Board(BOARD_SIZE, switch_allowed=False)
    for mv in moves:
        b.set_stone(mv)
    return b
