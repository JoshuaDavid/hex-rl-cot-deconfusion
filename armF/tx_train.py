"""Finger T: from-scratch encoder transformer (<=18L, d=1024) distilled from
the original HexHex CNN; goal = competitive with the rank-1024 distilled CNN.

Model: 121 cell tokens (empty/own/opp embedding + learned pos emb), pre-RMSNorm
blocks with 16-head bidirectional attention + per-layer per-head learned 2D
relative-position bias (441 buckets), SwiGLU MLP, final norm, per-cell scalar
policy head. Loss: KL(teacher || student) on illegal-masked logits; teacher
logits precomputed by tx_gen_data.py (rotation-averaged + masked).

WSD schedule (warmup / constant / linear anneal over last frac). Optional
DAgger refreshes: student self-play positions, teacher-labeled, appended to
the buffer mid-run. wandb project hex-rl-cot-deconfusion.

Usage: /venv/main/bin/python armF/tx_train.py --steps 60000 --tag txT18
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
from tx_gen_data import masked_teacher_logits  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
N_CELL = 121
N_REG = 7  # pad to 128 tokens: SDPA-with-bias needs alignment; registers are free
T_SEQ = N_CELL + N_REG


def rel_index():
    """(128,128) long: bucket (dr+10)*21 + (dc+10); any register pair -> 441."""
    r = torch.arange(N_CELL) // 11
    c = torch.arange(N_CELL) % 11
    dr = r[:, None] - r[None, :] + 10
    dc = c[:, None] - c[None, :] + 10
    idx = torch.full((T_SEQ, T_SEQ), 441, dtype=torch.long)
    idx[:N_CELL, :N_CELL] = dr * 21 + dc
    return idx


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Block(nn.Module):
    def __init__(self, d, heads, d_inter):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads, 442))
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, d_inter, bias=False)
        self.up = nn.Linear(d, d_inter, bias=False)
        self.down = nn.Linear(d_inter, d, bias=False)

    def forward(self, x, rel):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        ab = self.bias[:, rel].unsqueeze(0)  # (1,h,121,121)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=ab)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, D))
        n = self.n2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class TxPolicy(nn.Module):
    def __init__(self, layers=18, d=1024, heads=16, d_inter=2752):
        super().__init__()
        self.cfg = {"layers": layers, "d": d, "heads": heads, "d_inter": d_inter}
        self.emb = nn.Embedding(3, d)
        self.pos = nn.Embedding(N_CELL, d)
        self.reg = nn.Parameter(torch.randn(N_REG, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads, d_inter) for _ in range(layers))
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, 1)
        self.use_ckpt = False
        self.register_buffer("rel", rel_index(), persistent=False)

    def forward(self, states):
        """states: (B,121) long in {0 empty, 1 own, 2 opp} -> (B,121) logits."""
        x = self.emb(states) + self.pos.weight.unsqueeze(0)
        x = torch.cat([x, self.reg.unsqueeze(0).expand(len(x), -1, -1)], 1)
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, self.rel, use_reentrant=False)
            else:
                x = blk(x, self.rel)
        return self.head(self.norm_f(x[:, :N_CELL])).squeeze(-1)


def boards_to_states(boards_u8):
    """(B,2,13,13) uint8 on any device -> (B,121) long."""
    inner = boards_u8[:, :, 1:-1, 1:-1].long()
    return (inner[:, 0] + 2 * inner[:, 1]).reshape(-1, N_CELL)


def occ_of(states):
    return (states > 0).float()


@torch.no_grad()
def student_logits(model, boards_u8, rotavg=True):
    """Masked logits, optional 180-degree rotation averaging (teacher-parity)."""
    st = boards_to_states(boards_u8)
    lg = model(st)
    if rotavg:
        bf = torch.flip(boards_u8, [2, 3])
        lg = (lg + torch.flip(model(boards_to_states(bf)), [1])) / 2
    return lg - 1000.0 * occ_of(st)


# ---------------------------------------------------------------- play eval
def make_tx_player(model):
    @torch.no_grad()
    def fn(b):
        x = b.board_tensor.unsqueeze(0).to(torch.uint8).to(DEV)
        lg = student_logits(model, x)[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def make_cnn_logit_player(logit_fn):
    @torch.no_grad()
    def fn(b):
        x = b.board_tensor.unsqueeze(0).float().to(DEV)
        lg = logit_fn(x)[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def play_paired_1ply(fn_a, fn_b, n_openings, seed=123):
    """Paired single-move openings, both colors; returns a's wins / 2n games."""
    rng = random.Random(seed)
    wins = 0
    for g in range(n_openings):
        op = divmod(rng.randrange(121), 11)
        for a_is in (0, 1):
            b = Board(11, switch_allowed=False)
            b.set_stone(op)
            while not b.winner and b.legal_moves:
                fn = fn_a if b.player == a_is else fn_b
                b.set_stone(fn(b))
            if b.winner == [a_is]:
                wins += 1
    return wins


# ---------------------------------------------------------------- dagger
def gen_student_positions(model, cnn, target, par, rng):
    """Student self-play (temp 0.3, hybrid random openings k in 0..8);
    returns (boards_u8 cpu, teacher f16 logits cpu)."""
    model.eval()
    active = [{"b": Board(11, switch_allowed=False), "ply": 0,
               "k": rng.randrange(0, 9)} for _ in range(par)]
    out = []
    while len(out) < target:
        for g in active:
            out.append(g["b"].board_tensor.to(torch.uint8).clone())
        X = torch.stack([g["b"].board_tensor for g in active]).to(torch.uint8).to(DEV)
        with torch.no_grad():
            lg = student_logits(model, X, rotavg=False)
        probs = torch.softmax(lg / 0.3, dim=1)
        samp = torch.multinomial(probs, 1).squeeze(1).cpu()
        for j, g in enumerate(active):
            if g["ply"] < g["k"]:
                mv = rng.choice(sorted(g["b"].legal_moves))
            else:
                p1 = correct_position1d(samp[j].item(), 11, g["b"].player)
                mv = divmod(p1, 11)
                if mv not in g["b"].legal_moves:
                    mv = rng.choice(sorted(g["b"].legal_moves))
            g["b"].set_stone(mv)
            g["ply"] += 1
        for j, g in enumerate(active):
            if g["b"].winner or not g["b"].legal_moves:
                active[j] = {"b": Board(11, switch_allowed=False), "ply": 0,
                             "k": rng.randrange(0, 9)}
    boards = torch.stack(out[:target])
    tl = torch.empty(target, N_CELL, dtype=torch.float16)
    with torch.no_grad():
        for i in range(0, target, 2048):
            x = boards[i:i + 2048].float().to(DEV)
            tl[i:i + 2048] = masked_teacher_logits(cnn, x).half().cpu()
    model.train()
    return boards, tl


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--d-inter", type=int, default=2752)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop training loop after this much wallclock")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--anneal-frac", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--data", default="armF/data/tx_positions.pt")
    ap.add_argument("--teacher", default="armF/data/tx_teacher.pt")
    ap.add_argument("--n-val", type=int, default=8192)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--play-every", type=int, default=5000)
    ap.add_argument("--play-openings", type=int, default=15)
    ap.add_argument("--dagger-at", type=int, nargs="*", default=[])
    ap.add_argument("--dagger-n", type=int, default=150000)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--resume-ckpt", default=None)
    ap.add_argument("--tag", default="txT18")
    ap.add_argument("--ckpt-act", action="store_true",
                    help="activation checkpointing (bigger batch, ~1.3x slower)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.val_every, args.play_every = 60, 20, 50
        args.play_openings, args.warmup = 2, 10
    torch.manual_seed(0)
    rng = random.Random(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    boards = torch.load(args.data, weights_only=False)["boards"]
    tlog = torch.load(args.teacher, weights_only=False)["logits"]
    assert len(boards) == len(tlog)
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards, tlog = boards[perm].to(DEV), tlog[perm].to(DEV)
    n_train = len(boards) - args.n_val
    print(f"data {len(boards)} train {n_train} val {args.n_val}", flush=True)

    model = TxPolicy(layers=args.layers, d_inter=args.d_inter).to(DEV)
    model.use_ckpt = args.ckpt_act
    if args.compile:
        model = torch.compile(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {args.layers}L d1024: {n_par/1e6:.1f}M params", flush=True)
    start_step = 0
    if args.resume_ckpt:
        ck = torch.load(args.resume_ckpt, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start_step = ck.get("step", 0)
        print(f"resumed {args.resume_ckpt} at step {start_step}", flush=True)

    cnn = W.load_model()
    for p in cnn.parameters():
        p.requires_grad_(False)

    use_wandb = not args.no_wandb and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=f"armF_{args.tag}",
                   config={**vars(args), "params_M": n_par / 1e6})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        a0 = int(args.steps * (1 - args.anneal_frac))
        if s < a0:
            return args.lr
        return args.lr * max(0.02, 1 - (s - a0) / max(1, args.steps - a0))

    val_boards = boards[n_train:]
    val_states = boards_to_states(val_boards)
    val_tp = F.softmax(tlog[n_train:].float(), -1)
    val_targ_lp = F.log_softmax(tlog[n_train:].float(), -1)
    val_am = tlog[n_train:].float().argmax(1)

    @torch.no_grad()
    def run_val():
        model.eval()
        kls, t1 = [], 0
        for i in range(0, len(val_states), 2048):
            st = val_states[i:i + 2048]
            lg = model(st) - 1000.0 * occ_of(st)
            lp = F.log_softmax(lg.float(), -1)
            kls.append((val_tp[i:i + 2048]
                        * (val_targ_lp[i:i + 2048] - lp)).sum(-1).mean().item())
            t1 += (lg.argmax(1) == val_am[i:i + 2048]).sum().item()
        model.train()
        return sum(kls) / len(kls), t1 / len(val_states)

    # distilled opponent for periodic play eval
    from elo_temp_distilled import load_distilled, dist_logits
    student_cnn = load_distilled(cnn)
    dist_player = make_cnn_logit_player(
        lambda x: dist_logits(student_cnn, x)
        - 1000.0 * x[:, :, 1:-1, 1:-1].sum(1).reshape(len(x), 121))

    ckdir = Path(f"checkpoints/armF_{args.tag}")
    ckdir.mkdir(parents=True, exist_ok=True)
    best_kl = float("inf")
    model.train()
    t0 = time.time()
    dagger_set = set(args.dagger_at)
    for s in range(start_step, args.steps):
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"max-seconds reached at step {s}", flush=True)
            break
        if s in dagger_set:
            print(f"[{s}] DAgger refresh: {args.dagger_n} student positions",
                  flush=True)
            db, dt = gen_student_positions(model, cnn, args.dagger_n, 1024, rng)
            boards = torch.cat([boards[:n_train], db.to(DEV), boards[n_train:]])
            tlog = torch.cat([tlog[:n_train], dt.to(DEV), tlog[n_train:]])
            n_train += len(db)
            print(f"[{s}] buffer now {n_train} train ({time.time()-t0:.0f}s)",
                  flush=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        idx = torch.randint(0, n_train, (args.batch,), device=DEV)
        st = boards_to_states(boards[idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(st)
        lg = lg.float() - 1000.0 * occ_of(st)
        tp = F.softmax(tlog[idx].float(), -1)
        kl = (tp * (F.log_softmax(tlog[idx].float(), -1)
                    - F.log_softmax(lg, -1))).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        kl.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (s + 1) % 100 == 0 and use_wandb:
            import wandb
            wandb.log({"train_kl": kl.item(), "lr": lr_at(s),
                       "grad_norm": gn.item()}, step=s + 1)
        if (s + 1) % args.val_every == 0 or s + 1 == args.steps:
            vkl, vt1 = run_val()
            sps = (s + 1 - start_step) / (time.time() - t0)
            print(f"step {s+1}/{args.steps} kl {kl.item():.4f} val_kl {vkl:.4f} "
                  f"val_top1 {vt1:.3f} ({sps:.2f} it/s)", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"val_kl": vkl, "val_top1": vt1}, step=s + 1)
            if vkl < best_kl:
                best_kl = vkl
                torch.save({"state_dict": model.state_dict(), "cfg": model.cfg,
                            "step": s + 1, "val_kl": vkl}, ckdir / "best.pt")
        if (s + 1) % args.play_every == 0 or s + 1 == args.steps:
            model.eval()
            w = play_paired_1ply(make_tx_player(model), dist_player,
                                 args.play_openings, seed=123)
            model.train()
            ngames = 2 * args.play_openings
            print(f"step {s+1} play vs distilled: {w}/{ngames}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"play_vs_dist": w / ngames}, step=s + 1)
        if (s + 1) % args.ckpt_every == 0 or s + 1 == args.steps:
            torch.save({"state_dict": model.state_dict(), "cfg": model.cfg,
                        "step": s + 1}, ckdir / "last.pt")

    print(f"done in {(time.time()-t0)/3600:.2f}h; best val_kl {best_kl:.4f}",
          flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
