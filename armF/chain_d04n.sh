#!/bin/bash
# Sequential layer-by-layer containment of the full distilled net (Joshua
# 2026-08-13 overnight). Stage l trains ONLY transition block 4+l + adapter
# (freeze-below 4+l); frozen bottom preserves all prior layers exactly.
# Disk: strip backbone from checkpoints two stages back (adapters kept).
set -e
cd /workspace/hex-rl-cot-deconfusion
set -a && . ./.env && set +a
prev=checkpoints/armF_movesc0_d04n/final.pt
prevprev=""
for l in $(seq 1 18); do
  echo "=== stage c$l start $(date +%H:%M) ==="
  /venv/main/bin/python armF/train_movesc0.py --fmt d04n --layer $l \
    --freeze-below $((4 + l)) --init-ckpt $prev --ridge-init 1200 \
    --lr 3e-4 --adapter-lr 1e-3 --warmup 500 --steps 10000 \
    --eval-every 500 --early-stop-window 4 --early-stop-delta 0.01 \
    --run-name armF_chain_c$l > /tmp/chain_c$l.log 2>&1
  cur=checkpoints/armF_chain_c$l/final.pt
  if [ ! -f $cur ]; then
    echo "STAGE c$l FAILED"
    tail -20 /tmp/chain_c$l.log
    exit 1
  fi
  grep -E "val c$l" /tmp/chain_c$l.log | tail -1
  if [ -n "$prevprev" ] && [ "$prevprev" != "checkpoints/armF_movesc0_d04n/final.pt" ]; then
    /venv/main/bin/python -c "
import torch, sys
p = sys.argv[1]
ck = torch.load(p, map_location='cpu', weights_only=False)
if 'backbone' in ck:
    del ck['backbone']
    torch.save(ck, p)
    print('stripped backbone from', p)" $prevprev
  fi
  prevprev=$prev
  prev=$cur
done
echo "CHAIN COMPLETE $(date +%H:%M)"
