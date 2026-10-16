#!/bin/bash
# Phase-2 finetunes on the corrected pipeline. Command = run_bc_seeds.sh (the first build's categorical pipeline),
# base = new Phase-1 epoch 320 (aligned data + Eq.7 single-term loss), data = aligned npz, seeds 10..14, wandb LIVE.
# Draws 15 to 17 ran this same script with SEEDS set to their seeds.
set -u
cd /workspace/dreamer_v4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DIR=/workspace/wandb_logs PYTHONUNBUFFERED=1 MUJOCO_GL=egl
mkdir -p "$WANDB_DIR"
BASE=checkpoints-dynamics-phase1-aligned-joint/dynamics/dynamics_epoch_320.pt
TOK=checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt
NPZ=ball_in_cup_catch_aligned.npz
SEEDS="${SEEDS:-10 11 12 13 14}"
for SEED in $SEEDS; do
  CKDIR=checkpoints-phase2-aligned-seed${SEED}
  [ -f "$CKDIR/final.pt" ] && { echo "=== seed $SEED done already, skipping"; continue; }
  echo "=== [phase2 seed $SEED] start $(date -u +%FT%TZ)"
  python -u -m dynamics.train_agent \
    --dynamics-ckpt "$BASE" --tokenizer-ckpt "$TOK" \
    --dataset offline --dataset-path "$NPZ" --task ball_in_cup_catch \
    --action-dim 2 --policy-type categorical \
    --epochs 40 --batch-size 32 --steps-per-epoch 500 --amp \
    --checkpoint-dir "$CKDIR" --seed "$SEED" \
    --wandb-project dreamer-v4-agent --wandb-name "phase2_ALIGNED_jointnorm_seed${SEED}" \
    > "phase2_seed${SEED}_train.log" 2>&1 && echo "=== [phase2 seed $SEED] finished $(date -u +%FT%TZ)" || echo "=== [phase2 seed $SEED] FAILED $(date -u +%FT%TZ)"
done
echo "ALL PHASE-2 RUNS ATTEMPTED $(date -u +%FT%TZ)" > /workspace/phase2_done.txt
