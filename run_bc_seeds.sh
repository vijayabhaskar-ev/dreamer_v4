#!/usr/bin/env bash
# BC (Phase-2) seed study: 3 seeded retrains from the same Phase-1 base + n=500 evals.
# Measures the baseline's own training-seed variability. Fully unattended, idempotent
# at stage granularity; safe to relaunch after interruption.
# Usage:  nohup ./run_bc_seeds.sh > bc_seeds.log 2>&1 &
set -u
PY=${PY:-$HOME/.conda/envs/dreamer_v4/bin/python}
BASE=release/ball_in_cup/world_model.pt
TOK=checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt
NPZ=ball_in_cup_catch.npz

for SEED in 11 12 13; do
  CKDIR=checkpoints-phase2-cat-seed${SEED}
  CKPT=${CKDIR}/final.pt
  if [ ! -f "$CKPT" ]; then
    echo "=== [bc-seed ${SEED}] training (~8-9h laptop) — $(date) ==="
    $PY -u -m dynamics.train_agent \
      --dynamics-ckpt "$BASE" --tokenizer-ckpt "$TOK" \
      --dataset offline --dataset-path "$NPZ" --task ball_in_cup_catch \
      --action-dim 2 --policy-type categorical \
      --epochs 40 --batch-size 32 --steps-per-epoch 500 --amp \
      --checkpoint-dir "$CKDIR" --seed "$SEED" --wandb-disabled \
      > "bc_seed${SEED}_train.log" 2>&1 \
      || { echo "[bc-seed ${SEED}] TRAIN FAILED — see bc_seed${SEED}_train.log"; continue; }
  else
    echo "=== [bc-seed ${SEED}] checkpoint exists — skipping train"
  fi
  OUT=evaluation/tmlr-bc-seed${SEED}-n500
  if [ ! -f "$OUT/summary.json" ]; then
    echo "=== [bc-seed ${SEED}] evaluating n=500 (~85min) — $(date) ==="
    MUJOCO_GL=egl $PY -u -m dynamics.evaluate_env \
      --phase2-ckpt "$CKPT" \
      --phase3-ckpt checkpoints-phase3-categorical/dynamics/final.pt \
      --tokenizer-ckpt "$TOK" \
      --task ball_in_cup_catch --action-dim 2 \
      --seed-base 0 --num-episodes 500 --policies bc \
      --device cuda --readout sample --wandb-disabled --max-gifs 0 \
      --output-dir "$OUT" > "bc_seed${SEED}_eval.log" 2>&1 \
      || echo "[bc-seed ${SEED}] EVAL FAILED — see bc_seed${SEED}_eval.log"
  else
    echo "=== [bc-seed ${SEED}] eval exists — skipping"
  fi
done
echo "=== BC seed study complete — $(date) ==="
