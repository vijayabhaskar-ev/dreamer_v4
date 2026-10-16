#!/bin/bash
# Phase-3 imagination training + real-env evaluation, one (draw, imagination-seed) job at a time.
# JOBS="10:1 10:2 10:3"  (Phase-2 draw : imagination seed). Command = run_seed_study.sh verbatim (epochs 15, batch 48,
# 200 steps/epoch), wandb LIVE; eval = evaluate_env on boards 0-499. The evaluations
# reported in the paper were rerun with --policies phase3 (the opts in each
# evaluation/tmlr-aligned-p3-*/summary.json); the BC parents were evaluated separately.
set -u
cd /workspace/dreamer_v4
export WANDB_DIR=/workspace/wandb_logs PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR"
TOK=checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt
NPZ=ball_in_cup_catch_aligned.npz
for J in $JOBS; do
  D=${J%%:*}; S=${J##*:}
  P2=checkpoints-phase2-aligned-seed${D}/dynamics_epoch_040.pt
  CKDIR=checkpoints-phase3-aligned-draw${D}-seed${S}
  OUT=evaluation/tmlr-aligned-p3-draw${D}-seed${S}-n500
  if [ ! -f "$CKDIR/final.pt" ]; then
    echo "=== [p3 draw $D seed $S] train start $(date -u +%FT%TZ)"
    python -u -m imagination.train_imagination \
      --phase2-ckpt "$P2" --tokenizer-ckpt "$TOK" \
      --dataset offline --dataset-path "$NPZ" --task ball_in_cup_catch \
      --action-dim 2 --epochs 15 --batch-size 48 --steps-per-epoch 200 \
      --checkpoint-dir "$CKDIR" --seed "$S" \
      --wandb-project dreamer-v4-imagination --wandb-name "phase3_ALIGNED_draw${D}_seed${S}" \
      > "p3_draw${D}_seed${S}_train.log" 2>&1 && echo "=== [p3 draw $D seed $S] train done $(date -u +%FT%TZ)" \
      || { echo "=== [p3 draw $D seed $S] TRAIN FAILED $(date -u +%FT%TZ)"; continue; }
    grep -q "Could not load policy_head" "p3_draw${D}_seed${S}_train.log" && { echo "=== [p3 draw $D seed $S] ABORT: policy_head not loaded from Phase-2"; continue; }
  fi
  if [ ! -f "$OUT/summary.json" ]; then
    echo "=== [p3 draw $D seed $S] eval start $(date -u +%FT%TZ)"
    python -u -m dynamics.evaluate_env \
      --phase2-ckpt "$P2" --phase3-ckpt "$CKDIR/final.pt" --tokenizer-ckpt "$TOK" \
      --task ball_in_cup_catch --action-dim 2 \
      --seed-base 0 --num-episodes 500 --policies phase3 \
      --device cuda --readout sample --wandb-disabled --max-gifs 0 \
      --output-dir "$OUT" > "p3_draw${D}_seed${S}_eval.log" 2>&1 && echo "=== [p3 draw $D seed $S] eval done $(date -u +%FT%TZ)" \
      || echo "=== [p3 draw $D seed $S] EVAL FAILED $(date -u +%FT%TZ)"
  fi
done
echo "ALL PHASE-3 JOBS ATTEMPTED $(date -u +%FT%TZ)" > /workspace/phase3_done.txt
