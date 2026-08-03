#!/usr/bin/env bash
# TMLR paper experiments — CATEGORICAL pipeline (the published artifact:
# GitHub/HF agent_bc.pt + agent_imagination_rl.pt were stripped from these dirs).
#
#   Stage 0: n=500 closed-loop eval of the PUBLISHED categorical pipeline
#            (phase3 vs bc vs random) -> the paper's headline table.  ~3.2 h
#   Stages 1-3: Phase-3 training-seed study — retrain imagination-RL with
#            seeds 1,2,3 from the SAME categorical Phase-2, then evaluate
#            each at n=200 (eval seeds 0-199, pairs with stage-0 BC).
#                                                    ~7-8 h train + 35 min each
#
# NOTE the Gaussian-pipeline n=500 (evaluation/tmlr-n500-stoch) stays as an
# appendix robustness result; it used checkpoints-phase2-rerun-optionA +
# checkpoints-phase3-imagination. THIS script is the paper's main pipeline.
#
# IDEMPOTENT at stage granularity: completed stages (existing summary.json /
# epoch_15.pt) are skipped, so re-launching after an interruption resumes.
#
# Usage:  nohup ./run_seed_study.sh > seed_study.log 2>&1 &
set -u

PY=${PY:-~/.conda/envs/dreamer_v4/bin/python}   # override on RunPod: PY=python3 ./run_seed_study.sh
P2=checkpoints-phase2-categorical/dynamics/final.pt
P3=checkpoints-phase3-categorical/dynamics/final.pt
TOK=checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt
NPZ=ball_in_cup_catch.npz

# ── Stage 0: headline n=500 on the published categorical pipeline ──
OUT0=evaluation/tmlr-n500-categorical
if [ "${SKIP_STAGE0:-0}" = "1" ]; then
  echo "=== [stage 0] SKIP_STAGE0=1 — skipping (runs on the laptop)"
elif [ ! -f "$OUT0/summary.json" ]; then
  echo "=== [stage 0] categorical n=500 eval (phase3,bc,random) — $(date) ==="
  MUJOCO_GL=egl $PY -u -m dynamics.evaluate_env \
    --phase2-ckpt "$P2" --phase3-ckpt "$P3" --tokenizer-ckpt "$TOK" \
    --task ball_in_cup_catch --action-dim 2 \
    --seed-base 0 --num-episodes 500 --policies phase3,bc,random \
    --device cuda --readout sample --wandb-disabled --max-gifs 0 \
    --output-dir "$OUT0" > n500_categorical.log 2>&1 \
    || { echo "[stage 0] EVAL FAILED — see n500_categorical.log"; exit 1; }
else
  echo "=== [stage 0] categorical n=500 summary exists — skipping"
fi

# ── Stages 1-3: Phase-3 seed study on the categorical pipeline ──
for SEED in ${SEEDS:-1 2 3}; do   # override per pod: SEEDS="2" ./run_seed_study.sh
  CKDIR=checkpoints-phase3-cat-seed${SEED}
  CKPT=${CKDIR}/epoch_15.pt

  if [ ! -f "$CKPT" ]; then
    echo "=== [seed ${SEED}] training (~7-8h on laptop 5070 Ti) — $(date) ==="
    $PY -u -m imagination.train_imagination \
      --phase2-ckpt "$P2" --tokenizer-ckpt "$TOK" \
      --dataset offline --dataset-path "$NPZ" --task ball_in_cup_catch \
      --action-dim 2 --epochs 15 --batch-size 48 --steps-per-epoch 200 \
      --checkpoint-dir "$CKDIR" --seed "$SEED" --wandb-disabled \
      > "seed${SEED}_train.log" 2>&1 \
      || { echo "[seed ${SEED}] TRAIN FAILED — see seed${SEED}_train.log"; continue; }
    if grep -q "Could not load policy_head" "seed${SEED}_train.log"; then
      echo "[seed ${SEED}] ABORT: policy_head failed to load from Phase-2 (BC init missing)."
      continue
    fi
  else
    echo "=== [seed ${SEED}] train checkpoint exists — skipping train"
  fi

  OUT=evaluation/tmlr-cat-seed${SEED}-n200
  if [ ! -f "$OUT/summary.json" ]; then
    echo "=== [seed ${SEED}] evaluating n=200 (~35min) — $(date) ==="
    MUJOCO_GL=egl $PY -u -m dynamics.evaluate_env \
      --phase2-ckpt "$P2" --phase3-ckpt "$CKPT" --tokenizer-ckpt "$TOK" \
      --task ball_in_cup_catch --action-dim 2 \
      --seed-base 0 --num-episodes 200 --policies phase3 \
      --device cuda --readout sample --wandb-disabled --max-gifs 0 \
      --output-dir "$OUT" > "seed${SEED}_eval.log" 2>&1 \
      || echo "[seed ${SEED}] EVAL FAILED — see seed${SEED}_eval.log"
  else
    echo "=== [seed ${SEED}] eval summary exists — skipping eval"
  fi
done
echo "=== all experiments complete — $(date) ==="
