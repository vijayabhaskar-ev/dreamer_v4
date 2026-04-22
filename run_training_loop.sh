#!/bin/bash
# TODO(REMOVE-WHEN-LEAK-FIXED): Operational wrapper for the torch_xla 2.5
# MpDeviceLoader thread leak.  Pairs with the AUTO-EXIT block in
# dynamics/trainer.py fit().  Relaunches training after each clean 4h45m
# exit so the thread leak never reaches the ~5h crash cliff.
#
# NOT NEEDED ON CUDA: the leak is XLA-specific.  If you're training on
# CUDA, just run `python -m dynamics.train_dynamics ...` directly — this
# wrapper adds no value there.
#
# Delete this script when the trainer.py AUTO-EXIT block is removed
# (i.e. after upgrading to a torch_xla version without the leak).
#
# Usage:
#   chmod +x run_training_loop.sh
#   ./run_training_loop.sh
# Run inside tmux/screen so it survives SSH disconnects.

set -u  # fail on unset variables

MAX_RESTARTS=200
CKPT_DIR="checkpoints/dynamics"
TOKENIZER_CKPT="checkpoints/tokenizer/final.pt"
DATASET_PATH="cheetah_run.npz"

for i in $(seq 1 $MAX_RESTARTS); do
    echo "==> Training cycle $i at $(date)"

    # Pick the newest checkpoint by epoch number (sort -V handles
    # embedded integers correctly; plain sort would put epoch_100
    # before epoch_20 lexicographically).
    latest_ckpt=$(ls -1 "$CKPT_DIR"/dynamics_epoch_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$latest_ckpt" ]; then
        echo "    No checkpoint yet — starting fresh"
        resume_arg=""
    else
        echo "    Resuming from $latest_ckpt"
        resume_arg="--resume-from $latest_ckpt"
    fi

    python -m dynamics.train_dynamics \
        --tokenizer-ckpt "$TOKENIZER_CKPT" \
        --dataset offline --dataset-path "$DATASET_PATH" \
        --batch-size 32 --steps-per-epoch 200 --epochs 300 \
        --device tpu --lr 4e-4 \
        --seq-len-short 8 --seq-len-long 16 --long-batch-ratio 0.15 \
        --num-workers 0 --checkpoint-interval 20 \
        --wandb-project dreamer-v4-dynamics \
        --enable-mp-device-loader \
        $resume_arg
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "==> Cycle $i exited cleanly (AUTO-EXIT) at $(date).  Resuming in 60s..."
        sleep 60
    else
        echo "==> Cycle $i crashed with exit code $exit_code at $(date)."
        echo "    Not a clean auto-exit — stopping the loop for investigation."
        echo "    (Fail-loud: we do NOT blindly retry real crashes.)"
        exit $exit_code
    fi
done

echo "==> Reached MAX_RESTARTS ($MAX_RESTARTS cycles).  Training finished or give up."
