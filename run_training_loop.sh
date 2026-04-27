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
MAX_CONSECUTIVE_CRASHES=3   # non-zero exits tolerated before we stop
CKPT_DIR="checkpoints/dynamics"
TOKENIZER_CKPT="checkpoints/tokenizer/final.pt"
DATASET_PATH="cheetah_run.npz"

# Use a single wandb run across all cycles so charts (epoch, global_step,
# losses, host/threads, etc.) show continuous progression across the
# 4h30m auto-exit boundaries instead of appearing as N fragmented runs.
# Generated once per wrapper session; re-running this script starts a
# new wandb run.  train_dynamics.py reads WANDB_RUN_ID and passes it to
# wandb.init(id=..., resume="allow").
if [ -z "${WANDB_RUN_ID:-}" ]; then
    export WANDB_RUN_ID="dreamer_dyn_$(date +%Y%m%d_%H%M%S)"
fi
echo "==> WANDB_RUN_ID=$WANDB_RUN_ID (persists across all cycles below)"

consecutive_crashes=0

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
        --batch-size 32 --steps-per-epoch 200 --epochs 1500 \
        --device tpu --lr 4e-4 \
        --seq-len-short 8 --seq-len-long 16 --long-batch-ratio 0.15 \
        --num-workers 0 --checkpoint-interval 20 \
        --wandb-project dreamer-v4-dynamics \
        --enable-mp-device-loader \
        $resume_arg
    exit_code=$?

    # TODO(REMOVE-WHEN-LEAK-FIXED): check for the auto-exit marker.
    # The python side writes "$CKPT_DIR/.auto_exit_in_progress" the
    # moment it enters the auto-exit block.  If the marker is
    # present, treat ANY exit code as a clean auto-exit cycle —
    # regardless of whether the watchdog fired, xm.rendezvous
    # hung, or the xmp.spawn coordinator SIGTERMed survivors.  The
    # marker is cleaned up here so the next cycle starts fresh.
    marker_path="$CKPT_DIR/.auto_exit_in_progress"
    if [ -f "$marker_path" ]; then
        marker_content=$(cat "$marker_path" 2>/dev/null || echo "unknown")
        echo "==> Auto-exit marker found: $marker_content"
        rm -f "$marker_path"
        echo "==> Cycle $i intended auto-exit (exit code $exit_code treated as clean).  Resuming in 60s..."
        consecutive_crashes=0
        sleep 60
    elif [ $exit_code -eq 0 ]; then
        echo "==> Cycle $i exited cleanly (exit code 0) at $(date).  Resuming in 60s..."
        consecutive_crashes=0
        sleep 60
    else
        consecutive_crashes=$((consecutive_crashes + 1))
        echo "==> Cycle $i crashed with exit code $exit_code at $(date) (no auto-exit marker)."
        echo "    Consecutive crashes: $consecutive_crashes / $MAX_CONSECUTIVE_CRASHES"
        if [ $consecutive_crashes -ge $MAX_CONSECUTIVE_CRASHES ]; then
            echo "    Hit $MAX_CONSECUTIVE_CRASHES consecutive non-zero exits — stopping."
            echo "    (This isn't the expected thread-leak AUTO-EXIT pattern; a real"
            echo "     bug is likely.  Investigate the last training log above.)"
            exit $exit_code
        fi
        echo "    Likely a crash before auto-exit could start.  Retrying from latest checkpoint in 60s..."
        sleep 60
    fi
done

echo "==> Reached MAX_RESTARTS ($MAX_RESTARTS cycles).  Training finished or give up."
