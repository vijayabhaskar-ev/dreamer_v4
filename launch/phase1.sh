#!/bin/bash
# Corrected Phase 1 run: the effective command, rebuilt from the arguments the run recorded
# (where a flag was given twice, the later value won). The run also passed --dynamics-loss-norm joint; in this
# code one joint normalizer is the only mode, so that flag no longer exists. The run's --wandb-name is left out;
# it named the cloud vendor.
python -m dynamics.train_dynamics \
  --tokenizer-ckpt checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt \
  --dataset offline \
  --dataset-path ball_in_cup_catch_aligned.npz \
  --task ball_in_cup_catch \
  --action-dim 2 \
  --epochs 600 \
  --steps-per-epoch 500 \
  --batch-size 48 \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --min-lr 1e-6 \
  --weight-decay 0.01 \
  --weight-decay-heavy 0.1 \
  --grad-clip 5.0 \
  --curriculum-warmup-steps 2000 \
  --curriculum-ramp-steps 4000 \
  --K-max 64 \
  --K-inference 4 \
  --seq-len-short 8 \
  --seq-len-long 32 \
  --context-length 16 \
  --long-batch-ratio 0.15 \
  --embed-dim 512 \
  --depth 12 \
  --num-heads 8 \
  --num-kv-heads 2 \
  --num-register-tokens 4 \
  --temporal-interval 4 \
  --amp \
  --num-workers 4 \
  --val-steps-per-epoch 5 \
  --device cuda \
  --checkpoint-dir checkpoints-dynamics-phase1-aligned-joint/dynamics \
  --save-final checkpoints-dynamics-phase1-aligned-joint/dynamics/final.pt \
  --checkpoint-interval 10 \
  --wandb-project dreamer-v4-dynamics
