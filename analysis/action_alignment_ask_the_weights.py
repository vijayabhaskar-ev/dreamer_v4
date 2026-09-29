"""Ask the TRAINED WEIGHTS which action timing they were trained on.

Independent of the repo's history: even an uncommitted fix on a rented machine would show up here,
because a network predicts best under the input timing it saw during training.

For a window of frames j = 0..T-1 taken from the raw npz at row `start`, the model's slot j holds
frame j and predicts frame j. We vary only WHICH STORED ROW lands in slot j's action token:

    k = -2  two rows back      slot j <- stored[start + j - 2]
    k = -1  previous row       slot j <- stored[start + j - 1]   <- what the repo's loader + padding does
    k =  0  same row           slot j <- stored[start + j]       <- physically correct (the move that produced frame j)
    k = +1  next row           slot j <- stored[start + j + 1]

Same frames, same noise, same tau/d for every k (paired).

How to read it. The test is decisive in ONE direction: a model trained with the physically correct timing (k = 0) gets both
its training timing and the most causal information at k = 0, so its minimum cannot sit at k = -1. A minimum at k = -1
therefore proves the weights were trained one step stale. (A minimum at k = 0 would be ambiguous, because k = 0 also hands the
model strictly more causal information.)

The policy-head profile is NOT a discriminator on its own: the head's prediction is pulled toward the action visible in its own
slot token (persistence), so the lowest NLL sits on the visible row whatever the training target was. It is printed only to show
that the mass sits on already-executed moves (m = -1, 0) rather than on the move to make (m = +1).

Run:  WANDB_MODE=disabled python analysis/action_alignment_ask_the_weights.py     (needs the local checkpoints + the 5.9 GB npz)
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
from dynamics.flow_matching import add_noise, sample_tau_and_d
from dynamics.trainer import DynamicsTrainer, DynamicsTrainingConfig
from dynamics.evaluate_dynamics import load_tokenizer_config_from_ckpt, load_dynamics_config_from_ckpt, resolve_device
from dynamics.evaluate_agent import setup as phase2_setup
from tokenizer.dataset import OfflineDataset

TOK = "checkpoints/checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt"
NPZ = "ball_in_cup_catch.npz"
T, B, NB = 8, 32, 40
KS = [-2, -1, 0, 1]
NAMES = {-2: "two rows back", -1: "previous row  (repo loader+pad)", 0: "same row      (physically correct)", 1: "next row"}


def load_phase1(ckpt, device):
    tcfg = load_tokenizer_config_from_ckpt(TOK, device); dcfg = load_dynamics_config_from_ckpt(ckpt, device)
    tr = DynamicsTrainer(dynamics_cfg=dcfg, tokenizer_cfg=tcfg,
                         training_cfg=DynamicsTrainingConfig(epochs=1, batch_size=B, amp=False, device=str(device),
                                                             log_model_stats=False, log_memory=False,
                                                             log_interval=10_000, log_model_stats_interval=10_000),
                         tokenizer_ckpt=TOK)
    tr.load_checkpoint(ckpt, strict=True); tr.model.to(device).eval(); tr.tokenizer.eval()
    return tr, dcfg


def load_phase2(ckpt, device):
    opts = SimpleNamespace(device=str(device), tokenizer_ckpt=TOK, dynamics_ckpt=ckpt, num_tasks=1,
                           batch_size=B, amp=False, seq_len=T)
    tr, dcfg, *_ = phase2_setup(opts)
    return tr, dcfg


def windows(ds, rng):
    """Sample B windows from the split's episodes; return frames + the raw stored rows around them."""
    eps = ds.allowed_idx[rng.integers(0, len(ds.allowed_idx), B)]
    starts = rng.integers(3, ds.episode_len - T - 3, B)
    fr = np.stack([ds.frames[e, s:s + T] for e, s in zip(eps, starts)])                     # (B,T,H,W,3)
    fr = torch.from_numpy(fr).permute(0, 1, 4, 2, 3).float() / 255.0
    stored = np.stack([ds.actions[e, s - 3:s + T + 3] for e, s in zip(eps, starts)])        # (B,T+6,A); local row r <-> stored[s-3+r]
    return fr, torch.from_numpy(stored.copy()).float()


def feed(stored, k):
    """(B,T-1,A) tensor whose entry j-1 goes into slot j:  stored[start + j + k], j = 1..T-1."""
    return stored[:, 3 + 1 + k: 3 + T + k]


def report(title, mse):  # mse: dict k -> (N,) per-window errors
    ref = mse[-1]
    print(f"\n  {title}")
    for k in KS + ["shuf"]:
        v = mse[k]; dlt = v - ref; n = len(v)
        se = dlt.std(ddof=1) / np.sqrt(n) if k != -1 else 0.0
        tag = NAMES.get(k, "shuffled across batch (no information)")
        extra = "" if k == -1 else f"   vs previous row: {100 * dlt.mean() / ref.mean():+6.2f}%  (t={dlt.mean() / se:+.1f})"
        print(f"    k={str(k):>4}  {tag:<40} error={v.mean():.5f}{extra}")
    best = min(KS, key=lambda kk: mse[kk].mean())
    print(f"    --> lowest error at k={best:+d}: {NAMES[best].strip()}")


def world_model_test(tr, dcfg, ds, device, use_agent=False):
    rng = np.random.default_rng(0); torch.manual_seed(0)
    r1 = {k: [] for k in KS + ["shuf"]}; r2 = {k: [] for k in KS + ["shuf"]}
    with torch.no_grad():
        for _ in range(NB):
            fr, stored = windows(ds, rng); fr, stored = fr.to(device), stored.to(device)
            z = tr.model.encode_frames(fr)
            # R1: noise exactly as evaluate_dynamics / training samples it
            tau1, d1 = sample_tau_and_d(B, T, K_max=dcfg.K_max, device=device); tau1[:, 0] = 1.0 - dcfg.tau_ctx
            zn1, _ = add_noise(z, tau1)
            # R2: next-frame prediction: context nearly clean, LAST frame pure noise
            tau2 = torch.full((B, T), 1.0 - dcfg.tau_ctx, device=device); d2 = torch.full((B, T), 1.0 / dcfg.K_max, device=device)
            tau2[:, -1] = 0.0; d2[:, -1] = 0.25
            zn2, _ = add_noise(z, tau2)
            perm = torch.randperm(B, device=device)
            for k in KS + ["shuf"]:
                a = feed(stored, -1)[perm] if k == "shuf" else feed(stored, k)
                zh1 = tr.model(zn1, a, tau1, d1, use_agent_tokens=use_agent).z_hat
                zh2 = tr.model(zn2, a, tau2, d2, use_agent_tokens=use_agent).z_hat
                r1[k].append(((zh1 - z) ** 2)[:, 1:].mean(dim=(1, 2, 3)).cpu().numpy())
                r2[k].append(((zh2 - z) ** 2)[:, -1].mean(dim=(1, 2)).cpu().numpy())
    r1 = {k: np.concatenate(v) for k, v in r1.items()}; r2 = {k: np.concatenate(v) for k, v in r2.items()}
    report(f"WORLD MODEL, training-style noise, latent error over frames 1..{T-1}   (n={len(r1[-1])} windows)", r1)
    report(f"WORLD MODEL, next-frame prediction: last frame from pure noise      (n={len(r2[-1])} windows)", r2)


def policy_test(tr, dcfg, ds, device):
    """Which stored row does the cloned policy at frame j actually predict?
       m = 0: the move that PRODUCED frame j (already executed).   m = +1: the move the expert makes AT frame j (the correct BC target)."""
    rng = np.random.default_rng(1); torch.manual_seed(1)
    MS = [-1, 0, 1, 2]; nll = {m: [] for m in MS}
    with torch.no_grad():
        for _ in range(NB):
            fr, stored = windows(ds, rng); fr, stored = fr.to(device), stored.to(device)
            z = tr.model.encode_frames(fr)
            tau = torch.full((B, T), 1.0 - dcfg.tau_ctx, device=device); d = torch.full((B, T), 1.0 / dcfg.K_max, device=device)
            zn, _ = add_noise(z, tau)
            h = tr.model(zn, feed(stored, -1), tau, d, use_agent_tokens=True).agent_out      # fed exactly as the repo feeds it
            for m in MS:
                tgt = stored[:, 3 + 1 + m: 3 + T - 1 + m]                                    # stored[start + j + m], j = 1..T-2
                nll[m].append((-tr.policy_head.log_prob(h[:, 1:T - 1], tgt, mtp_offset=0)).mean(dim=1).cpu().numpy())
    nll = {m: np.concatenate(v) for m, v in nll.items()}
    lab = {-1: "row before the frame's own row", 0: "the move that PRODUCED frame j  (already executed)",
           1: "the move made AT frame j       (correct BC target)", 2: "one move later"}
    print(f"\n  POLICY HEAD, negative log-likelihood of stored[start+j+m] at frame j   (n={len(nll[0])} windows, lower = what it predicts)")
    for m in MS:
        print(f"    m={m:+d}  {lab[m]:<52} NLL={nll[m].mean():.3f}")
    print("    (lowest NLL sits on the row visible in the slot's own action token = persistence; see module docstring)")


if __name__ == "__main__":
    device = resolve_device("cuda")
    ds = OfflineDataset(NPZ, seq_len=T, batch_size=B, steps_per_epoch=1, split="val")       # held-out episodes, same split rule as training
    print(f"held-out episodes: {len(ds.allowed_idx)}   window T={T}   {NB} batches x {B}")

    print("\n================ PHASE 1 world model (released, epoch 320) ================")
    tr, dcfg = load_phase1("checkpoints/checkpoints-dynamics-iter46-phase1-bf16-optim/dynamics/dynamics_epoch_320.pt", device)
    world_model_test(tr, dcfg, ds, device)
    del tr; torch.cuda.empty_cache()

    for seed in (11, 13):
        print(f"\n================ PHASE 2 agent, finetune seed {seed} (paper checkpoint) ================")
        tr, dcfg = load_phase2(f"checkpoints/checkpoints-phase2-cat-seed{seed}/final.pt", device)
        world_model_test(tr, dcfg, ds, device, use_agent=True)
        policy_test(tr, dcfg, ds, device)
        del tr; torch.cuda.empty_cache()
