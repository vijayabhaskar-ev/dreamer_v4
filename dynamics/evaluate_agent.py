"""Phase-2 Agent Evaluation for Dreamer V4.

Evaluates the agent-finetuning heads (reward, continue/termination, policy/BC)
that read from the dynamics model's agent-token output `h_t`. This is the
agent-side counterpart to `evaluate_dynamics.py` (which evaluates the world
model only).

The agent heads are trained with multi-token prediction (MTP, paper §3.3,
length L=8): from the agent state at frame t they predict targets at offsets
0..L. At inference/imagination the policy acts with offset 0, but the other
offsets are an auxiliary training signal — so this script measures BOTH the
deployment metric (offset 0) AND how prediction quality degrades with offset.

Sections
--------
1. Policy / behavior cloning : action MSE, NLL, calibration, action histograms.
2. Reward model             : MSE, Pearson/Spearman, calibration, event AUC/AP,
                              per-true-value breakdown.
3. Continue / termination   : BCE, Brier, AUC, AP, confusion, reliability.
4. MTP-offset degradation   : every metric above as a curve over offset 0..L.
5. Noise robustness         : head metrics as a function of signal level tau.
6. Action-conditioning sanity: shuffle actions, measure prediction sensitivity.
7. Behavioral rollout + video: policy-driven vs teacher-forced rollout, decoded
                              to a side-by-side GIF, with latent-divergence and
                              predicted reward/continue trajectories.

All scalar metrics are evaluated at the *inference operating point* (near-clean
latents, signal level `1 - tau_ctx`), because that is the regime the heads see
during imagination. Section 5 sweeps tau to disentangle "noisy latent" from
"bad head".

The script has no sklearn/scipy dependency — ROC-AUC (Mann-Whitney U), average
precision, and Spearman correlation are implemented in-file.
"""

from __future__ import annotations

import _env_setup  # noqa: F401  (side-effect import: must precede torch)

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import wandb

from .flow_matching import add_noise
from .trainer import (
    DynamicsTrainer,
    DynamicsTrainingConfig,
    build_mtp_targets,
    build_mtp_action_targets,
)
# Reuse the dynamics-eval helpers verbatim so the two evals stay in lockstep.
from .evaluate_dynamics import (
    safe_torch_load,
    resolve_device,
    load_tokenizer_config_from_ckpt,
    load_dynamics_config_from_ckpt,
    decode_latents_to_frames,
    save_video_gif,
    save_line_plot,
    write_csv,
)
# IMPORTANT: the imagination denoise_one_frame returns (z, h); the one in
# evaluate_dynamics returns only z. We need the agent state h here.
from imagination.rollout import denoise_one_frame as denoise_one_frame_agent
from tokenizer.config import TokenizerConfig
from tokenizer.dataset import DatasetFactory
from heads import CategoricalPolicyHead


# Plot colors (R, G, B)
C_POLICY = (33, 102, 172)
C_TEACHER = (178, 24, 43)
C_GT = (27, 120, 55)
C_A = (33, 102, 172)
C_B = (178, 24, 43)


def _safe_line_plot(path, title, x_values, series, y_label) -> None:
    """NaN-robust wrapper over evaluate_dynamics.save_line_plot.

    Metric curves legitimately contain NaN (e.g. Pearson on constant
    predictions, AUC with no positive samples). The base plotter does
    int(NaN) and crashes, so sanitize non-finite values to 0.0 for the
    *plot only* (CSV/JSON keep the true NaN). Skip if there is nothing
    finite to draw.
    """
    clean_series = []
    any_finite = False
    for name, vals, color in series:
        cv = [v if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0 for v in vals]
        any_finite = any_finite or any(
            isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
        clean_series.append((name, cv, color))
    if not x_values or not any_finite:
        return
    save_line_plot(path, title, x_values, clean_series, y_label)


def _decode_chunked(tokenizer, z_seq, chunk: int = 8) -> torch.Tensor:
    """Decode (1, L, S_z, D) latents in <=`chunk`-frame blocks at the tokenizer's
    trained length, so RoPE positions stay in-distribution.

    The iter46 tokenizer uses length-dependent absolute RoPE positions and was
    trained only at seq_len=`chunk` (8). Decoding the full L>chunk sequence at
    once degrades frames past `chunk` and smears future->past (TEST A/C/G). We
    instead decode in fixed `chunk`-size windows — end-aligning the final
    partial window and keeping only its new tail — so every decode runs at the
    trained length. Returns (L, C, H, W).
    """
    L = z_seq.shape[1]
    chunk = max(1, min(chunk, L))
    out = [None] * L
    start = 0
    while start < L:
        end = min(start + chunk, L)
        if end - start < chunk:                # final partial window: end-align a full chunk
            ws = L - chunk
            dec = decode_latents_to_frames(tokenizer, z_seq[:, ws:L])[0]   # (chunk,C,H,W)
            for i in range(start, L):
                out[i] = dec[i - ws]
            break
        dec = decode_latents_to_frames(tokenizer, z_seq[:, start:end])[0]  # (chunk,C,H,W)
        for i in range(start, end):
            out[i] = dec[i - start]
        start = end
    return torch.stack(out, dim=0)             # (L,C,H,W)


def _encode_chunked(model, frames, chunk: int = 8) -> torch.Tensor:
    """Encode (B,T,3,H,W) frames into clean latents using <=`chunk`-frame windows.

    The tokenizer's RoPE positions are length-dependent and it was trained only
    at seq_len=`chunk` (8): encoding T>chunk frames at once shifts the latents
    (TEST D, ~18%) and degrades frames past `chunk` (TEST A). The dynamics model
    length-generalizes (trained at 8 and 32) and consumes these latents, so we
    keep the *tokenizer* in-distribution by encoding in fixed `chunk` windows
    (end-aligned final window). Returns (B,T,S_z,D)."""
    T = frames.shape[1]
    chunk = max(1, min(chunk, T))
    outs = [None] * T
    s = 0
    while s < T:
        e = min(s + chunk, T)
        if e - s < chunk:                      # end-align final window
            ws = T - chunk
            z = model.encode_frames(frames[:, ws:T])      # (B,chunk,S_z,D)
            for i in range(s, T):
                outs[i] = z[:, i - ws]
            break
        z = model.encode_frames(frames[:, s:e])           # (B,chunk,S_z,D)
        for i in range(s, e):
            outs[i] = z[:, i - s]
        s = e
    return torch.stack(outs, dim=1)            # (B,T,S_z,D)


def _decode_anchored(tokenizer, z_ctx, z_gen, chunk: int = 8) -> torch.Tensor:
    """Decode a [context, generated] rollout into frames (C+H, Cc, Hh, W).

    A *rollout* visualization can't just chunk the generated latents in
    isolation — pure-generated windows render near-blank because the tokenizer
    decoder expects latents that follow real context. So we keep the real
    `z_ctx` as an in-distribution anchor in every window and decode
    `chunk - C` generated frames per window (<= `chunk` total, always anchored).
    This shows the true generated-frame quality instead of an artifact."""
    C = z_ctx.shape[1]
    H = z_gen.shape[1]
    g = max(1, chunk - C)                       # generated frames per window
    out = []
    first = decode_latents_to_frames(tokenizer, torch.cat([z_ctx, z_gen[:, :min(g, H)]], 1))[0]
    out.append(first[:C])                       # (C,...) real context, decoded with anchor
    t = 0
    while t < H:
        win = torch.cat([z_ctx, z_gen[:, t:t + g]], 1)     # <= chunk frames, context-anchored
        dec = decode_latents_to_frames(tokenizer, win)[0]
        out.append(dec[C:C + min(g, H - t)])    # generated tail
        t += g
    return torch.cat(out, 0)                    # (C+H,...)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dreamer V4 Phase-2 agent evaluation")
    p.add_argument("--dynamics-ckpt", required=True, help="Phase-2 (agent) checkpoint")
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--dataset", default="offline")
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--task", default="ball_in_cup_catch")
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--num-tasks", type=int, default=1,
                   help="Number of agent tasks; auto-inferred from ckpt, warns on mismatch.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=20, help="Val batches for the scalar metrics.")
    p.add_argument("--seq-len", type=int, default=16,
                   help="Eval sequence length (>= mtp_length+2). 0 = checkpoint seq_len.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    # readout / noise
    p.add_argument("--readout-tau", type=float, default=-1.0,
                   help="Signal level for head readout. <0 => 1 - tau_ctx (deployment point).")
    p.add_argument("--tau-sweep", type=str, default="0.1,0.3,0.5,0.7,0.9,1.0")
    p.add_argument("--mtp-max-offset", type=int, default=0, help="0 = checkpoint mtp_length.")
    # held-out split
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    # rollout
    p.add_argument("--num-context-frames", type=int, default=4)
    p.add_argument("--rollout-horizon", type=int, default=0, help="0 = auto (cap 15).")
    p.add_argument("--rollout-batches", type=int, default=8)
    p.add_argument("--rollout-K", type=int, default=0, help="0 = dynamics K_inference.")
    # output
    p.add_argument("--output-dir", default="evaluation/agent")
    p.add_argument("--max-gifs", type=int, default=4)
    p.add_argument("--gif-fps", type=int, default=8)
    # wandb
    p.add_argument("--wandb-project", default="dreamer-v4-agent-eval")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--wandb-disabled", action="store_true")
    return p


# ─────────────────────────────────────────────────────────────────────────
# Dependency-free metric primitives (operate on flat CPU float tensors)
# ─────────────────────────────────────────────────────────────────────────
def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.double()
    y = y.double()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom.item() < 1e-12:
        return float("nan")
    return (xc.dot(yc) / denom).item()


def _rankdata(v: torch.Tensor) -> torch.Tensor:
    """Average ranks (1-based), ties averaged — like scipy.stats.rankdata."""
    n = v.numel()
    order = torch.argsort(v)
    sorted_v = v[order]
    ranks = torch.empty(n, dtype=torch.double)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based ranks i+1..j+1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    return _pearson(_rankdata(x.double()), _rankdata(y.double()))


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney U ROC-AUC. labels in {0,1}."""
    labels = labels.double()
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    ranks = _rankdata(scores.double())
    sum_ranks_pos = ranks[labels > 0.5].sum().item()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under precision-recall curve (AP), via the step-wise sum."""
    labels = labels.double()
    n_pos = labels.sum().item()
    if n_pos < 1:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / (tp + fp).clamp(min=1e-12)
    recall = tp / n_pos
    # AP = sum over thresholds of (recall_k - recall_{k-1}) * precision_k
    recall_prev = torch.cat([torch.zeros(1, dtype=torch.double), recall[:-1]])
    ap = ((recall - recall_prev) * precision).sum().item()
    return ap


def _binary_at_threshold(scores: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    pred = (scores > thr).double()
    lab = labels.double()
    tp = (pred * lab).sum().item()
    fp = (pred * (1 - lab)).sum().item()
    fn = ((1 - pred) * lab).sum().item()
    tn = ((1 - pred) * (1 - lab)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 and not math.isnan(prec) and not math.isnan(rec) else float("nan")
    acc = (tp + tn) / max(1.0, (tp + fp + fn + tn))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


# ─────────────────────────────────────────────────────────────────────────
# Buffered collectors (eval sets are small; buffering keeps metric math exact)
# ─────────────────────────────────────────────────────────────────────────
_BUF_CAP = 500_000  # per-tensor row cap; logged in summary if hit


class Collector:
    """Accumulates valid (pred, target) rows on CPU for one MTP offset."""

    def __init__(self):
        self.r_pred: List[torch.Tensor] = []
        self.r_true: List[torch.Tensor] = []
        self.c_pred: List[torch.Tensor] = []     # continue prob
        self.done_true: List[torch.Tensor] = []  # done in {0,1}
        self.a_mu: List[torch.Tensor] = []        # gaussian path
        self.a_logstd: List[torch.Tensor] = []    # gaussian path
        self.a_logits: List[torch.Tensor] = []    # categorical path: (N, A, K) per-dim logits
        self.a_true: List[torch.Tensor] = []      # ground-truth actions (head-agnostic)
        self.truncated = False

    def _append(self, lst: List[torch.Tensor], t: torch.Tensor):
        cur = sum(x.shape[0] for x in lst)
        if cur >= _BUF_CAP:
            self.truncated = True
            return
        room = _BUF_CAP - cur
        lst.append(t[:room].detach().cpu())

    def add_reward(self, pred: torch.Tensor, true: torch.Tensor):
        self._append(self.r_pred, pred.reshape(-1))
        self._append(self.r_true, true.reshape(-1))

    def add_continue(self, cont_prob: torch.Tensor, done: torch.Tensor):
        self._append(self.c_pred, cont_prob.reshape(-1))
        self._append(self.done_true, done.reshape(-1))

    def add_policy(self, mu: torch.Tensor, log_std: torch.Tensor, true: torch.Tensor):
        A = mu.shape[-1]
        self._append(self.a_mu, mu.reshape(-1, A))
        self._append(self.a_logstd, log_std.reshape(-1, A))
        self._append(self.a_true, true.reshape(-1, A))

    def add_policy_categorical(self, logits: torch.Tensor, true: torch.Tensor):
        # logits: (*b, A, K)  true: (*b, A) — buffer raw logits so policy_metrics can
        # compute NLL / per-dim std / entropy / mass-histogram without a second forward.
        # NOTE: K× heavier than the Gaussian (N,A) buffers — at the _BUF_CAP row cap this is
        # ~_BUF_CAP·A·K·4 bytes/offset × (max_offset+1) collectors. Fine for low-DoF eval
        # (cup-catch A=2 ≈ 164 MB/offset); lower _BUF_CAP for high-DoF action spaces.
        A, K = logits.shape[-2], logits.shape[-1]
        self._append(self.a_logits, logits.reshape(-1, A, K))
        self._append(self.a_true, true.reshape(-1, A))

    # -- finalizers --
    def reward_metrics(self) -> Dict[str, object]:
        if not self.r_pred:
            return {}
        p = torch.cat(self.r_pred)
        t = torch.cat(self.r_true)
        mse = (p - t).pow(2).mean().item()
        out = {
            "n": int(p.numel()),
            "mse": mse,
            "rmse": math.sqrt(max(mse, 0.0)),
            "mae": (p - t).abs().mean().item(),
            "pearson": _pearson(p, t),
            "spearman": _spearman(p, t),
        }
        # event detection: reward > 0
        labels = (t > 0).double()
        out["event_auc"] = _roc_auc(p, labels)
        out["event_ap"] = _average_precision(p, labels)
        # per-true-value breakdown
        by_val = []
        for v in sorted(set(t.tolist())):
            m = (t == v)
            if m.sum() == 0:
                continue
            by_val.append({
                "true_value": float(v),
                "count": int(m.sum().item()),
                "pred_mean": p[m].mean().item(),
                "pred_std": p[m].std(unbiased=False).item() if m.sum() > 1 else 0.0,
                "mse": (p[m] - t[m]).pow(2).mean().item(),
            })
        out["by_true_value"] = by_val
        return out

    def continue_metrics(self) -> Dict[str, object]:
        if not self.c_pred:
            return {}
        cont = torch.cat(self.c_pred).clamp(1e-6, 1 - 1e-6)
        done = torch.cat(self.done_true)
        done_prob = 1.0 - cont  # probability of the (rare) done event
        cont_true = 1.0 - done  # continue target
        bce = -(cont_true * cont.log() + (1 - cont_true) * (1 - cont).log()).mean().item()
        brier = (done_prob - done).pow(2).mean().item()
        n_pos = int(done.sum().item())
        out = {
            "n": int(done.numel()),
            "done_positive_count": n_pos,
            "done_rate": done.mean().item(),
            "bce": bce,
            "brier": brier,
            "auc": _roc_auc(done_prob, done),
            "ap": _average_precision(done_prob, done),
        }
        out.update({f"cm_{k}": v for k, v in
                    _binary_at_threshold(done_prob, done, 0.5).items()})
        # reliability diagram (10 bins on continue prob vs empirical continue freq)
        bins = []
        for b in range(10):
            lo, hi = b / 10.0, (b + 1) / 10.0
            m = (cont >= lo) & (cont < hi if b < 9 else cont <= hi)
            if m.sum() == 0:
                continue
            bins.append({"bin_lo": lo, "bin_hi": hi, "count": int(m.sum().item()),
                         "pred_continue": cont[m].mean().item(),
                         "emp_continue": cont_true[m].mean().item()})
        out["reliability"] = bins
        return out

    def policy_metrics(self, policy_head) -> Dict[str, object]:
        if isinstance(policy_head, CategoricalPolicyHead):
            return self._policy_metrics_categorical(policy_head)
        # ── Gaussian path (unchanged) ──
        if not self.a_mu:
            return {}
        mu = torch.cat(self.a_mu)
        log_std = torch.cat(self.a_logstd)
        a = torch.cat(self.a_true)
        A = mu.shape[-1]
        sq = (mu - a).pow(2)
        nll = (-policy_head._log_prob(mu, log_std, a)).mean().item()
        sigma = log_std.exp()
        resid = a - mu
        per_dim = []
        for k in range(A):
            sp = sigma[:, k].mean().item()
            sr = resid[:, k].std(unbiased=False).item()
            per_dim.append({
                "dim": k,
                "mse": sq[:, k].mean().item(),
                "sigma_pred": sp,
                "resid_std": sr,
                "calib_ratio": (sp / sr) if sr > 1e-9 else float("nan"),
            })
        return {
            "n": int(mu.shape[0]),
            "action_mse": sq.mean().item(),
            "action_rmse": math.sqrt(max(sq.mean().item(), 0.0)),
            "nll": nll,
            "per_dim": per_dim,
            # store small samples for histograms
            "_mu_sample": mu[:5000],
            "_a_sample": a[:5000],
        }

    def _policy_metrics_categorical(self, policy_head) -> Dict[str, object]:
        """Categorical analogue of policy_metrics — same key set + additive keys.

        All quantities are computed from the buffered per-dim logits (no model
        forward). The point action is predict()=argmax bin (the DEPLOYED action),
        so action_mse rewards picking the right mode instead of mode-averaging;
        'sigma_pred' carries the per-dim categorical std (the honest spread
        analogue) so the policy_per_dim.csv schema is unchanged.
        """
        if not self.a_logits:
            return {}
        logits = torch.cat(self.a_logits)                          # (N, A, K)
        a = torch.cat(self.a_true)                                 # (N, A)
        N, A, K = logits.shape
        c = policy_head.bin_centers.to(logits.device, logits.dtype)  # (K,) match buffered (cpu) tensors
        bound = float(policy_head.action_bound)
        logK = math.log(K)

        logp = logits.log_softmax(dim=-1)                          # (N, A, K)
        p = logp.exp()
        m = (p * c).sum(-1)                                        # (N, A) distribution mean
        s = (p * (c - m.unsqueeze(-1)).pow(2)).sum(-1).clamp_min(0).sqrt()  # (N, A) per-dim std
        a_hat = c[logits.argmax(-1)]                               # (N, A) predict()=deployed argmax

        sq = (a_hat - a).pow(2)                                    # (N, A) error about deployed action
        bins = policy_head._nearest_bin(a)                        # (N, A) — head's own binning
        lp_dim = logp.gather(-1, bins.unsqueeze(-1)).squeeze(-1)   # (N, A)
        nll = (-lp_dim.sum(-1)).mean().item()                     # Σ over dims (Gaussian summed convention)
        ent = -(p * logp).sum(-1).mean(0)                         # (A,) nats

        # Calibration residual is about the distribution MEAN m — the SAME center as
        # sigma_pred (=s) — so calib_ratio is a coherent spread/error ratio and matches the
        # Gaussian definition exactly (there mean == the deployed action, so its resid is
        # about mu too). The previous mixed-center form (s about m, resid about the argmax
        # a_hat) is not a well-defined ratio for per-position-varying multimodal policies.
        # Deployment accuracy is captured separately by 'mse' (argmax); a genuinely
        # over-dispersed multimodal policy can still (correctly) show calib_ratio >> 1.
        resid = a - m                                             # residual about the distribution mean
        per_dim = []
        for k in range(A):
            sp = s[:, k].mean().item()                            # per-dim categorical std -> 'sigma_pred'
            sr = resid[:, k].std(unbiased=False).item()
            per_dim.append({
                "dim": k, "mse": sq[:, k].mean().item(),
                "sigma_pred": sp, "resid_std": sr,
                "calib_ratio": (sp / sr) if sr > 1e-9 else float("nan"),
                "entropy": ent[k].item(),
                "norm_entropy": ent[k].item() / logK,
                "perplexity": math.exp(ent[k].item()),
            })

        emp_hist = torch.zeros(A, K)                              # empirical action histogram (true bins)
        for k in range(A):
            emp_hist[k].scatter_add_(0, bins[:, k], torch.ones(N))
        emp_hist /= max(1, N)

        return {
            "n": int(N),
            "action_mse": sq.mean().item(),                       # headline: deployed (argmax) MSE
            "action_rmse": math.sqrt(max(sq.mean().item(), 0.0)),
            "nll": nll,                                           # discrete bin NLL — NOT cross-head comparable
            "per_dim": per_dim,
            # additive context keys (downstream .get()/column-lists ignore unknown keys)
            "action_mse_mean": (m - a).pow(2).mean().item(),      # secondary: dist-mean MSE (mode-averaged)
            "quantization_floor_mse": (2 * bound / (K - 1)) ** 2 / 12,  # mean achievable MSE (uniform-within-bin)
            "nll_uniform_ref": A * logK,                          # NLL of a uniform categorical
            "entropy_mean": ent.mean().item(),
            "max_entropy": logK,
            # histogram material (_-prefixed -> stripped from JSON by _clean)
            "_pred_mass": p.mean(0),                              # (A, K) predicted marginal mass
            "_emp_hist": emp_hist,                                # (A, K) empirical histogram
            "_bin_centers": c,                                    # (K,)
        }


# ─────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────
def peek_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[Optional[int], bool]:
    """Peek a checkpoint: return (num_tasks_inferred, has_heads).

    num_tasks is read from the agent_embedding parameter shape; has_heads is
    True iff the three Phase-2 head state_dicts were saved (i.e. this is a
    Phase-2 / agent checkpoint, not a Phase-1 world-model checkpoint).
    """
    state = safe_torch_load(ckpt_path, device)
    has_heads = all(k in state for k in ("reward_head", "continue_head", "policy_head"))
    num_tasks: Optional[int] = None
    model_state = state.get("model", {})
    for k, v in model_state.items():
        kk = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        if kk == "agent_embedding.embedding" and v.dim() == 3:
            num_tasks = 1            # num_tasks==1 path: nn.Parameter (1,1,D)
        elif kk == "agent_embedding.embedding.weight":
            num_tasks = int(v.shape[0])  # num_tasks>1 path: nn.Embedding (N,D)
    return num_tasks, has_heads


def setup(opts) -> Tuple[DynamicsTrainer, object, object, torch.device, int, int]:
    device = resolve_device(opts.device)
    tokenizer_cfg = load_tokenizer_config_from_ckpt(opts.tokenizer_ckpt, device)
    dynamics_cfg = load_dynamics_config_from_ckpt(opts.dynamics_ckpt, device)
    dynamics_cfg.validate_against_tokenizer(tokenizer_cfg)

    # Peek the checkpoint: infer num_tasks + verify it is a Phase-2 (agent)
    # checkpoint. Loading a Phase-1 world-model checkpoint here would silently
    # leave all three heads randomly initialized (load_checkpoint only restores
    # heads that are present in the state dict), producing meaningless metrics.
    inferred, has_heads = peek_checkpoint(opts.dynamics_ckpt, device)
    if not has_heads:
        raise RuntimeError(
            f"{opts.dynamics_ckpt} contains no reward/continue/policy head "
            "state_dicts — it is not a Phase-2 (agent) checkpoint. Run "
            "dynamics.train_agent first, or point --dynamics-ckpt at a Phase-2 "
            "checkpoint. (Evaluating with fresh random heads would be meaningless.)")
    num_tasks = opts.num_tasks
    if inferred is not None and inferred != opts.num_tasks:
        print(f"[WARN] --num-tasks={opts.num_tasks} but checkpoint has "
              f"num_tasks={inferred}. Using checkpoint value (a mismatch would "
              f"load a fresh random task embedding under strict=False).")
        num_tasks = inferred
    elif inferred is None:
        print("[WARN] Could not infer num_tasks from checkpoint (no agent_embedding "
              f"key). Using --num-tasks={opts.num_tasks}. If agent_out is None below, "
              "this checkpoint is likely Phase-1 (no agent tokens).")

    eval_train_cfg = DynamicsTrainingConfig(
        epochs=1, batch_size=opts.batch_size, amp=opts.amp, device=str(device),
        train_heads=True, log_model_stats=False, log_memory=False,
        log_interval=10_000, log_model_stats_interval=10_000,
    )
    trainer = DynamicsTrainer(
        dynamics_cfg=dynamics_cfg, tokenizer_cfg=tokenizer_cfg,
        training_cfg=eval_train_cfg, tokenizer_ckpt=opts.tokenizer_ckpt,
    )
    # CRITICAL ORDERING: enable agent tokens BEFORE load_checkpoint, else the
    # loader strips agent_embedding.* keys (trainer.py:519-521) and agent_out=None.
    trainer.model.enable_agent_tokens(num_tasks=num_tasks)
    trainer.model.agent_embedding.to(device)
    trainer.load_checkpoint(opts.dynamics_ckpt, strict=False)

    trainer.model.to(device)
    trainer.model.eval()
    trainer.tokenizer.eval()
    trainer.reward_head.eval()
    trainer.continue_head.eval()
    trainer.policy_head.eval()

    eval_seq_len = opts.seq_len if opts.seq_len > 0 else dynamics_cfg.seq_len
    eval_seq_len = min(eval_seq_len, dynamics_cfg.seq_len_long)
    mtp_len = dynamics_cfg.mtp_length
    return trainer, dynamics_cfg, tokenizer_cfg, device, eval_seq_len, mtp_len


def make_val_loader(opts, tokenizer_cfg, eval_seq_len, steps, device) -> DataLoader:
    dataset_cfg = replace(tokenizer_cfg, dataset_name=opts.dataset,
                          task_name=opts.task, seq_len=eval_seq_len)
    ds = DatasetFactory.get_dataset(
        dataset_cfg, batch_size=opts.batch_size, steps_per_epoch=steps,
        dataset_path=opts.dataset_path, expected_action_dim=opts.action_dim,
        split="val", val_fraction=opts.val_fraction, split_seed=opts.split_seed,
    )
    return DataLoader(ds, batch_size=None, num_workers=opts.num_workers,
                      pin_memory=(device.type == "cuda"))


def agent_forward(trainer, dynamics_cfg, frames, actions, readout_tau, device):
    """Clean-readout forward: returns agent state h (B,T,D_embed) and z_clean."""
    B, T = frames.shape[0], frames.shape[1]
    # Encode in <=seq_len chunks: the tokenizer is length-limited (trained at 8),
    # so encoding T>8 frames at once degrades the latents the heads read from.
    chunk = int(getattr(trainer.tokenizer_cfg, "seq_len", 8))
    z_clean = _encode_chunked(trainer.model, frames, chunk)    # (B,T,S_z,D_lat)
    tau = torch.full((B, T), float(readout_tau), device=device)
    d = torch.full((B, T), 1.0 / dynamics_cfg.K_max, device=device)
    z_noised, _ = add_noise(z_clean, tau)
    out = trainer.model(z_noised, actions, tau, d, use_agent_tokens=True)
    if out.agent_out is None:
        raise RuntimeError(
            "agent_out is None — agent tokens not active. The checkpoint is "
            "likely Phase-1, or enable_agent_tokens was not called before load.")
    return out.agent_out, z_clean


# ─────────────────────────────────────────────────────────────────────────
# Sections 1-4: scalar head metrics + per-offset degradation (shared loop)
# ─────────────────────────────────────────────────────────────────────────
def _validity_mask(T: int, n: int, device, length: Optional[int] = None) -> torch.Tensor:
    """Boolean (L_pos,) over positions valid at offset n: pos + n < limit."""
    limit = (length if length is not None else T)
    pos = torch.arange(limit, device=device)
    return (pos + n) < T  # for reward/continue T==limit; for policy limit=T-1


def run_core_and_mtp(trainer, dynamics_cfg, opts, device, readout_tau, max_offset):
    loader = make_val_loader(opts, trainer.tokenizer_cfg, _eval_seq_len(opts, dynamics_cfg),
                             opts.steps, device)
    collectors = {n: Collector() for n in range(max_offset + 1)}
    mtp_len = dynamics_cfg.mtp_length
    # Match training precision: the dynamics model was trained with bf16 AMP
    # (not fp16). Default (--amp off) runs fp32, which is exact for eval.
    autocast = torch.amp.autocast(device.type, enabled=opts.amp, dtype=torch.bfloat16)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if processed >= opts.steps:
                break
            frames, actions, rewards, dones = (x.to(device) for x in batch)
            B, T = frames.shape[0], frames.shape[1]
            with autocast:
                h, _ = agent_forward(trainer, dynamics_cfg, frames, actions, readout_tau, device)
            h = h.float()
            rewards_future, dones_future = build_mtp_targets(rewards, dones, mtp_len)   # (B,T,L+1)
            actions_future = build_mtp_action_targets(actions, mtp_len)                 # (B,T-1,L+1,A)
            for n in range(max_offset + 1):
                col = collectors[n]
                # reward / continue read full h (B,T,*)
                r_pred = trainer.reward_head.predict(h, mtp_offset=n)        # (B,T)
                c_pred = trainer.continue_head.predict(h, mtp_offset=n)      # (B,T)
                vmask = _validity_mask(T, n, device, length=T)              # (T,)
                vmask_bt = vmask.unsqueeze(0).expand(B, T)
                r_true = rewards_future[..., n]
                d_true = dones_future[..., n]
                col.add_reward(r_pred[vmask_bt], r_true[vmask_bt])
                col.add_continue(c_pred[vmask_bt], d_true[vmask_bt])
                # policy reads h[:, :-1] (T-1 positions)
                a_true = actions_future[..., n, :]                                  # (B,T-1,A)
                # Action target at policy position t, offset n is dataset action[t+n],
                # valid when t+n <= T-2  <=>  t+n < T-1.
                pmask = (torch.arange(T - 1, device=device) + n) < (T - 1)
                pmask_bt = pmask.unsqueeze(0).expand(B, T - 1)
                if isinstance(trainer.policy_head, CategoricalPolicyHead):
                    logits = trainer.policy_head.forward(h[:, :-1], mtp_offset=n)    # (B,T-1,A,K)
                    col.add_policy_categorical(logits[pmask_bt], a_true[pmask_bt])   # masks leading (B,T-1)
                else:
                    mu, log_std = trainer.policy_head.forward(h[:, :-1], mtp_offset=n)  # (B,T-1,A)
                    col.add_policy(mu[pmask_bt], log_std[pmask_bt], a_true[pmask_bt])
            processed += 1
    # finalize
    per_offset = {}
    for n, col in collectors.items():
        per_offset[n] = {
            "reward": col.reward_metrics(),
            "continue": col.continue_metrics(),
            "policy": col.policy_metrics(trainer.policy_head),
            "truncated": col.truncated,
        }
    return per_offset, processed


def _eval_seq_len(opts, dynamics_cfg) -> int:
    sl = opts.seq_len if opts.seq_len > 0 else dynamics_cfg.seq_len
    return min(sl, dynamics_cfg.seq_len_long)


# ─────────────────────────────────────────────────────────────────────────
# Section 5: noise robustness (tau sweep)
# ─────────────────────────────────────────────────────────────────────────
def run_noise_sweep(trainer, dynamics_cfg, opts, device, taus):
    rows = []
    for tau in taus:
        per_offset, _ = run_core_and_mtp(trainer, dynamics_cfg, opts, device, tau, max_offset=0)
        m0 = per_offset[0]
        rows.append({
            "tau": tau,
            "reward_mse": m0["reward"].get("mse", float("nan")),
            "reward_pearson": m0["reward"].get("pearson", float("nan")),
            "reward_event_auc": m0["reward"].get("event_auc", float("nan")),
            "continue_auc": m0["continue"].get("auc", float("nan")),
            "continue_bce": m0["continue"].get("bce", float("nan")),
            "action_mse": m0["policy"].get("action_mse", float("nan")),
            "action_nll": m0["policy"].get("nll", float("nan")),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Section 6: action-conditioning sanity (shuffle test on the agent path)
# ─────────────────────────────────────────────────────────────────────────
def run_action_conditioning(trainer, dynamics_cfg, opts, device, readout_tau):
    loader = make_val_loader(opts, trainer.tokenizer_cfg, _eval_seq_len(opts, dynamics_cfg),
                             opts.steps, device)
    A = dynamics_cfg.action_dim
    sum_r_delta = 0.0
    sum_mu_delta = torch.zeros(A, dtype=torch.float64)
    n_r = 0
    n_mu = 0
    batches = 0
    # Match training precision: the dynamics model was trained with bf16 AMP
    # (not fp16). Default (--amp off) runs fp32, which is exact for eval.
    autocast = torch.amp.autocast(device.type, enabled=opts.amp, dtype=torch.bfloat16)
    with torch.no_grad():
        for batch in loader:
            if batches >= opts.steps:
                break
            frames, actions, _, _ = (x.to(device) for x in batch)
            B, T = frames.shape[0], frames.shape[1]
            if B < 2:
                continue
            perm = torch.randperm(B, device=device)
            with autocast:
                h, _ = agent_forward(trainer, dynamics_cfg, frames, actions, readout_tau, device)
                h_shuf, _ = agent_forward(trainer, dynamics_cfg, frames, actions[perm], readout_tau, device)
            h, h_shuf = h.float(), h_shuf.float()
            r = trainer.reward_head.predict(h, mtp_offset=0)
            r_s = trainer.reward_head.predict(h_shuf, mtp_offset=0)
            sum_r_delta += (r - r_s).abs().sum().item()
            n_r += r.numel()
            # predict() is on both heads (== mu for Gaussian, argmax-bin for categorical),
            # so this shuffle-sensitivity delta is head-agnostic. NOTE categorical predict()
            # is discrete (argmax), so small belief shifts can read exactly 0 — the metric
            # stays valid but is less sensitive than the Gaussian continuous mean.
            a_det = trainer.policy_head.predict(h[:, :-1], mtp_offset=0)         # (B,T-1,A)
            a_det_s = trainer.policy_head.predict(h_shuf[:, :-1], mtp_offset=0)
            sum_mu_delta += (a_det - a_det_s).abs().sum(dim=(0, 1)).double().cpu()
            n_mu += a_det.shape[0] * a_det.shape[1]
            batches += 1
    reward_delta = sum_r_delta / max(1, n_r)
    mu_delta = (sum_mu_delta / max(1, n_mu)).tolist()
    return {"reward_abs_delta": reward_delta, "policy_mu_abs_delta_per_dim": mu_delta,
            "batches": batches}


# ─────────────────────────────────────────────────────────────────────────
# Section 7: behavioral rollout (policy vs teacher) + decoded video
# ─────────────────────────────────────────────────────────────────────────
def _rollout_one(trainer, dynamics_cfg, z_context, actions_context, future_actions,
                 horizon, K, tau_ctx, K_max, context_window, device, mode):
    """One autoregressive rollout. mode='policy' or 'teacher'.

    Returns (z_clean_gen (B,H,S_z,D), rewards (B,H), continues (B,H)).
    Mirrors imagination.rollout.imagine_rollout buffer/noise handling, but
    keeps the *clean* generated latents for decoding.
    """
    B, C, S_z, D_lat = z_context.shape
    A = dynamics_cfg.action_dim
    tau_ctx_val = 1.0 - tau_ctx
    z_ctx_noised, _ = add_noise(z_context, torch.full((B, C), tau_ctx_val, device=device))

    max_buf = C + horizon
    z_buffer = torch.zeros(B, max_buf, S_z, D_lat, device=device)
    tau_buffer = torch.full((B, max_buf), tau_ctx_val, device=device)
    d_buffer = torch.full((B, max_buf), 1.0 / K_max, device=device)
    actions_buffer = torch.zeros(B, max_buf, A, device=device)
    z_buffer[:, :C] = z_ctx_noised
    # Convention inherited from imagination.rollout.imagine_rollout (Phase-3
    # deployment): only the C-1 *context* actions are seeded; the C-1->C
    # transition action is left zero, and a_new (policy or teacher) starts
    # filling buffer position C. We match this so the eval reflects the actual
    # deployed rollout. (Whether zeroing that first transition is itself ideal
    # is a separate question about imagine_rollout, surfaced to the user.)
    actions_buffer[:, :C - 1] = actions_context
    buf_len = C

    z_gen = torch.zeros(B, horizon, S_z, D_lat, device=device)
    rewards = torch.zeros(B, horizon, device=device)
    continues = torch.zeros(B, horizon, device=device)

    for t in range(horizon):
        ws = max(0, buf_len - context_window)
        z_new, h_new = denoise_one_frame_agent(
            trainer.model, z_buffer[:, ws:buf_len], tau_buffer[:, ws:buf_len],
            d_buffer[:, ws:buf_len], actions_buffer[:, ws:buf_len], K)
        z_gen[:, t] = z_new[:, 0]
        rewards[:, t] = trainer.reward_head.predict(h_new)
        continues[:, t] = trainer.continue_head.predict(h_new)
        if mode == "policy":
            a_new = trainer.policy_head.sample(h_new)
        else:  # teacher-forced
            if t < future_actions.shape[1]:
                a_new = future_actions[:, t]
            else:
                a_new = trainer.policy_head.sample(h_new)
        z_noised, _ = add_noise(z_new.detach(), torch.full((B, 1), tau_ctx_val, device=device))
        z_buffer[:, buf_len:buf_len + 1] = z_noised
        actions_buffer[:, buf_len] = a_new
        buf_len += 1
    return z_gen, rewards, continues


def run_rollout(trainer, dynamics_cfg, opts, device, out_dir, gif_paths):
    C = opts.num_context_frames
    K = opts.rollout_K if opts.rollout_K > 0 else dynamics_cfg.K_inference
    tau_ctx = dynamics_cfg.tau_ctx
    K_max = dynamics_cfg.K_max
    context_window = 16
    seq_len = max(C + (opts.rollout_horizon if opts.rollout_horizon > 0 else 15) + 1,
                  C + 2)
    seq_len = min(seq_len, dynamics_cfg.seq_len_long)
    horizon = opts.rollout_horizon if opts.rollout_horizon > 0 else (seq_len - C)
    horizon = max(1, min(horizon, seq_len - C))

    loader = make_val_loader(opts, trainer.tokenizer_cfg, seq_len, opts.rollout_batches, device)

    div_pt = torch.zeros(horizon, dtype=torch.float64)   # policy vs teacher
    div_tg = torch.zeros(horizon, dtype=torch.float64)   # teacher vs GT
    r_pol = torch.zeros(horizon, dtype=torch.float64)
    r_tch = torch.zeros(horizon, dtype=torch.float64)
    c_pol = torch.zeros(horizon, dtype=torch.float64)
    c_tch = torch.zeros(horizon, dtype=torch.float64)
    n_div = 0
    gif_cpu: List[torch.Tensor] = []
    batches = 0
    # Match training precision: the dynamics model was trained with bf16 AMP
    # (not fp16). Default (--amp off) runs fp32, which is exact for eval.
    autocast = torch.amp.autocast(device.type, enabled=opts.amp, dtype=torch.bfloat16)

    with torch.no_grad():
        for batch in loader:
            if batches >= opts.rollout_batches:
                break
            frames, actions, _, _ = (x.to(device) for x in batch)
            B, T = frames.shape[0], frames.shape[1]
            if T < C + horizon:
                continue
            # Chunked encode: seq_len here (C+horizon, up to ~20) exceeds the
            # tokenizer's trained length, so encoding at once degrades the GT
            # latents (z_gt) and the rollout seed (z_context). Chunking keeps
            # the tokenizer in-distribution; the dynamics length-generalizes.
            tok_chunk = int(getattr(trainer.tokenizer_cfg, "seq_len", 8))
            z_all = _encode_chunked(trainer.model, frames, tok_chunk)    # (B,T,S_z,D)
            z_context = z_all[:, :C]
            actions_context = actions[:, :C - 1]
            # Teacher action at rollout step t is stored at buffer pos C+t (same
            # slot the policy's a_new fills), i.e. dataset action index C+t.
            # The guard in _rollout_one falls back to a policy sample past the end.
            future_actions = actions[:, C:]                              # actions driving steps 0..
            with autocast:
                z_pol, rp, cp = _rollout_one(
                    trainer, dynamics_cfg, z_context, actions_context, future_actions,
                    horizon, K, tau_ctx, K_max, context_window, device, "policy")
                z_tch, rt, ct = _rollout_one(
                    trainer, dynamics_cfg, z_context, actions_context, future_actions,
                    horizon, K, tau_ctx, K_max, context_window, device, "teacher")
            z_gt = z_all[:, C:C + horizon]                               # GT future latents
            div_pt += (z_pol - z_tch).pow(2).mean(dim=(0, 2, 3)).double().cpu()
            div_tg += (z_tch - z_gt).pow(2).mean(dim=(0, 2, 3)).double().cpu()
            r_pol += rp.mean(dim=0).double().cpu(); r_tch += rt.mean(dim=0).double().cpu()
            c_pol += cp.mean(dim=0).double().cpu(); c_tch += ct.mean(dim=0).double().cpu()
            n_div += 1
            # decode one example to a side-by-side GIF
            if len(gif_cpu) < opts.max_gifs:
                ctx = z_context[:1]
                vids = []
                # Context-anchored decode in <=seq_len windows. The tokenizer is
                # length-limited and decodes generated latents well only when a
                # real context anchors them; decoding the full C+H>8 rollout at
                # once (or pure-generated chunks) renders blank past frame 8 and
                # falsely looks like world-model collapse. Anchoring + chunking
                # shows the true generated-frame quality. (GT panel uses the same
                # path so all three panels are comparable.)
                chunk = int(getattr(trainer.tokenizer_cfg, "seq_len", 8))
                for zg in (z_pol[:1], z_tch[:1], z_gt[:1]):
                    vids.append(_decode_anchored(trainer.tokenizer, ctx, zg[:1], chunk).cpu())  # (C+H,Cc,H,W)
                side = torch.cat(vids, dim=-1)                           # concat along width
                gif_cpu.append(side)
            batches += 1

    n_div = max(1, n_div)
    div_pt /= n_div; div_tg /= n_div
    r_pol /= n_div; r_tch /= n_div; c_pol /= n_div; c_tch /= n_div

    for idx, vid in enumerate(gif_cpu):
        p = out_dir / f"rollout_{idx:03d}_policy_vs_teacher_vs_gt.gif"
        save_video_gif(vid, p, fps=opts.gif_fps)
        gif_paths.append(p)

    steps = list(range(horizon))
    write_csv(out_dir / "rollout_latent_divergence.csv",
              ["step", "policy_vs_teacher_mse", "teacher_vs_gt_mse"],
              [[s, div_pt[s].item(), div_tg[s].item()] for s in steps])
    _safe_line_plot(out_dir / "rollout_latent_divergence.png", "Rollout latent divergence",
                   steps, [("policy_vs_teacher", div_pt.tolist(), C_POLICY),
                           ("teacher_vs_gt", div_tg.tolist(), C_GT)], "latent MSE")
    write_csv(out_dir / "rollout_reward_continue_traj.csv",
              ["step", "reward_policy", "reward_teacher", "continue_policy", "continue_teacher"],
              [[s, r_pol[s].item(), r_tch[s].item(), c_pol[s].item(), c_tch[s].item()] for s in steps])
    _safe_line_plot(out_dir / "rollout_reward_traj.png", "Predicted reward (rollout)",
                   steps, [("policy", r_pol.tolist(), C_POLICY),
                           ("teacher", r_tch.tolist(), C_TEACHER)], "reward")
    _safe_line_plot(out_dir / "rollout_continue_traj.png", "Predicted continue (rollout)",
                   steps, [("policy", c_pol.tolist(), C_POLICY),
                           ("teacher", c_tch.tolist(), C_TEACHER)], "continue prob")
    return {
        "horizon": horizon, "K": K, "rollout_batches_processed": batches,
        "policy_vs_teacher_final_mse": div_pt[-1].item() if horizon else float("nan"),
        "teacher_vs_gt_final_mse": div_tg[-1].item() if horizon else float("nan"),
        "policy_vs_teacher_mean_mse": div_pt.mean().item() if horizon else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────
# Output writers (CSV / PNG) for sections 1-5
# ─────────────────────────────────────────────────────────────────────────
def _hist_counts(x: torch.Tensor, lo=-1.0, hi=1.0, bins=30) -> Tuple[List[float], List[float]]:
    edges = torch.linspace(lo, hi, bins + 1)
    idx = torch.bucketize(x.clamp(lo, hi - 1e-6), edges) - 1
    idx = idx.clamp(0, bins - 1)
    counts = torch.zeros(bins)
    counts.scatter_add_(0, idx, torch.ones_like(x))
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return centers, (counts / max(1, x.numel())).tolist()


def write_core_outputs(out_dir: Path, per_offset: Dict, policy_head):
    m0 = per_offset[0]
    # Section 1: policy
    pol = m0["policy"]
    if pol:
        write_csv(out_dir / "policy_per_dim.csv",
                  ["dim", "mse", "sigma_pred", "resid_std", "calib_ratio"],
                  [[d["dim"], d["mse"], d["sigma_pred"], d["resid_std"], d["calib_ratio"]]
                   for d in pol["per_dim"]])
        # action histograms per dim
        if isinstance(policy_head, CategoricalPolicyHead):
            # categorical: predicted marginal prob-mass vs empirical, on the bin grid.
            # Strictly more honest than a μ-histogram — exposes the multimodality the
            # categorical head exists to capture (a μ-scatter would hide it).
            pred_mass = pol.get("_pred_mass")          # (A, K)
            emp_hist = pol.get("_emp_hist")            # (A, K)
            centers = pol.get("_bin_centers")          # (K,)
            if pred_mass is not None and emp_hist is not None and centers is not None:
                A = pred_mass.shape[0]
                xs = [int(round(v * 100)) for v in centers.tolist()]  # bin center in centi-action units
                for k in range(A):
                    _safe_line_plot(out_dir / f"policy_action_hist_dim{k}.png",
                                   f"Action dim {k}: empirical vs predicted prob mass",
                                   xs, [("true", emp_hist[k].tolist(), C_TEACHER),
                                        ("pred_prob", pred_mass[k].tolist(), C_POLICY)],
                                   "probability")
        else:
            mu_s = pol.get("_mu_sample")
            a_s = pol.get("_a_sample")
            if mu_s is not None and a_s is not None and mu_s.numel() > 0:
                A = mu_s.shape[-1]
                for k in range(A):
                    ctr, true_h = _hist_counts(a_s[:, k])
                    _, pred_h = _hist_counts(mu_s[:, k])
                    xs = list(range(len(ctr)))
                    _safe_line_plot(out_dir / f"policy_action_hist_dim{k}.png",
                                   f"Action dim {k}: true vs predicted-mean (bin={ctr[0]:.2f}..{ctr[-1]:.2f})",
                                   xs, [("true", true_h, C_TEACHER), ("pred_mean", pred_h, C_POLICY)],
                                   "frequency")
    # Section 2: reward
    rew = m0["reward"]
    if rew and "by_true_value" in rew:
        write_csv(out_dir / "reward_by_true_value.csv",
                  ["true_value", "count", "pred_mean", "pred_std", "mse"],
                  [[d["true_value"], d["count"], d["pred_mean"], d["pred_std"], d["mse"]]
                   for d in rew["by_true_value"]])
    # Section 3: continue
    con = m0["continue"]
    if con:
        write_csv(out_dir / "continue_confusion.csv",
                  ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"],
                  [[con.get(f"cm_{k}") for k in
                    ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"]]])
        if con.get("reliability"):
            rel = con["reliability"]
            xs = list(range(len(rel)))
            _safe_line_plot(out_dir / "continue_reliability.png", "Continue reliability diagram",
                           xs, [("predicted", [b["pred_continue"] for b in rel], C_POLICY),
                                ("empirical", [b["emp_continue"] for b in rel], C_TEACHER)],
                           "continue prob")
            write_csv(out_dir / "continue_reliability.csv",
                      ["bin_lo", "bin_hi", "count", "pred_continue", "emp_continue"],
                      [[b["bin_lo"], b["bin_hi"], b["count"], b["pred_continue"], b["emp_continue"]] for b in rel])


def write_mtp_outputs(out_dir: Path, per_offset: Dict, max_offset: int):
    offsets = list(range(max_offset + 1))

    def col(path_metric):
        sect, key = path_metric
        return [per_offset[n][sect].get(key, float("nan")) for n in offsets]

    curves = {
        "mtp_action_mse_vs_offset": ("policy", "action_mse", "action MSE"),
        "mtp_action_nll_vs_offset": ("policy", "nll", "action NLL"),
        "mtp_reward_mse_vs_offset": ("reward", "mse", "reward MSE"),
        "mtp_reward_pearson_vs_offset": ("reward", "pearson", "reward Pearson"),
        "mtp_reward_event_auc_vs_offset": ("reward", "event_auc", "reward>0 AUC"),
        "mtp_continue_auc_vs_offset": ("continue", "auc", "done AUC"),
        "mtp_continue_bce_vs_offset": ("continue", "bce", "continue BCE"),
    }
    rows = [["offset"] + [name for name in curves]]
    table = {name: col((sect, key)) for name, (sect, key, _) in curves.items()}
    for n in offsets:
        rows.append([n] + [table[name][n] for name in curves])
    write_csv(out_dir / "mtp_metrics_vs_offset.csv", rows[0], rows[1:])
    for name, (sect, key, ylabel) in curves.items():
        _safe_line_plot(out_dir / f"{name}.png", name.replace("_", " "),
                       offsets, [(key, table[name], C_A)], ylabel)


def write_tau_outputs(out_dir: Path, tau_rows: List[Dict]):
    if not tau_rows:
        return
    keys = ["reward_mse", "reward_pearson", "reward_event_auc",
            "continue_auc", "continue_bce", "action_mse", "action_nll"]
    write_csv(out_dir / "head_metrics_vs_tau.csv", ["tau"] + keys,
              [[r["tau"]] + [r[k] for k in keys] for r in tau_rows])
    taus_int = list(range(len(tau_rows)))  # x-axis index (labels are tau values via CSV)
    for k in keys:
        _safe_line_plot(out_dir / f"{k}_vs_tau.png", f"{k} vs signal tau",
                       [int(round(r["tau"] * 100)) for r in tau_rows],
                       [(k, [r[k] for r in tau_rows], C_A)], k)


# ─────────────────────────────────────────────────────────────────────────
# Summary + wandb
# ─────────────────────────────────────────────────────────────────────────
def _clean(obj):
    """Drop private/sample tensors from a metrics dict for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def build_summary(opts, dynamics_cfg, readout_tau, num_tasks, per_offset, shuffle, rollout, max_offset):
    m0 = per_offset[0]
    return {
        "checkpoint": opts.dynamics_ckpt,
        "readout_tau": readout_tau,
        "mtp_length": dynamics_cfg.mtp_length,
        "max_offset": max_offset,
        "num_tasks": num_tasks,
        "seq_len": _eval_seq_len(opts, dynamics_cfg),
        "policy": _clean(m0["policy"]),
        "reward": _clean(m0["reward"]),
        "continue": _clean(m0["continue"]),
        "action_conditioning": shuffle,
        "rollout": rollout,
        "offset0_vs_maxoffset": {
            "action_mse": [m0["policy"].get("action_mse"),
                           per_offset[max_offset]["policy"].get("action_mse")],
            "reward_mse": [m0["reward"].get("mse"),
                           per_offset[max_offset]["reward"].get("mse")],
            "continue_auc": [m0["continue"].get("auc"),
                             per_offset[max_offset]["continue"].get("auc")],
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def log_wandb(opts, out_dir: Path, summary: Dict, gif_paths: List[Path]):
    if opts.wandb_disabled:
        return
    flat = {}
    for sect in ("policy", "reward", "continue"):
        for k, v in summary.get(sect, {}).items():
            if isinstance(v, (int, float)):
                flat[f"agent/{sect}/{k}"] = v
    for k, v in (summary.get("rollout") or {}).items():
        if isinstance(v, (int, float)):
            flat[f"agent/rollout/{k}"] = v
    wandb.log(flat)
    # images
    images = {}
    for png in sorted(out_dir.glob("*.png")):
        images[f"agent/plot/{png.stem}"] = wandb.Image(str(png))
    if images:
        wandb.log(images)
    # videos
    vids = {}
    for i, g in enumerate(gif_paths):
        vids[f"agent/rollout_gif_{i}"] = wandb.Video(str(g), fps=opts.gif_fps, format="gif")
    if vids:
        wandb.log(vids)
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def main(args: Optional[List[str]] = None) -> None:
    opts = build_parser().parse_args(args)
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not opts.wandb_disabled:
        mode = "offline" if opts.wandb_offline else "online"
        wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                   name=opts.wandb_name or f"agent_eval_{opts.task}_{datetime.utcnow():%Y%m%d_%H%M%S}",
                   mode=mode, config=vars(opts))

    trainer, dynamics_cfg, tokenizer_cfg, device, eval_seq_len, mtp_len = setup(opts)
    # stash tokenizer_cfg on the trainer for loader helpers
    trainer.tokenizer_cfg = tokenizer_cfg

    readout_tau = opts.readout_tau if opts.readout_tau >= 0 else (1.0 - dynamics_cfg.tau_ctx)
    max_offset = mtp_len if opts.mtp_max_offset <= 0 else min(opts.mtp_max_offset, mtp_len)

    print(f"[setup] seq_len={eval_seq_len} mtp_length={mtp_len} max_offset={max_offset} "
          f"readout_tau={readout_tau:.3f} device={device}")

    # Sections 1-4
    print("[1-4] core + MTP-offset metrics ...")
    per_offset, processed = run_core_and_mtp(trainer, dynamics_cfg, opts, device, readout_tau, max_offset)
    write_core_outputs(out_dir, per_offset, trainer.policy_head)
    write_mtp_outputs(out_dir, per_offset, max_offset)

    # Section 5
    taus = [float(x) for x in opts.tau_sweep.split(",") if x.strip()]
    print(f"[5] noise robustness sweep over tau={taus} ...")
    tau_rows = run_noise_sweep(trainer, dynamics_cfg, opts, device, taus)
    write_tau_outputs(out_dir, tau_rows)

    # Section 6
    print("[6] action-conditioning shuffle test ...")
    shuffle = run_action_conditioning(trainer, dynamics_cfg, opts, device, readout_tau)
    write_csv(out_dir / "action_shuffle_agent.csv",
              ["metric", "value"],
              [["reward_abs_delta", shuffle["reward_abs_delta"]]] +
              [[f"policy_mu_abs_delta_dim{i}", v]
               for i, v in enumerate(shuffle["policy_mu_abs_delta_per_dim"])])

    # Section 7
    print("[7] behavioral rollout (policy vs teacher) + video ...")
    gif_paths: List[Path] = []
    rollout = run_rollout(trainer, dynamics_cfg, opts, device, out_dir, gif_paths)

    # Summary + wandb
    summary = build_summary(opts, dynamics_cfg, readout_tau, getattr(trainer.model.agent_embedding, "num_tasks", 1),
                            per_offset, shuffle, rollout, max_offset)
    summary["tau_sweep"] = tau_rows
    summary["processed_batches"] = processed
    summary["buffer_truncated"] = any(per_offset[n]["truncated"] for n in per_offset)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_wandb(opts, out_dir, summary, gif_paths)

    # console digest
    m0 = per_offset[0]
    print("\n=== Agent eval summary (offset 0) ===")
    print(f"  policy   : action_mse={m0['policy'].get('action_mse'):.5f} "
          f"nll={m0['policy'].get('nll'):.4f}")
    print(f"  reward   : mse={m0['reward'].get('mse'):.5f} "
          f"pearson={m0['reward'].get('pearson'):.4f} event_auc={m0['reward'].get('event_auc')}")
    print(f"  continue : auc={m0['continue'].get('auc')} bce={m0['continue'].get('bce'):.4f} "
          f"done_positives={m0['continue'].get('done_positive_count')}")
    print(f"  rollout  : policy_vs_teacher_mean_mse={rollout.get('policy_vs_teacher_mean_mse')}")
    print(f"  outputs  -> {out_dir}")


if __name__ == "__main__":
    main()
