"""End-to-end action-timing test that does NOT rely on reading the code.
Run:  MUJOCO_GL=egl python analysis/action_alignment_e2e.py     (needs torch + dm_control; loads the 5.9 GB npz into RAM)

Path under test (the real classes, unmodified):
    npz file -> tokenizer.dataset.OfflineDataset -> dynamics.dynamic_model.DynamicsModel.forward
A forward PRE-hook on model.action_embedding.proj records the exact tensor the model turns into
the per-slot action token. Slot t also holds frame t's latent and predicts frame t.

Physics: cup acceleration acc(t) = p(t+1) - 2p(t) + p(t-1) is caused by the forces applied during
[t-1, t) and [t, t+1).  In a CORRECT pipeline slot t holds "the action that produced frame t",
i.e. the force during [t-1, t), so acc(t) must load on slot lags k = 0 and k = +1.
If the actions enter one step stale, the same forces sit one slot later: k = +1 and k = +2.
"""
import copy, os, sys, tempfile
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("ALIGN_CTRL_DIR", tempfile.gettempdir())   # where the ~300 MB control npz is cached
sys.path.insert(0, REPO)
from tokenizer.dataset import OfflineDataset          # the real loader
from tokenizer.config import TokenizerConfig
from dynamics.config import DynamicsConfig
from dynamics.dynamic_model import DynamicsModel      # the real model

LAGS = [-2, -1, 0, 1, 2, 3]
SEQ_LEN, BATCH, NBATCH = 64, 16, 30
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def erode(m, n=1):
    for _ in range(n):
        p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
        m = (p[:, 1:-1, 1:-1] & p[:, :-2, 1:-1] & p[:, 2:, 1:-1] & p[:, 1:-1, :-2] & p[:, 1:-1, 2:]
             & p[:, :-2, :-2] & p[:, :-2, 2:] & p[:, 2:, :-2] & p[:, 2:, 2:])
    return m


def track(fr):
    """fr (T,H,W,3) uint8 -> cup centroid (T,2) as (col, -row)."""
    f = fr.astype(np.int16)
    m = (f[..., 0] - f[..., 2]) > 40
    core = erode(m, 2)
    T, H, W = m.shape
    rows, cols = np.mgrid[0:H, 0:W]
    out = np.full((T, 2), np.nan)
    for t in range(T):
        mt = m[t]
        if core[t].any():
            br, bc = rows[core[t]].mean(), cols[core[t]].mean()
            mt = mt & (((rows - br) ** 2 + (cols - bc) ** 2) > 6.0 ** 2)
        if mt.sum() >= 15:
            out[t] = (cols[mt].mean(), -rows[mt].mean())
    return out


def build_model():
    class StubTokenizer(torch.nn.Module):            # only .config is used by the action path
        def __init__(self):
            super().__init__()
            self.config = TokenizerConfig()
    tok = StubTokenizer()
    cfg = DynamicsConfig.from_tokenizer(tok.config, action_dim=2, depth=1)
    model = DynamicsModel(cfg, tok).to(DEV).eval()
    captured = []
    model.action_embedding.proj.register_forward_pre_hook(
        lambda mod, inp: captured.append(inp[0].detach().float().cpu().numpy()))
    return model, cfg, captured


def run_arm(name, ds, model, cfg, captured):
    rows = {0: ([], []), 1: ([], [])}
    n_seq = 0
    it = iter(ds)
    for _ in range(NBATCH):
        frames, actions, _, _ = next(it)                       # exactly what the trainer receives
        B, T = frames.shape[:2]
        z = torch.randn(B, T, cfg.num_latent_tokens, cfg.latent_input_dim, device=DEV)
        tau = torch.full((B, T), 0.5, device=DEV); d = torch.full((B, T), 1.0 / cfg.K_max, device=DEV)
        captured.clear()
        with torch.no_grad():
            model(z, actions.to(DEV), tau, d)                  # real forward; hook records the slot actions
        slot = captured[0]                                     # (B, T, A): what the model consumed per slot
        assert slot.shape == (B, T, 2), slot.shape
        u8 = (frames * 255.0).round().clamp(0, 255).byte().permute(0, 1, 3, 4, 2).numpy()
        for b in range(B):
            p2 = track(u8[b]); n_seq += 1
            for dim in (0, 1):
                p = p2[:, dim]
                for t in range(3, T - 3):                      # t+k in [1, T-1]: never touches the pad slot 0
                    if not np.isfinite(p[t - 1:t + 2]).all():
                        continue
                    rows[dim][0].append([slot[b, t + k, dim] for k in LAGS] + [p[t] - p[t - 1], 1.0])
                    rows[dim][1].append(p[t + 1] - 2 * p[t] + p[t - 1])
    print(f"\n=== {name}   ({n_seq} windows of {SEQ_LEN} frames) ===")
    for dim, label in ((0, "x"), (1, "z")):
        X, y = np.array(rows[dim][0]), np.array(rows[dim][1])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1])))
        print(f"  {label}  n={len(y)}  R^2={1 - res.var() / y.var():.3f}   "
              + "  ".join(f"k={k:+d}:{b:+.2f}(t={b / s:+.0f})" for k, b, s in zip(LAGS, beta, se)))


def make_control_npz(path, n_eps=24, T=251, seed=0):
    """Known timing, generated the way the pipeline's own DMControlDataset does it:
    render frame t -> choose action -> store it at index t -> step the env with it."""
    from dm_control import suite
    rng = np.random.default_rng(seed)
    F = np.empty((n_eps, T, 128, 128, 3), np.uint8); A = np.empty((n_eps, T, 2), np.float32)
    for e in range(n_eps):
        env = suite.load("ball_in_cup", "catch", task_kwargs={"random": int(rng.integers(1 << 30))})
        env.reset(); a = np.zeros(2)
        for t in range(T):
            img = env.physics.render(height=224, width=224, camera_id=0)
            F[e, t] = np.asarray(Image.fromarray(img).resize((128, 128), Image.BILINEAR))
            a = np.clip(0.5 * a + rng.normal(0, 0.6, 2), -1, 1)
            A[e, t] = a
            for _ in range(2):
                env.step(a)
    np.savez(path, frames=F, actions=A, rewards=np.zeros((n_eps, T), np.float32), dones=np.zeros((n_eps, T), np.float32))


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    model, cfg, captured = build_model()

    ctrl = os.path.join(SCRATCH, "control_known_timing.npz")
    if not os.path.exists(ctrl):
        make_control_npz(ctrl)
    ds_c = OfflineDataset(ctrl, seq_len=SEQ_LEN, batch_size=BATCH, steps_per_epoch=NBATCH)
    run_arm("ARM 1  CONTROL with KNOWN timing (action t applied at frame t) -> real loader -> real model", ds_c, model, cfg, captured)

    hansen = os.path.join(REPO, "ball_in_cup_catch.npz")
    ds_h = OfflineDataset(hansen, seq_len=SEQ_LEN, batch_size=BATCH, steps_per_epoch=NBATCH)   # same class + file as training
    run_arm("ARM 2  HANSEN npz exactly as training read it -> real loader -> real model", ds_h, model, cfg, captured)

    ds_f = copy.copy(ds_h)                                                                       # same frames object
    acts = np.asarray(ds_h.actions)
    ds_f.actions = np.concatenate([acts[:, 1:], np.zeros_like(acts[:, :1])], axis=1)             # Hansen's own rule: read one row ahead
    run_arm("ARM 3  HANSEN npz with Hansen's own rule (actions read one row ahead) -> real loader -> real model", ds_f, model, cfg, captured)
