import os
"""Controlled test: which stored action index drives the cup's acceleration around frame t?
acc(t) = p(t+1) - 2p(t) + p(t-1) is caused by the forces applied during [t-1,t) and [t,t+1).
  convention A (action[t] applied AT frame t)      -> those forces are stored at k = -1 and k = 0
  convention B (action[t] is what LED TO frame t)  -> those forces are stored at k =  0 and k = +1
The pipeline conditions the transition t -> t+1 on stored index t, i.e. it assumes convention A."""
import ast, struct, zipfile, numpy as np
from PIL import Image

LAGS = [-2, -1, 0, 1, 2]

def erode(m, n=1):
    for _ in range(n):
        p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
        m = (p[:, 1:-1, 1:-1] & p[:, :-2, 1:-1] & p[:, 2:, 1:-1] & p[:, 1:-1, :-2] & p[:, 1:-1, 2:]
             & p[:, :-2, :-2] & p[:, :-2, 2:] & p[:, 2:, :-2] & p[:, 2:, 2:])
    return m

def track(fr):
    """fr: (T,H,W,3) uint8 -> cup centroid (T,2) as (col, -row); NaN where detection fails."""
    f = fr.astype(np.int16)
    m = (f[..., 0] - f[..., 2]) > 40                      # orange/tan vs blue background
    core = erode(m, 2)                                    # 5x5 erosion: only the solid ball survives
    T, H, W = m.shape
    rows, cols = np.mgrid[0:H, 0:W]
    out = np.full((T, 2), np.nan)
    for t in range(T):
        mt = m[t]
        if core[t].any():
            br, bc = rows[core[t]].mean(), cols[core[t]].mean()
            mt = mt & (((rows - br) ** 2 + (cols - bc) ** 2) > 6.0 ** 2)   # drop the ball
        if mt.sum() >= 15:
            out[t] = (cols[mt].mean(), -rows[mt].mean())
    return out

def analyse(name, episodes):
    """episodes: list of (frames, actions). OLS of acc(t) on u[t+k], k in LAGS, + v(t-1) + 1."""
    print(f"\n=== {name} ===")
    ok_frac = []
    for dim, label in ((0, "x (horizontal force -> horizontal accel)"), (1, "z (vertical force   -> vertical accel)")):
        X, y = [], []
        for fr, u in episodes:
            p = track(fr)[:, dim]; ok_frac.append(np.isfinite(p).mean())
            T = len(p)
            for t in range(3, T - 3):
                w = p[t - 1:t + 2]
                if not np.isfinite(w).all(): continue
                X.append([u[t + k, dim] for k in LAGS] + [p[t] - p[t - 1], 1.0])
                y.append(p[t + 1] - 2 * p[t] + p[t - 1])
        X, y = np.array(X), np.array(y)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        cov = np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1])
        se = np.sqrt(np.diag(cov))
        r2 = 1 - res.var() / y.var()
        print(f"  {label}   n={len(y)}  R^2={r2:.3f}")
        print("     " + "  ".join(f"k={k:+d}: {b:+.3f} (t={b/s:+.0f})" for k, b, s in zip(LAGS, beta, se)))
    print(f"  cup detected in {100*np.mean(ok_frac):.1f}% of frames")

# ---------- positive control: self-generated data with KNOWN timing ----------
def make_control(n_eps=24, T=251, seed=0):
    from dm_control import suite
    rng = np.random.default_rng(seed); eps = []
    for e in range(n_eps):
        env = suite.load("ball_in_cup", "catch", task_kwargs={"random": int(rng.integers(1 << 30))})
        env.reset(); frames, acts = [], []; a = np.zeros(2)
        for t in range(T):
            img = env.physics.render(height=224, width=224, camera_id=0)
            frames.append(np.asarray(Image.fromarray(img).resize((128, 128), Image.BILINEAR)))
            a = np.clip(0.5 * a + rng.normal(0, 0.6, 2), -1, 1)       # lag-1 autocorr ~0.5, like the demos
            acts.append(a.copy())
            for _ in range(2): env.step(a)                           # action_repeat = 2
        eps.append((np.stack(frames), np.stack(acts)))               # convention A by construction
    return eps

ctrl_A = make_control()
ctrl_B = [(fr, np.concatenate([np.zeros((1, 2)), u[:-1]])) for fr, u in ctrl_A]   # same data, stored one step late
analyse("CONTROL, stored as convention A (action[t] applied at frame t) -- what the pipeline assumes", ctrl_A)
analyse("CONTROL, same episodes deliberately stored as convention B (action[t] led TO frame t)", ctrl_B)

# ---------- Hansen's demos as converted (ball_in_cup_catch.npz) ----------
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ball_in_cup_catch.npz")  # repo root
zf = zipfile.ZipFile(path); info = zf.getinfo("frames.npy")
with open(path, "rb") as f:
    f.seek(info.header_offset); lh = f.read(30)
    fnlen, exlen = struct.unpack("<HH", lh[26:30]); ds = info.header_offset + 30 + fnlen + exlen
    f.seek(ds); f.read(6); major = f.read(1)[0]; f.read(1)
    hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
    hoff = 10 if major == 1 else 12; header = ast.literal_eval(f.read(hlen).decode("latin1"))
frames = np.memmap(path, dtype=np.uint8, mode="r", offset=ds + hoff + hlen, shape=header["shape"])
acts = np.load(path)["actions"]
hansen = [(np.asarray(frames[e]), acts[e]) for e in range(0, 240, 6)]           # 40 episodes across the buffer
analyse("HANSEN demos exactly as the training pipeline reads them", hansen)
