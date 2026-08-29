"""Hero-media generator: verified rollout videos + reward-head belief-vs-reality curves.

Produces assets/hero_static.png, assets/hero_exploiting.gif, assets/hero_healthy.gif —
the README's lead visuals: the exploiting run's reward head accumulating belief on an
episode where the ball never lands in the cup, next to a healthy run whose belief
tracks reality.

Usage (from the repo root):
    python -m analysis.render_hero render    # capture verified episodes (factorial ckpts + GPU)
    python -m analysis.render_hero compose   # build the media (numpy + matplotlib + PIL)

Honesty contract: every candidate episode is re-run with the recorded eval's exact
configuration (amp=False, readout=sample, per-board seeding, phase-3 policy loaded per
dynamics/evaluate_env.py:566) and is eligible only if its (return, caught) matches the
released evaluation/tmlr-*/episodes.csv row exactly. The model still sees 128x128
frames; only the *video* is rendered at 480x480 from the same physics states.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "assets" / "hero_work"
OUT = REPO / "assets"

TOK = REPO / "checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt"
CELLS = {
    "exploiting_rl21": dict(
        p2=REPO / "checkpoints-phase2-cat-seed11/final.pt",
        p3=REPO / "checkpoints-phase3-cat-bc11-rl21/epoch_15.pt",
        stored=REPO / "evaluation/tmlr-fact-bc11-rl21-n500/episodes.csv", n_scan=50),
    "healthy_rl24": dict(
        p2=REPO / "checkpoints-phase2-cat-seed12/final.pt",
        p3=REPO / "checkpoints-phase3-cat-bc12-rl24/epoch_15.pt",
        stored=REPO / "evaluation/tmlr-fact-bc12-rl24-n500/episodes.csv", n_scan=12),
}

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b7280"


# ======== render step (torch + dm_control) ========

def _stored_outcomes(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["policy"] == "phase3":
            out[int(r["seed"])] = (float(r["return"]), int(r["caught"]))
    return out


def render():
    import torch
    sys.path.insert(0, str(REPO))
    import dynamics.evaluate_env as ev

    class DualEnv(ev.DMCEnvWrapper):
        """Adds a 480px render of the same physics alongside the model's 128px view."""

        def __init__(self, *a, hi_size=480, capture=False, **k):
            super().__init__(*a, **k)
            self.hi_size, self.capture, self.hi_frames = hi_size, capture, []

        def _hi(self):
            return self.env.physics.render(height=self.hi_size, width=self.hi_size,
                                           camera_id=self.camera_id)

        def reset(self):
            px = super().reset()
            if self.capture:
                self.hi_frames = [self._hi()]
            return px

        def step(self, action):
            out = super().step(action)
            if self.capture:
                self.hi_frames.append(self._hi())
            return out

    cache = {}

    def run(cell, seed, capture):
        args = ["--phase2-ckpt", str(CELLS[cell]["p2"]), "--phase3-ckpt", str(CELLS[cell]["p3"]),
                "--tokenizer-ckpt", str(TOK), "--readout", "sample", "--wandb-disabled",
                "--policies", "phase3", "--device", "cuda"]  # amp OFF: matches the recorded eval
        opts = ev.build_parser().parse_args(args)
        if cell not in cache:
            trainer, cfg, device, nt = ev.setup_world_model(opts)
            # setup leaves the Phase-2 BC policy in policy_head; swap in Phase-3
            # exactly as evaluate_env.py's main loop does.
            ev.load_policy_head(trainer, str(CELLS[cell]["p3"]), device)
            cache[cell] = (trainer, cfg, device, nt)
        trainer, cfg, device, _ = cache[cell]
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        env = DualEnv(task=opts.task, image_size=128, camera_id=0, action_repeat=2,
                      seed=seed, capture=capture)
        res, _ = ev.run_episode(env, trainer, cfg, is_random=False, readout="sample",
                                device=device, max_steps=0, success_threshold=1.0,
                                rng=rng, collect_frames=False)
        return res, env.hi_frames

    WORK.mkdir(parents=True, exist_ok=True)
    picks = {}
    for cell, spec in CELLS.items():
        stored = _stored_outcomes(spec["stored"])
        rows = []
        for seed in range(spec["n_scan"]):
            r, _ = run(cell, seed, capture=False)
            gap = sum(r.pred_rewards) - sum(r.actual_rewards)
            sr, sc = stored[seed]
            match = abs(r.return_ - sr) < 1e-6 and int(r.caught) == sc
            rows.append((seed, r.return_, r.caught, gap, match))
            print(f"[{cell}] seed {seed}: return {r.return_:.0f} caught {r.caught} "
                  f"gap {gap:+.0f} stored=({sr:.0f},{sc}) match={match}", flush=True)
        matched = [x for x in rows if x[4]]
        print(f"[{cell}] reproduction: {len(matched)}/{len(rows)} boards match stored", flush=True)
        if cell.startswith("exploiting"):
            cand = [x for x in matched if not x[2]] or matched
            pick = max(cand, key=lambda x: x[3])[0]   # biggest belief-reality gap, no catch
        else:
            cand = [x for x in matched if x[2]] or matched
            pick = max(cand, key=lambda x: x[1])[0]   # best real return, caught
        picks[cell] = pick
        r, hi = run(cell, pick, capture=True)
        np.savez_compressed(WORK / f"{cell}_seed{pick}.npz",
                            frames=np.stack(hi).astype(np.uint8),
                            pred=np.array(r.pred_rewards, np.float32),
                            actual=np.array(r.actual_rewards, np.float32),
                            ret=r.return_, caught=int(r.caught))
        print(f"[{cell}] PICK seed {pick}: saved {len(hi)} frames, return {r.return_:.0f}", flush=True)
    json.dump(picks, open(WORK / "picks.json", "w"))


# ======== compose step (numpy + matplotlib + PIL only) ========

def _load(cell_glob):
    f = sorted(glob.glob(str(WORK / cell_glob)))[0]
    d = np.load(f)
    return d["frames"], np.cumsum(d["pred"]), np.cumsum(d["actual"]), float(d["ret"]), Path(f).stem


def compose():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    def make_gif(cell_glob, title, out_name, stride=4, fps=14):
        frames, pred, actual, ret, _ = _load(cell_glob)
        n = len(pred)
        ymax = max(pred.max(), actual.max(), 1.0) * 1.06
        imgs = []
        fig = plt.figure(figsize=(9.6, 4.15), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.12], left=0.005, right=0.975,
                              top=0.86, bottom=0.135, wspace=0.16)
        ax_im, ax_cv = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        fig.suptitle(title, fontsize=13.5, y=0.965, fontweight="bold")
        im = ax_im.imshow(frames[0]); ax_im.axis("off")
        (l1,) = ax_cv.plot([], [], color=BLUE, lw=2.4,
                           label="reward model's belief (cumulative predicted)")
        (l2,) = ax_cv.plot([], [], color=ORANGE, lw=2.4,
                           label="reality (cumulative actual reward)")
        ax_cv.set_xlim(0, n); ax_cv.set_ylim(0, ymax)
        ax_cv.set_xlabel("decision step"); ax_cv.set_ylabel("cumulative reward")
        ax_cv.legend(loc="upper left", frameon=False, fontsize=9.5)
        for s in ("top", "right"):
            ax_cv.spines[s].set_visible(False)
        txt = ax_cv.text(0.985, 0.06, "", transform=ax_cv.transAxes, ha="right",
                         fontsize=11, fontweight="bold")
        for i in range(0, n, stride):
            im.set_data(frames[min(i, len(frames) - 1)])
            l1.set_data(np.arange(i + 1), pred[:i + 1])
            l2.set_data(np.arange(i + 1), actual[:i + 1])
            gap = pred[i] - actual[i]
            txt.set_text(f"belief − reality = {gap:+.0f}")
            txt.set_color(BLUE if gap > 5 else GREY)
            fig.canvas.draw()
            imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
        imgs[0].save(OUT / out_name, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        print(f"{out_name}: {len(imgs)} frames, {(OUT / out_name).stat().st_size / 1e6:.1f} MB")

    def make_static():
        fe, pe, ae, _, _ = _load("exploiting_rl21_seed*.npz")
        fh, ph, ah, _, _ = _load("healthy_rl24_seed*.npz")
        n = len(pe)
        ymax = max(pe.max(), ph.max(), ah.max()) * 1.05
        fig = plt.figure(figsize=(12.6, 7.0), dpi=150)
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.65], hspace=0.42, wspace=0.06,
                              left=0.005, right=0.985, top=0.845, bottom=0.075)
        rows = [("Exploiting run", "the policy its own reward model prefers",
                 "true catch rate 0.126", fe, pe, ae, 0),
                ("Healthy run", "same recipe, different Phase-2 draw",
                 "true catch rate 0.704", fh, ph, ah, 1)]
        for name, sub, rate, fr, pred, act, r in rows:
            for c, pos in enumerate([0.05, 0.5, 0.98]):
                ax = fig.add_subplot(gs[r, c]); ax.axis("off")
                ax.imshow(fr[int(pos * (len(fr) - 1))])
                if r == 0:
                    ax.set_title(["start", "middle", "end"][c], fontsize=9, color=GREY, pad=3)
            ax = fig.add_subplot(gs[r, 3])
            ax.plot(np.arange(len(pred)), pred, color=BLUE, lw=2.4, label="reward model's belief")
            ax.plot(np.arange(len(act)), act, color=ORANGE, lw=2.4, label="reality")
            ax.set_xlim(0, n); ax.set_ylim(0, ymax)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_ylabel("cumulative reward", fontsize=9, labelpad=2)
            if r == 1:
                ax.set_xlabel("decision step", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            ax.legend(loc="upper left", frameon=False, fontsize=9)
            y_hdr = 0.878 if r == 0 else 0.442
            fig.text(0.005, y_hdr, f"{name} — {sub}", fontsize=12.5,
                     fontweight="bold", va="bottom")
            fig.text(0.985, y_hdr, rate, fontsize=11, color=GREY, va="bottom", ha="right")
        fig.suptitle("Same training recipe, same data, same pretrained base — "
                     "only the finetune draw differs",
                     fontsize=14, fontweight="bold", y=0.975)
        fig.savefig(OUT / "hero_static.png", bbox_inches="tight")
        print("hero_static.png saved")

    make_gif("exploiting_rl21_seed*.npz",
             "Hallucinated success: the reward model believes; the ball never lands in the cup",
             "hero_exploiting.gif")
    make_gif("healthy_rl24_seed*.npz",
             "Healthy run: belief tracks reality; the ball is caught",
             "hero_healthy.gif")
    make_static()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "compose"
    {"render": render, "compose": compose}[step]()
