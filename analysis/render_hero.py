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

    INK, INK2, GREY2, EDGE, GRID = "#18181b", "#3f3f46", "#71717a", "#d4d4d8", "#ececef"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 6})

    def prep(frame):
        """Display prep only: trim empty margins (never the top, where the cup rides
        when carrying) and apply a mild uniform brightness lift for a white page."""
        f = frame[6:442, 24:456].astype(np.float32) * 1.15
        return np.clip(f, 0, 255).astype(np.uint8)

    def still(ax, img, label):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EDGE); sp.set_linewidth(0.9)
        ax.set_xlabel(label, fontsize=9, color=GREY2, labelpad=4)

    def chart(ax, n, ymax, bottom_axis):
        ax.set_xlim(0, n); ax.set_ylim(-20, ymax)
        ax.set_yticks(np.arange(0, ymax, 200))
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#a1a1aa")
        ax.tick_params(colors=INK2, labelsize=8.5, length=3)
        ax.set_ylabel("cumulative reward", fontsize=9, color=INK2, labelpad=4)
        if bottom_axis:
            ax.set_xlabel("decision step", fontsize=9, color=INK2)
        else:
            ax.tick_params(labelbottom=False)

    def end_value(ax, x, y, text, color):
        ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", color=color, fontsize=12, fontweight="bold")

    def row_header(fig, y, name, sub, chip):
        t = fig.text(0.010, y, name, fontsize=13, fontweight="bold", color=INK, va="bottom")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.006, y, "— " + sub, fontsize=11, color=INK2, va="bottom")
        fig.text(0.985, y, chip, fontsize=10, color=GREY2, va="bottom", ha="right")

    def make_static():
        fe, pe, ae, _, _ = _load("exploiting_rl21_seed*.npz")
        fh, ph, ah, _, _ = _load("healthy_rl24_seed*.npz")
        n = len(pe)
        ymax = max(pe.max(), ph.max(), ah.max()) * 1.08
        fig = plt.figure(figsize=(13.0, 7.4), dpi=150)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.14, 1.75], wspace=0.08,
                              hspace=0.55, left=0.010, right=0.985, top=0.835, bottom=0.08)

        # --- row 0: exploiting -------------------------------------------------
        times = [0, (len(fe) - 1) // 2, len(fe) - 1]
        for c, t in enumerate(times):
            still(fig.add_subplot(gs[0, c]), prep(fe[t]), f"t = {t}")
        ax = fig.add_subplot(gs[0, 4]); chart(ax, n, ymax, bottom_axis=False)
        ax.plot(np.arange(n), pe, color=BLUE, lw=2.6, solid_capstyle="round")
        ax.plot(np.arange(n), ae, color=ORANGE, lw=2.6, solid_capstyle="round")
        t0 = int(np.argmax(pe > 1.0))
        ax.axvline(t0, color=GREY2, lw=0.8, ls=(0, (2, 3)))
        ax.annotate(f"belief starts accruing, t = {t0}", xy=(t0, ymax * 0.93),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=GREY2)
        ax.annotate("reward model's belief", xy=(0.27, 0.36), xycoords="axes fraction",
                    color=BLUE, fontsize=10.5, fontweight="bold")
        ax.annotate("reality — never caught", xy=(0.97, 0.07), xycoords="axes fraction",
                    ha="right", color=ORANGE, fontsize=10.5, fontweight="bold")
        end_value(ax, n - 1, pe[-1], f"{pe[-1]:.0f}", BLUE)
        end_value(ax, n - 1, 0, "0", ORANGE)
        row_header(fig, 0.868, "Exploiting run", "the policy its own reward model prefers",
                   "0 catches this episode · true catch rate 0.126")

        # --- row 1: healthy ------------------------------------------------------
        rewarded = np.nonzero(np.diff(ah, prepend=0.0) > 0)[0]   # ball in cup at these t
        times = [0, int(rewarded[len(rewarded) // 2]), int(rewarded[-1])]
        for c, t in enumerate(times):
            still(fig.add_subplot(gs[1, c]), prep(fh[t]), f"t = {t}")
        ax = fig.add_subplot(gs[1, 4]); chart(ax, n, ymax, bottom_axis=True)
        ax.plot(np.arange(n), ph, color=BLUE, lw=2.6, solid_capstyle="round")
        ax.plot(np.arange(n), ah, color=ORANGE, lw=2.6, solid_capstyle="round")
        ax.annotate("belief ≈ reality", xy=(0.30, 0.68), xycoords="axes fraction",
                    color=INK, fontsize=10.5, fontweight="bold")
        end_value(ax, n - 1, ah[-1], f"{ah[-1]:.0f}", ORANGE)
        row_header(fig, 0.418, "Healthy run", "same recipe, different Phase-2 draw",
                   "caught and held · true catch rate 0.704")

        fig.suptitle("Same recipe, same data, same pretrained base — "
                     "only the finetune draw differs",
                     fontsize=15, fontweight="bold", color=INK, y=0.975)
        fig.savefig(OUT / "hero_static.png", bbox_inches="tight", facecolor="white", dpi=150)
        plt.close(fig)
        print("hero_static.png saved")

    def make_gif(cell_glob, title, out_name, stride=4, fps=14):
        frames, pred, actual, ret, _ = _load(cell_glob)
        n = len(pred)
        ymax = max(pred.max(), actual.max(), 1.0) * 1.10
        imgs = []
        fig = plt.figure(figsize=(9.8, 4.3), dpi=100)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.12, 1.2], left=0.012, right=0.965,
                              top=0.845, bottom=0.145, wspace=0.05)
        ax_im, ax_cv = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])
        fig.suptitle(title, fontsize=13, y=0.965, fontweight="bold", color=INK)
        im = ax_im.imshow(prep(frames[0]))
        ax_im.set_xticks([]); ax_im.set_yticks([])
        for sp in ax_im.spines.values():
            sp.set_color(EDGE); sp.set_linewidth(0.9)
        chart(ax_cv, n, ymax, bottom_axis=True)
        (l1,) = ax_cv.plot([], [], color=BLUE, lw=2.6)
        (l2,) = ax_cv.plot([], [], color=ORANGE, lw=2.6)
        ax_cv.annotate("reward model's belief", xy=(0.03, 0.90), xycoords="axes fraction",
                       color=BLUE, fontsize=10.5, fontweight="bold")
        ax_cv.annotate("reality", xy=(0.03, 0.82), xycoords="axes fraction",
                       color=ORANGE, fontsize=10.5, fontweight="bold")
        txt = ax_cv.text(0.97, 0.06, "", transform=ax_cv.transAxes, ha="right",
                         fontsize=11.5, fontweight="bold")
        tl = ax_im.set_xlabel("t = 0", fontsize=9, color=GREY2, labelpad=4)
        for i in range(0, n, stride):
            im.set_data(prep(frames[min(i, len(frames) - 1)]))
            tl.set_text(f"t = {i}")
            l1.set_data(np.arange(i + 1), pred[:i + 1])
            l2.set_data(np.arange(i + 1), actual[:i + 1])
            gap = pred[i] - actual[i]
            txt.set_text(f"belief − reality = {gap:+.0f}")
            txt.set_color(BLUE if gap > 5 else GREY2)
            fig.canvas.draw()
            imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
        imgs[0].save(OUT / out_name, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        print(f"{out_name}: {len(imgs)} frames, {(OUT / out_name).stat().st_size / 1e6:.1f} MB")

    make_static()
    make_gif("exploiting_rl21_seed*.npz",
             "Hallucinated success: the reward model believes; the ball never lands in the cup",
             "hero_exploiting.gif")
    make_gif("healthy_rl24_seed*.npz",
             "Healthy run: belief tracks reality; the ball is caught",
             "hero_healthy.gif")



def site_panel():
    """Column-width cut for the personal site: one still + one chart per row, larger
    type, so the figure stays legible at ~1000 px wide (the README composite does not)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GREY2, EDGE, GRID = "#18181b", "#3f3f46", "#71717a", "#d4d4d8", "#ececef"
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    def prep(frame):
        f = frame[6:442, 24:456].astype(np.float32) * 1.15
        return np.clip(f, 0, 255).astype(np.uint8)

    fe, pe, ae, _, _ = _load("exploiting_rl21_seed*.npz")
    fh, ph, ah, _, _ = _load("healthy_rl24_seed*.npz")
    n = len(pe); ymax = max(pe.max(), ph.max(), ah.max()) * 1.08
    rewarded = np.nonzero(np.diff(ah, prepend=0.0) > 0)[0]
    rows = [("Exploiting run", "0 catches this episode  ·  true catch rate 0.126", fe[len(fe) - 1], pe, ae, 0),
            ("Healthy run, different finetune draw", "caught and held  ·  true catch rate 0.704",
             fh[int(rewarded[-1])], ph, ah, 1)]
    fig = plt.figure(figsize=(11.0, 7.6), dpi=150); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.10, 2.05], wspace=0.04, hspace=0.62,
                          left=0.008, right=0.975, top=0.90, bottom=0.085)
    for name, chip, still, pred, act, r in rows:
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(prep(still)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(EDGE); sp.set_linewidth(1.0)
        ax.set_xlabel("t = 500" if r == 0 else f"t = {int(rewarded[-1])}", fontsize=11, color=GREY2, labelpad=5)
        ax = fig.add_subplot(gs[r, 2])
        ax.set_xlim(0, n); ax.set_ylim(-20, ymax); ax.set_yticks(np.arange(0, ymax, 200))
        ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#a1a1aa")
        ax.tick_params(colors=INK2, labelsize=10.5, length=3)
        ax.set_ylabel("cumulative reward", fontsize=11, color=INK2, labelpad=5)
        if r == 1: ax.set_xlabel("decision step", fontsize=11, color=INK2)
        else: ax.tick_params(labelbottom=False)
        ax.plot(np.arange(n), pred, color=BLUE, lw=3.0, solid_capstyle="round")
        ax.plot(np.arange(n), act, color=ORANGE, lw=3.0, solid_capstyle="round")
        if r == 0:
            t0 = int(np.argmax(pred > 1.0))
            ax.axvline(t0, color=GREY2, lw=0.9, ls=(0, (2, 3)))
            ax.annotate(f"belief starts accruing, t = {t0}", xy=(t0, ymax * 0.92), xytext=(-7, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=10, color=GREY2)
            ax.annotate("reward model's belief", xy=(0.27, 0.40), xycoords="axes fraction",
                        color=BLUE, fontsize=12.5, fontweight="bold")
            ax.annotate("reality — never caught", xy=(0.97, 0.07), xycoords="axes fraction",
                        ha="right", color=ORANGE, fontsize=12.5, fontweight="bold")
            for y, txt, c in ((pred[-1], f"{pred[-1]:.0f}", BLUE), (0, "0", ORANGE)):
                ax.annotate(txt, xy=(n - 1, y), xytext=(7, 0), textcoords="offset points",
                            va="center", color=c, fontsize=14, fontweight="bold")
        else:
            ax.annotate("belief ≈ reality", xy=(0.30, 0.68), xycoords="axes fraction",
                        color=INK, fontsize=12.5, fontweight="bold")
            ax.annotate(f"{act[-1]:.0f}", xy=(n - 1, act[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", color=ORANGE, fontsize=14, fontweight="bold")
        y_hdr = 0.935 if r == 0 else 0.455
        fig.text(0.008, y_hdr, name, fontsize=15, fontweight="bold", color=INK, va="bottom")
        fig.text(0.975, y_hdr, chip, fontsize=11.5, color=GREY2, va="bottom", ha="right")
    fig.savefig(OUT / "hero_site.png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig); print("hero_site.png saved")

    # ---- single-row variants: mobile (exploiting row, big type) and 1200x630 social card ----
    for out_name, figsize, chip_fs, title_fs in (("hero_site_mobile.png", (6.67, 4.27), 9.5, 13),
                                                  ("og_image.png", (8.0, 4.2), 10.5, 14)):
        fig = plt.figure(figsize=figsize, dpi=150); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.08, 2.0], wspace=0.04,
                              left=0.015, right=0.975, top=0.80, bottom=0.16)
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(prep(fe[len(fe) - 1])); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(EDGE); sp.set_linewidth(1.0)
        ax.set_xlabel("t = 500", fontsize=10, color=GREY2, labelpad=4)
        ax = fig.add_subplot(gs[0, 2])
        ax.set_xlim(0, n); ax.set_ylim(-20, pe.max() * 1.18); ax.set_yticks(np.arange(0, pe.max() * 1.18, 200))
        ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#a1a1aa")
        ax.tick_params(colors=INK2, labelsize=10, length=3)
        ax.set_xlabel("decision step", fontsize=10.5, color=INK2)
        ax.plot(np.arange(n), pe, color=BLUE, lw=3.2, solid_capstyle="round")
        ax.plot(np.arange(n), ae, color=ORANGE, lw=3.2, solid_capstyle="round")
        ax.annotate("reward model's belief", xy=(0.22, 0.42), xycoords="axes fraction",
                    color=BLUE, fontsize=12.5, fontweight="bold")
        ax.annotate("reality — never caught", xy=(0.97, 0.08), xycoords="axes fraction",
                    ha="right", color=ORANGE, fontsize=12.5, fontweight="bold")
        for y, txt, c in ((pe[-1], f"{pe[-1]:.0f}", BLUE), (0, "0", ORANGE)):
            ax.annotate(txt, xy=(n - 1, y), xytext=(7, 0), textcoords="offset points",
                        va="center", color=c, fontsize=15, fontweight="bold")
        fig.text(0.015, 0.90, "The policy its own reward model prefers", fontsize=title_fs,
                 fontweight="bold", color=INK, va="center")
        fig.text(0.975, 0.90, "0 catches this episode  ·  true catch rate 0.126", fontsize=chip_fs,
                 color=GREY2, va="center", ha="right")
        fig.savefig(OUT / out_name, facecolor="white", dpi=150)   # exact pixel size, no tight-crop
        plt.close(fig); print(f"{out_name} saved")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "compose"
    {"render": render, "compose": compose, "site": site_panel}[step]()
