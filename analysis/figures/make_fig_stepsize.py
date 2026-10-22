"""Figure 2: the fingerprint of defect 2 (the starved shortcut loss) in the held-out one-step prediction error.

Data (nothing typed in): backup eval_held_out/evaldyn_{OLD,JOINT}_e320/{d_bucket_metrics.csv, tau_d_joint_metrics.csv},
the same files behind \\errOldSmallStep, \\errOldShortcut*, \\errNew*, \\tdOld*, \\tdNew* (analysis/paper_numbers_v2.py,
C7 and C9). Both evaluations: dynamics/evaluate_dynamics.py, --split val (24 held-out episodes), seed 123, 4 context
frames, seq_len 8 (eval_held_out/INDEX.txt). OLD_e320 = the Phase 1 model of the defective pipeline (epoch 320);
JOINT_e320 = the corrected Phase 1 model (aligned actions + joint loss normalization, epoch 320).
Left: error by step size d (all signal levels), seven step sizes per model. Right: error by signal level of the input
at the step the rollouts use (d = 1/4, K = 4 denoising steps) and at the smallest step (1/64). The 0.8 to 1.0 signal
bin is left out: at d >= 1/4 it holds only frame 0, which the evaluator pins at signal level 0.9
(paper_numbers_v2.py, C9 comment), so it is not comparable with the other bins.
The annotations are computed from the data; the script asserts the two qualitative facts it states.
Style: Okabe-Ito pair (vermillion = defective, blue = corrected), serif 9 pt, as in make_fig_beforeafter.py.
Run: python figures/make_fig_stepsize.py   (python with matplotlib; writes fig_stepsize.pdf next to itself)
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BK = HERE.parents[1] / "results_corrected"
TAGS = {"old": "OLD_e320", "new": "JOINT_e320"}
D_GEN, TAU_MAX = 0.25, 0.8          # the rollout step size (K = 4); signal bins starting at or above this are dropped

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
VERM, BLUE, GRID = "#D55E00", "#0072B2", "0.88"
COLOR = {"old": VERM, "new": BLUE}
NAME = {"old": "defective pipeline (old Phase 1 model)", "new": "corrected pipeline"}


def by_step(tag):
    rows = csv.DictReader(open(BK / f"evaldyn_{tag}/d_bucket_metrics.csv"))
    return dict(sorted((float(r["d_value"]), float(r["latent_mse"])) for r in rows))


def by_signal(tag, d):
    """signal-level bin centre -> error, for one step size; bins with no entries and the frame-0 bin are skipped."""
    out = {}
    for r in csv.DictReader(open(BK / f"evaldyn_{tag}/tau_d_joint_metrics.csv")):
        lo, hi = float(r["tau_start"]), float(r["tau_end"])
        if abs(float(r["d_value"]) - d) < 1e-9 and int(r["count"]) > 0 and lo < TAU_MAX - 1e-6:
            out[(lo + hi) / 2] = float(r["latent_mse"])
    return dict(sorted(out.items()))


step = {k: by_step(t) for k, t in TAGS.items()}
d_small = min(step["old"])
sig = {(k, d): by_signal(t, d) for k, t in TAGS.items() for d in (D_GEN, d_small)}

# ---- the facts the annotations state, checked against the data ----
old_sc = [v for d, v in step["old"].items() if d != d_small]
ratio_lo, ratio_hi = min(old_sc) / step["old"][d_small], max(old_sc) / step["old"][d_small]
old_gen = list(sig[("old", D_GEN)].values()); new_gen = list(sig[("new", D_GEN)].values())
assert all(b > a for a, b in zip(old_gen, old_gen[1:])), "old model at d=1/4 should rise with cleaner input"
assert all(b < a for a, b in zip(new_gen, new_gen[1:])), "corrected model at d=1/4 should fall with cleaner input"
print("check vs numbers.tex: errOldSmallStep", round(step["old"][d_small], 4), "| errOldShortcut", round(min(old_sc), 2), round(max(old_sc), 2),
      "| ratios", round(ratio_lo), round(ratio_hi), "| errNew", round(min(step["new"].values()), 4), round(max(step["new"].values()), 4))
print("signal-level bins used:", list(sig[("old", D_GEN)]), "| old d=1/4:", [round(v, 4) for v in old_gen], "| new d=1/4:", [round(v, 4) for v in new_gen])

# ---- plot ----
fig, (axd, axt) = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
for ax in (axd, axt):
    ax.set_yscale("log"); ax.set_ylim(0.002, 1.5)
    ax.tick_params(length=0, pad=3); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, color=GRID); ax.set_axisbelow(True)

# left: error by step size
ds = sorted(step["old"])
for k in ("old", "new"):
    axd.plot(np.log2(ds), [step[k][d] for d in ds], "-o", color=COLOR[k], lw=1.2, ms=3.6, label=NAME[k], zorder=4)
axd.set_xticks(np.log2(ds)); axd.set_xticklabels([f"1/{int(round(1 / d))}" if d < 1 else "1" for d in ds])
axd.set_xlabel("step size", labelpad=3); axd.set_ylabel("prediction error (latent MSE)")
axd.set_title("by step size (all signal levels)", fontsize=9, pad=4)
axd.text(np.log2(ds[1]) - 0.1, min(old_sc) * 0.62, f"six shortcut steps: {ratio_lo:.0f} to {ratio_hi:.0f}$\\times$\nthe smallest step",
         ha="left", va="top", fontsize=7.4, color=VERM)
axd.legend(loc="lower right", frameon=False, fontsize=7.2, handlelength=1.6, borderaxespad=0.3)

# right: error by signal level, at the rollout step and at the smallest step
for k in ("old", "new"):
    for d, ls, mk, lab in ((D_GEN, "-", "o", f"step 1/{int(round(1 / D_GEN))} (the rollout step)"), (d_small, "--", "s", f"step 1/{int(round(1 / d_small))} (smallest)")):
        xs, ys = zip(*sig[(k, d)].items())
        axt.plot(xs, ys, ls, marker=mk, color=COLOR[k], lw=1.2, ms=3.4, mfc=COLOR[k] if d == D_GEN else "white", zorder=4,
                 label=lab if k == "new" else None)
axt.set_xticks(list(sig[("old", D_GEN)])); axt.set_xticklabels([f"{c - 0.1:.1f}–{c + 0.1:.1f}" for c in sig[("old", D_GEN)]])
axt.set_xlim(0, TAU_MAX); axt.set_xlabel("signal level of the input (1 = clean)", labelpad=3)
axt.set_title("by signal level", fontsize=9, pad=4)
axt.text(0.03, 0.97, "old model at the rollout step:\nerror rises as the input gets cleaner", transform=axt.transAxes,
         ha="left", va="top", fontsize=7.4, color=VERM)
l = axt.get_legend_handles_labels()[1]
h = [plt.Line2D([], [], color="0.3", lw=1.2, ls=hh.get_linestyle(), marker=hh.get_marker(), ms=3.4,
                mfc="0.3" if hh.get_linestyle() == "-" else "white", mec="0.3") for hh in axt.get_legend_handles_labels()[0]]
axt.legend(h, l, loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=7.2, handlelength=2.0, borderaxespad=0.3)

fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.17)
out = HERE / "fig_stepsize.pdf"
fig.savefig(out); print("wrote", out)
