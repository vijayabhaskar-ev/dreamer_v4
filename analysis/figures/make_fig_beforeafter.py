"""Figure 1: gain of imagination training over its own BC parent, per run, defective vs corrected pipeline.

Data (nothing typed in): old pipeline = code repo evaluation/tmlr-* per-board episodes.csv (the files behind
\\oldAnchorGain, \\oldRetrain*); corrected pipeline = backup phase3/analysis/p3_runs.csv (the file behind
\\gainMeanExact). Gain = paired difference in catch rate on the same 500 games, in points; bars = +-1 paired SE.
Marks: ink line = mean of the model's runs; blue band (right panel) = mean of the 8 retrain means with its 95% t interval
over the 8 means (df 7), the same estimand as \\gainMeanExact [\\gainLoExact, \\gainHiExact]; the run log prints it
so it can be checked against generated/numbers.tex. The per-panel captions ("2 of 3 retrains hurt", "all 8 retrains
gained") are counted from the data. The left panel's tick labels (first model, A, B, C) are the defective pipeline's
anchor model and its three retrains; they are deliberately not numbered, so they cannot be confused with the
corrected draws 10 to 17 on the right (panel review F23).
Style: Okabe-Ito pair (vermillion = defective, blue = corrected) as in the v1 figures, serif 9 pt to match the
body text, top/right spines off, legend below the panels, vector PDF (figures4papers conventions via the
scientific-figure-making skill).
Run: python figures/make_fig_beforeafter.py   (python with matplotlib and scipy, e.g. the playground1 conda env;
scipy is used only for the t quantile of the interval; writes fig_beforeafter.pdf next to itself)
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]                                   # repository root
E = CODE / "evaluation"
P3 = CODE / "results_corrected/p3_runs.csv"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
VERM, BLUE, INK, GRID, TINT = "#D55E00", "#0072B2", "0.15", "0.88", "0.955"


def load_eps(path, policy):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["policy"] == policy:
                out[int(r["seed"])] = int(r["caught"])
    return out


def paired(child, parent):
    boards = sorted(parent); assert len(boards) == 500 and set(child) == set(parent)
    d = 100.0 * np.array([child[b] - parent[b] for b in boards], float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


# ---- defective pipeline: anchor model (6 runs) + 3 retrains x 2 runs ----
FIRST = "first\nmodel"
bc0 = load_eps(E / "tmlr-n500-categorical/episodes.csv", "bc")
anchor = [load_eps(E / "tmlr-n500-categorical/episodes.csv", "phase3")]
for s in (1, 2, 3):
    m = load_eps(E / f"tmlr-cat-seed{s}-n200/episodes.csv", "phase3")
    m.update(load_eps(E / f"tmlr-cat-seed{s}-ext300/episodes.csv", "phase3")); anchor.append(m)
for s in (4, 5):
    anchor.append(load_eps(E / f"tmlr-cat-seed{s}-n500/episodes.csv", "phase3"))
old = {FIRST: [paired(c, bc0) for c in anchor]}
for lab, d, rls in (("A", 11, (21, 22)), ("B", 12, (23, 24)), ("C", 13, (25, 26))):
    b = load_eps(E / f"tmlr-bc-seed{d}-n500/episodes.csv", "bc")
    old[lab] = [paired(load_eps(E / f"tmlr-fact-bc{d}-rl{r}-n500/episodes.csv", "phase3"), b) for r in rls]

# ---- corrected pipeline: 8 draws x 3 seeds (same per-run numbers as the paper macros) ----
new = {}
for r in csv.DictReader(open(P3)):
    new.setdefault(r["draw"], []).append((float(r["delta_pts"]), float(r["se_paired_pts"])))
new = {k: new[k] for k in sorted(new, key=int)}

# ---- the headline and the per-panel counts, from the same numbers ----
dm_new = np.array([np.mean([m for m, _ in v]) for v in new.values()])
mean_new = dm_new.mean()
half = stats.t.ppf(0.975, len(dm_new) - 1) * dm_new.std(ddof=1) / np.sqrt(len(dm_new))
dm_old = {k: np.mean([m for m, _ in v]) for k, v in old.items()}
n_old_retrains = len(old) - 1
n_old_neg = sum(v < 0 for k, v in dm_old.items() if k != FIRST)
n_new_pos = int((dm_new > 0).sum())
print("defective draw means:", {k.replace("\n", " "): round(v, 1) for k, v in dm_old.items()})
print("corrected draw means:", {k: round(v, 1) for k, v in zip(new, dm_new)})
print(f"mean of the {len(dm_new)} draw means: {mean_new:.2f}  95% t interval [{mean_new - half:.2f}, {mean_new + half:.2f}]"
      "  <- check against \\gainMeanExact [\\gainLoExact, \\gainHiExact]")

# ---- plot ----
fig, (axo, axn) = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True,
                               gridspec_kw={"width_ratios": [len(old), len(new)], "wspace": 0.08})
for ax, data, color, title in ((axo, old, VERM, "defective pipeline"), (axn, new, BLUE, "corrected pipeline")):
    ax.axhspan(-45, 0, color=TINT, lw=0, zorder=0)                      # below zero = worse than its own BC parent
    ax.axhline(0, ls=(0, (1, 2)), lw=0.8, color="0.35", zorder=1)
    for i, (draw, runs) in enumerate(data.items()):
        k = len(runs); xs = i + (np.arange(k) - (k - 1) / 2) * (0.12 if k <= 3 else 0.11)
        for x, (m, se) in zip(xs, runs):
            ax.errorbar(x, m, yerr=se, fmt="o", ms=3.4, color=color, ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
        ax.hlines(np.mean([m for m, _ in runs]), i - 0.38, i + 0.38, color=INK, lw=1.5, zorder=5)   # mean of the retrain
    ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data))
    ax.set_xlim(-0.6, len(data) - 0.4)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(length=0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, color=GRID); ax.set_axisbelow(True)
# the headline: mean of the 8 retrain means with its 95% t interval
axn.axhspan(mean_new - half, mean_new + half, color=BLUE, alpha=0.13, lw=0, zorder=2)
axn.axhline(mean_new, color=BLUE, lw=0.9, zorder=2)
# the finding, said once per panel (counts come from the data above)
axo.text(0.03, 0.97, f"sign depends on the retrain\n({n_old_neg} of the {n_old_retrains} other retrains hurt)",
         transform=axo.transAxes, ha="left", va="top", fontsize=7.6, color=VERM)
axn.text(0.03, 0.08, f"all {n_new_pos} retrains gained", transform=axn.transAxes, ha="left", va="bottom",
         fontsize=7.6, color=BLUE)
axo.text(0.03, 0.03, "below 0: worse than its BC parent", transform=axo.transAxes, ha="left", va="bottom",
         fontsize=6.8, color="0.45")
axo.set_ylabel("gain over own BC parent (points)")
axo.set_xlabel("Phase 2 model", labelpad=3); axn.set_xlabel("Phase 2 retrain (draw)", labelpad=3)
axo.set_ylim(-45, 45); axo.set_yticks([-40, -20, 0, 20, 40])
handles = [plt.Line2D([], [], marker="o", ls="", ms=3.4, color="0.3"),
           plt.Line2D([], [], lw=1.5, color=INK),
           plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.25, ec=BLUE, lw=0.9)]
fig.legend(handles, ["one imagination run (±1 SE, 500 paired games)", "mean of the model's runs",
                     f"mean of the {len(dm_new)} retrains (95% CI)"],
           loc="lower center", ncol=3, frameon=False, fontsize=6.9, handlelength=1.3, columnspacing=1.0,
           handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.27)
out = HERE / "fig_beforeafter.pdf"
fig.savefig(out); print("wrote", out)
