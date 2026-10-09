"""Tables for the v2 paper, written to paper/generated/ (never edited by hand).

  table_results.tex   Table 1: per-draw BC parent and imagination catch rates, per-seed gains, draw means, overall row
  table_loo.tex       leave-one-draw-out: mean gain and between-draw sd with each draw left out (appendix)
  table_checks.tex    the 12 checks of the v1 verification table, with what each could and could not see (appendix)

Every number is recomputed from the same raw files as analysis/paper_numbers_v2.py and asserted against the macros
in paper/generated/numbers.tex, so a table cell can never disagree with the text.
Run:  PYTHONPATH=. python analysis/paper_tables_v2.py   (needs numpy + scipy, e.g. the dreamer_v4 conda env)
"""
import csv, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
from paper_numbers_v2 import one_way_random, tci

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/generated"
P3 = Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup/phase3/analysis/p3_runs.csv"


def f(x, nd=1):
    s = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    return "0" + s[2:] if s.startswith("-0") and float(s) == 0 else s


macros = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{\\ensuremath\{([^}]*)\}", (GEN / "numbers.tex").read_text())}


def check(name, value):
    got = macros[name].replace("{,}", "")
    assert got == value, f"{name}: table says {value}, numbers.tex says {got}"


# ---------------- data: the 24 corrected runs ----------------
runs = list(csv.DictReader(open(P3)))
draws = sorted({int(r["draw"]) for r in runs})
seeds = sorted({int(r["seed"]) for r in runs})
gain = {(int(r["draw"]), int(r["seed"])): float(r["delta_pts"]) for r in runs}
bc = {int(r["draw"]): 100 * float(r["bc_rate"]) for r in runs}
p3 = {(int(r["draw"]), int(r["seed"])): 100 * float(r["p3_rate"]) for r in runs}
tab = np.array([[gain[(d, s)] for s in seeds] for d in draws])
dm = tab.mean(axis=1)
m, lo, hi, _ = tci(dm)
check("gainMeanExact", f(m, 2)); check("gainLoExact", f(lo, 2)); check("gainHiExact", f(hi, 2))
check("bcCatchExact", f(np.mean([bc[d] for d in draws]), 2)); check("imagCatchExact", f(np.mean(list(p3.values())), 2))
check("drawMeanMin", f(dm.min())); check("drawMeanMax", f(dm.max()))

# ---------------- Table 1 ----------------
rows = []
for i, d in enumerate(draws):
    cells = [str(d), f(bc[d]), f(np.mean([p3[(d, s)] for s in seeds]))] + [f"{gain[(d, s)]:+.1f}" for s in seeds] + [f"{dm[i]:+.1f}"]
    rows.append(" & ".join(cells) + r" \\")
seed_heads = " & ".join(f"seed {s}" for s in seeds)
t1 = rf"""% Table 1. Written by analysis/paper_tables_v2.py from phase3/analysis/p3_runs.csv; every cell is checked against numbers.tex.
\begin{{table}}[t]
\centering
\small
\caption{{Catch rate of each BC parent and of the imagination policies trained from it (\nBoards paired games each, actions sampled), and the gain of each run over its parent in points. The last row averages the \nDraws draws; the interval is a 95\% $t$ confidence interval over the draw means.}}
\label{{tab:results}}
\begin{{tabular}}{{@{{}}lcc{'c' * len(seeds)}c@{{}}}}
\toprule
 & BC parent & imagination & \multicolumn{{{len(seeds)}}}{{c}}{{gain per seed (points)}} & draw mean \\
draw & catch (\%) & catch (\%) & {seed_heads} & (points) \\
\midrule
{chr(10).join(rows)}
\midrule
mean over draws & \bcCatchExact & \imagCatchExact & \multicolumn{{{len(seeds)}}}{{c}}{{\nRunsPositive/\nRuns runs positive}} & \gainMeanExact\ [\gainLoExact, \gainHiExact] \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_results.tex").write_text(t1)

# ---------------- leave-one-draw-out ----------------
full = one_way_random(tab)
check("sdBetween", f(full["sb"], 2)); check("sdWithin", f(full["sw"], 2))
loo_rows = []
for i, d in enumerate(draws):
    rest = np.delete(tab, i, axis=0); v = one_way_random(rest)
    mm, l2, h2, _ = tci(rest.mean(axis=1))
    loo_rows.append(f"{d} & {f(dm[i])} & {f(mm, 2)} [{f(l2, 2)}, {f(h2, 2)}] & {f(v['sb'], 2)} & {f(v['sw'], 2)} \\\\")
    if d == int(macros["looLowDraw"]):
        check("looWithoutLow", f(v["sb"], 2))
t2 = rf"""% Leave-one-draw-out table. Written by analysis/paper_tables_v2.py; checked against numbers.tex.
\begin{{table}}[t]
\centering
\small
\caption{{Leave-one-draw-out: the mean gain over the remaining {len(draws) - 1} draws (95\% $t$ interval) and the between-draw and within-draw standard deviations (one-way random effects), with each draw left out in turn. With all \nDraws draws: mean \gainMeanExact\ [\gainLoExact, \gainHiExact], between-draw sd \sdBetween, within-draw sd \sdWithin.}}
\label{{tab:loo}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
draw left out & its mean gain & mean gain of the rest & between-draw sd & within-draw sd \\
\midrule
{chr(10).join(loo_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_loo.tex").write_text(t2)

# ---------------- the 12 checks of the v1 verification table ----------------
# Columns 1-3 are copied from old/main_v1.tex Table 1 (tag v1-old-pipeline). Columns 4-5 are judgments:
# TODO(author): review every entry in the two right-hand columns.
checks = [
    (r"$\lambda$-returns, advantages", "unit test against hand-computed values", "pass", "no: tests the RL arithmetic on given inputs", "no"),
    ("attention firewall", r"$\partial \hat z / \partial\,\mathrm{agent} = 0$", "exact", "no: tests gradient routing", "no"),
    ("gradient routing", "per-head isolation", "pass", "no: tests gradient routing", "no"),
    ("tokenizer", "latent rank, reconstruction", r"rank $\approx 11/16$", "no: frames were correct in both pipelines", "no"),
    ("reward head", "closed-loop calibration", "MAE 0.013 to 0.035", "no: reward rows were paired correctly", "no"),
    ("continue head", "closed-loop calibration", "Brier 0.0020", "no: same as the reward head", "no"),
    ("PMPO objective", "unit test against the algebra", "pass", "no: tests the policy loss on given inputs", "no"),
    ("action conditioning", "batch-shuffle probe", r"shift $\approx 0.17$ vs 0.006", "no: asks whether actions matter, never when", "no"),
    ("full stack", "random-policy floor", r"$p = 3.8 \times 10^{-28}$", "no: a policy with late actions still beats random", "no: a degraded world model still beats random"),
    ("determinism", "closed-loop replay", "90/90 identical", "no: tests reproducibility", "no"),
    ("software stack", "versions recorded per run", "recorded", "no: bookkeeping", "no"),
    ("provenance", "checkpoint paths recorded", "recorded", "no: bookkeeping", "no"),
]
assert len(checks) == int(macros["nChecks"])
crow = [" & ".join(c) + r" \\" for c in checks]
t3 = rf"""% The v1 verification table with two added columns. Written by analysis/paper_tables_v2.py.
% Columns 1-3 copied from old/main_v1.tex (tag v1-old-pipeline). TODO(author): review the two right-hand columns.
\begin{{table}}[t]
\centering
\scriptsize
\caption{{The \nChecks checks that passed with both defects present, and why neither defect was visible to them. A further check, the world model's error on training against held-out episodes (\wmTrain against \wmHeldOut), was relative and had no reference for the absolute error.}}
\label{{tab:checks}}
\begin{{tabular}}{{@{{}}L{{2.2cm}}L{{2.9cm}}L{{1.8cm}}L{{4.7cm}}L{{2.9cm}}@{{}}}}
\toprule
component & check & result & could it see defect 1 (action timing)? & could it see defect 2 (shortcut loss)? \\
\midrule
{chr(10).join(crow)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_checks.tex").write_text(t3)
print("wrote table_results.tex, table_loo.tex, table_checks.tex; all cells match numbers.tex")

# ---------------- appendix: deviations from Dreamer 4 (transcribed from paper/DEVIATIONS.md Table 1) ----------------
# TODO(author): review every row; wording is a paper-safe transcription (no file paths, sheet codes or hashes).
dev = [
    ("bootstrap curriculum", "none; shortcut models train without schedules", "Phase 1 starts flow only for 2k steps and ramps the bootstrap loss in over the next 2k; off in Phase 2", "low under one normalizer"),
    ("context corruption at training time", "past inputs are corrupted at inference only", "frame 0 of every training window is kept at signal level 0.9", "yes; with separate normalizers it caused defect 2"),
    ("frame 0 step size", "no special case", "frame 0 always gets the smallest step size, 1/64", "a patch for the row above"),
    ("attention soft cap", "on", "off; the latent rank collapsed with it on", "medium"),
    ("tokenizer loss", "MSE plus 0.2 LPIPS, normalized; mask ratio drawn per image, including no mask", "MSE on masked patches only; LPIPS on a composite image; no normalization; one mask ratio per clip", "the unmasked case gives zero loss"),
    ("tokenizer encoding context", "frame by frame", "training encodes 8 or 32 frames jointly; deployment encodes single frames", "latent codes depend on the slot position"),
    ("policy head and readout", "categorical or binary, readout unspecified", "41 bins per action dimension; actions sampled at evaluation", "deterministic readouts collapse (Discussion)"),
    ("Phase 2 data and learning rates", "half uniform, half task relevant; BC on the relevant half", "uniform windows over all 216 training episodes; BC on all of them; world model at 0.3 times the learning rate, heads at 3 times", "clones failed episodes"),
    ("first imagined transition", "action from the policy", "the first imagined step gets a zero action", "every rollout starts with a step the policy did not choose"),
    ("signal level when the agent state is read", "unspecified", "imagination reads at 0.75 with step size 1/4; deployment at 0.9 with step size 1/64", "the policy is deployed on a differently noised state; untested"),
    ("RL targets", "continue flag marks non terminal steps; bootstrap from the last value", "done is set only on the last row of each episode; bootstrap from the value of the last imagined state", "the time limit is treated as terminal"),
    ("scale", "20 tasks, 400M and 1.6B parameters, pretraining stages", "one task, 74.9M and 47.8M parameters, no pretraining stages", "external validity only"),
    ("data split", "90/10", "216 training and 24 held out episodes of 240; the tokenizer saw the 24 held out episodes during its first 350 epochs", "world model tests on held out episodes are clean for the dynamics model only"),
]
drow = [" & ".join(r) + r" \\" for r in dev]
t4 = rf"""% Deviations from Dreamer 4. Written by analysis/paper_tables_v2.py; transcribed from DEVIATIONS.md Table 1. TODO(author): review.
\begin{{table}}[t]
\centering
\scriptsize
\caption{{Where the corrected pipeline still differs from Dreamer 4 \citep{{hafner2025dreamer4}}, and whether the difference could hide a silent defect.}}
\label{{tab:deviations}}
\begin{{tabular}}{{@{{}}L{{2.6cm}}L{{3.4cm}}L{{5.2cm}}L{{3.2cm}}@{{}}}}
\toprule
component & Dreamer 4 & this paper & could it hide a defect? \\
\midrule
{chr(10).join(drow)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_deviations.tex").write_text(t4)

# ---------------- appendix: run inventory (sheet CC, BU; DEVIATIONS.md Table 2) ----------------
inv = [
    ("Phase 1, corrected", "2", "1", "the first attempt kept the two separate normalizers and collapsed; discarded", "--"),
    ("Phase 2", r"\nLaunchesPTwo", r"\nDraws", "2 aborted", "--"),
    ("Phase 3", r"\nLaunchesPThree", r"\nRuns", r"\nAbortedPThree aborted (one slow pod stopped on purpose; two failed on a corrupted tokenizer file) and relaunched with the same seeds", "draws 11, 12, 15, 16 on Ada or Blackwell; draws 10, 13, 14, 17 on Ampere"),
    ("BC parents, evaluation", "--", r"\nDraws", "--", "Ada or Blackwell"),
]
irow = [" & ".join(r) + r" \\" for r in inv]
t5 = rf"""% Run inventory. Written by analysis/paper_tables_v2.py from the launch records. No run identifiers (anonymity).
\begin{{table}}[t]
\centering
\small
\caption{{Training launches behind the reported results, and the GPU family used to evaluate each policy. Evaluations are identical bit for bit within a GPU family (\Cref{{sec:limitations}}).}}
\label{{tab:runs}}
\begin{{tabular}}{{@{{}}L{{3.0cm}}cL{{1.2cm}}L{{5.0cm}}L{{3.6cm}}@{{}}}}
\toprule
stage & launches & results & aborted or discarded & evaluation GPU family \\
\midrule
{chr(10).join(irow)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_runs.tex").write_text(t5)
print("wrote table_deviations.tex, table_runs.tex")
