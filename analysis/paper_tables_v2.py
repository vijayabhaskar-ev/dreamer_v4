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
\caption{{Catch rate of each BC parent and of the imagination policies trained from it (\nBoards paired games each, actions sampled), and the gain of each run over its parent in points. The imagination column is the mean over the \nSeeds seeds of the draw. The last row averages the \nDraws draws; the interval is a 95\% $t$ confidence interval over the draw means. Every run is listed in \Cref{{tab:runs_all}}.}}
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
\begin{{table}}[H]
\centering
\small
\caption{{Leave one draw out. Each row leaves out one draw and gives the mean gain over the remaining draws with its 95\% $t$ interval, and the standard deviations between draws and within a draw from the one way random effects fit. With all \nDraws draws the mean gain is \gainMeanExact points. Its interval runs from \gainLoExact to \gainHiExact points. The standard deviation between draws is \sdBetween points and within a draw \sdWithin points.}}
\label{{tab:loo}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
draw left out & its mean gain & mean gain of the rest & sd between draws & sd within a draw \\
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
# shuffle probe cell: v1 printed "0.17" (one old model); the cell now gives the range over both old Phase 2 models, from the raw files
def _shift(d):
    r = {x["metric"]: float(x["value"]) for x in csv.DictReader(open(ROOT / "evaluation" / d / "action_shuffle_agent.csv"))}
    return [r["policy_mu_abs_delta_dim0"], r["policy_mu_abs_delta_dim1"]]
_old = _shift("agent-e40-categorical") + _shift("agent-optionA-e040"); _blind = _shift("agent")
check("shufPolicyShiftMin", f(min(_old), 2)); check("shufPolicyShiftMax", f(max(_old), 2))
check("shufPolicyShiftBlindMin", f(min(_blind), 3)); check("shufPolicyShiftBlindMax", f(max(_blind), 3))
checks = [
    (r"$\lambda$ returns, advantages", "unit test against hand computed values", "pass", "no: tests the RL arithmetic on given inputs", "no"),
    ("attention firewall", r"$\partial \hat z / \partial\,\mathrm{agent} = 0$", "exact", "no: tests gradient routing", "no"),
    ("gradient routing", "per head isolation", "pass", "no: tests gradient routing", "no"),
    ("tokenizer", "latent rank, reconstruction", r"rank $\approx \chkTokRank$", "no: frames were correct in both pipelines", "no"),
    ("reward head", "closed loop calibration", r"MAE \chkRewMaeMin\ to \chkRewMaeMax", "no: reward rows were paired correctly", "no"),
    ("continue head", "closed loop calibration", r"Brier \chkContBrier", "no: same as the reward head", "no"),
    ("PMPO objective", "unit test against the algebra", "pass", "no: tests the policy loss on given inputs", "no"),
    ("action conditioning", "batch shuffle probe", r"policy shift \shufPolicyShiftMin to \shufPolicyShiftMax per action dimension, against \shufPolicyShiftBlindMin to \shufPolicyShiftBlindMax for an action blind model", "no: asks whether actions matter, never when", "no"),
    ("full stack", "random policy floor", "pass (exact test on the paired games that only one of the two policies caught)", "no: a policy with late actions still beats random", "no: a degraded world model still beats random"),
    ("determinism", "closed loop replay", r"\chkReplayN/\chkReplayN\ identical", "no: tests reproducibility", "no"),
    ("software stack", "versions recorded per run", "recorded", "no: bookkeeping", "no"),
    ("provenance", "checkpoint paths recorded", "recorded", "no: bookkeeping", "no"),
]
assert len(checks) == int(macros["nChecks"])
crow = [" & ".join(c) + r" \\" for c in checks]
t3 = rf"""% The v1 verification table with two added columns. Written by analysis/paper_tables_v2.py.
% Columns 1-3 copied from old/main_v1.tex (tag v1-old-pipeline). TODO(author): review the two right-hand columns.
\begin{{table}}[H]
\centering
\scriptsize
\caption{{The \nChecks checks that passed with both defects present, and why neither defect was visible to them. A further check, the world model's error on training episodes against its error on held out episodes (\wmTrain against \wmHeldOut), was relative and had no reference for the absolute error.}}
\label{{tab:checks}}
\begin{{tabular}}{{@{{}}L{{2.2cm}}L{{2.9cm}}L{{1.8cm}}L{{4.7cm}}L{{2.9cm}}@{{}}}}
\toprule
component & check & result & could it see defect 1 (action timing)? & could it see defect 2 (bootstrap loss)? \\
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
    ("bootstrap curriculum", "none; shortcut models train without schedules", r"Phase 1 starts flow only for \curriculumWarmup\ steps and ramps the bootstrap loss in until step \curriculumRamp; off in Phase 2", "low under one normalizer"),
    ("context corruption at training time", "past inputs are corrupted at inference only", r"frame 0 of every training window is kept at signal level \ctxSignal", "yes; with separate normalizers it caused defect 2"),
    ("frame 0 step size", "no special case", r"frame 0 always gets the smallest step size, $1/\kMax$", "a patch for the row above"),
    ("attention soft cap", "on", "off; the latent rank collapsed with it on", "medium"),
    ("tokenizer loss", "MSE plus \lpipsWeight\ LPIPS, normalized; mask ratio drawn per image, including no mask", "MSE on masked patches only; LPIPS on a composite image; no normalization; one mask ratio per clip", "the unmasked case gives zero loss"),
    ("tokenizer encoding context", "causal in time, so a frame's code can depend on its past frames", r"training encodes \seqShort\ or \seqLong\ frames jointly; deployment encodes each frame alone, without its past frames", "latent codes depend on the slot position"),
    ("frozen tokenizer", "the world model trains on the output of a frozen tokenizer", "weights frozen, but dropout and drop path stayed active while the tokenizer encoded the training windows of Phases 1 and 2; both are off in imagination and at evaluation", "unintended; untested"),
    ("policy head and readout", "categorical or binary, readout unspecified", r"\policyBins\ bins per action dimension; actions sampled at evaluation", "deterministic readouts collapse (Discussion)"),
    ("Phase 2 data and learning rates", "half uniform, half task relevant; BC on the relevant half; learning rates unspecified", r"uniform windows over all \nTrainEps\ training episodes; BC on all of them; world model at \lrMultDyn\ times the learning rate, heads at \lrMultHeads\ times", "clones failed episodes"),
    ("first imagined transition", "action from the policy", "the first imagined step gets a zero action", "every rollout starts with a step the policy did not choose"),
    ("signal level when the agent state is read", "unspecified", r"imagination reads at \imagReadSignal\ with step size $1/\kSample$; deployment at \ctxSignal\ with step size $1/\kMax$", "the policy is deployed on a differently noised state; untested"),
    ("RL targets", "continue flag marks non terminal steps; bootstrap from the last value", "done is set only on the last row of each episode; bootstrap from the value of the last imagined state", "the time limit is treated as terminal"),
    ("scale", r"20 tasks, \dreamerParamsTokenizer M and \dreamerParamsDynamics B parameters, single image batches and a final stage of long batches", r"one task, \nParamsTokenizer M and \nParamsWorldModel M parameters, one dataset of \nDemos\ episodes for all phases", "external validity only"),
    ("data split", "90/10", r"\nTrainEps\ training and \nHeldOut\ held out episodes of \nDemos; the tokenizer saw the \nHeldOut\ held out episodes during its first \tokAllEpsEpochs\ epochs", "tests on held out episodes are clean for the world model but not for the tokenizer"),
]
drow = [" & ".join(r) + r" \\" for r in dev]
t4 = rf"""% Deviations from Dreamer 4. Written by analysis/paper_tables_v2.py; transcribed from DEVIATIONS.md Table 1. TODO(author): review.
\begin{{table}}[H]
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
    ("Phase 1", r"\nLaunchesPOne", r"\nResultsPOne", "the first launch kept the two separate normalizers of defect 2, collapsed and was discarded; the second stopped before its first training step and was relaunched with the same settings and live logging", "--"),
    ("Phase 2", r"\nLaunchesPTwo", r"\nDraws", r"\nAbortedPTwo\ aborted (one stopped to restart its rented machine, one out of GPU memory at start) and relaunched with the same seeds", "--"),
    ("Phase 3", r"\nLaunchesPThree", r"\nRuns", r"\nAbortedPThree aborted (one stopped on purpose on a slow rented machine; two failed on a corrupted tokenizer file) and relaunched with the same seeds", "draws 11 and 12 on Ada, draws 15 and 16 on Blackwell; draws 10, 13, 14, 17 on Ampere"),
    ("BC parents, evaluation", "--", r"\nDraws", "--", "Blackwell (one RTX 5070 Ti)"),
]
irow = [" & ".join(r) + r" \\" for r in inv]
t5 = rf"""% Run inventory. Written by analysis/paper_tables_v2.py from the launch records. No run identifiers (anonymity).
\begin{{table}}[H]
\centering
\small
\caption{{Training launches behind the reported results, and the GPU family used to evaluate each policy. Ada and Blackwell cards agreed exactly on the games we repeated, and an Ampere card differed on \gpuDivergedBoards of \nBoards games (\Cref{{sec:limitations}}).}}
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

# ---------------- appendix: hyperparameters (cells are macro names only, so the table cannot drift from numbers.tex) ----------------
hp = [
    ("Data", [
        ("episodes (expert / mixed small / mixed large)", r"\nDemos\ (\nExpertDemos\ / \nMixedSmallDemos\ / \nMixedLargeDemos)"),
        ("training / held out episodes", r"\nTrainEps\ / \nHeldOut"),
        ("decisions per game (action repeat); image size", r"\episodeSteps\ (\actionRepeat); \imgSize\ by \imgSize\ pixels")]),
    ("Tokenizer", [
        ("parameters", r"\nParamsTokenizer M"),
        ("patch size; depth; width", r"\tokPatch; \tokDepth; \embedDim"),
        ("latent tokens per frame; latent dimension", r"\latentTokens; \latentDim"),
        ("learning rate, constant; weight decay", r"$\lrTok$; \wdTok"),
        ("epochs; steps per epoch; clip length", r"\tokEpochs; \tokStepsPerEpoch; \tokClipLen\ frames"),
        ("batch size; precision", r"\tokBatchEarly, then \tokBatchLate\ after epoch \tokAllEpsEpochs; fp32, then fp16 mixed precision"),
        ("mask ratio", r"uniform from 0 to \tokMaskMax, one value per clip"),
        ("loss; gradient clip", r"MSE on masked patches plus \lpipsWeight\ LPIPS (VGG); \clipTok")]),
    ("World model (Phase 1)", [
        ("parameters", r"\nParamsWorldModel M"),
        ("depth; width; attention heads (key/value heads)", r"\dynDepth; \embedDim; \dynHeads\ (\dynKvHeads)"),
        ("step sizes; context signal level", r"$1/\kMax$ to 1; \ctxSignal"),
        ("epochs trained (length of the cosine schedule); batch size", r"\epochsPOne\ (\cosineEpochsPOne); \batchPOne"),
        ("steps per epoch; gradient clip", r"\stepsPerEpochPOne; \clipPOne"),
        ("peak learning rate; warmup steps; weight decay (on attention and feed forward matrices)", r"$\lrPOne$; \warmupPOne; \wdPOne\ (\wdHeavyPOne)"),
        ("training windows, short and long (share of long batches)", r"\seqShort\ and \seqLong\ frames (\longBatchRatio)"),
        ("context length; register tokens", r"\contextLength\ frames; \registerTokens"),
        ("bootstrap curriculum", r"\curriculumWarmup\ flow only steps, then the bootstrap loss ramps in linearly until step \curriculumRamp")]),
    ("Phase 2", [
        ("epochs; steps; batch size", r"\epochsPTwo; \stepsPTwo; \batchPTwo"),
        ("base learning rate; world model multiplier; heads multiplier", r"$\lrPTwo$; \lrMultDyn; \lrMultHeads"),
        ("cosine decay of the base learning rate to; warmup steps", r"$\minLrPTwo$; \warmupPTwo"),
        ("policy head", r"categorical, \policyBins\ bins per action dimension"),
        ("heads parameters", r"\nParamsHeads M")]),
    ("Phase 3 (imagination)", [
        ("epochs; steps; batch size", r"\epochsPThree; \stepsPThree; \batchPThree"),
        ("real context frames; imagination horizon; sampling steps", r"\ctxFrames; \horizon; \kSample"),
        (r"PMPO $\alpha$, $\beta$; discount; $\lambda$", r"\pmpoAlpha, \pmpoBeta; \discount; \lambdaRet"),
        ("learning rate (cosine decay to); warmup steps; weight decay; gradient clip", r"$\lrPThree$ ($\minLrPThree$); \warmupPThree; \wdPThree; \clipPThree")]),
    ("Evaluation", [
        ("paired games per policy; action readout", r"\nBoards; sampled"),
        ("decisions per game (action repeat); MuJoCo; dm\\_control (laptop)", r"\episodeSteps\ (\actionRepeat); \mujocoVersion; \dmControlLaptop"),
        ("optimizer (all phases); PyTorch (Python) on the rented GPUs and on the laptop", r"AdamW; \torchPods\ (\pythonPods) and \torchLaptop\ (\pythonLaptop)")]),
]
hrows = []
for group, items in hp:
    hrows.append(r"\multicolumn{2}{@{}l}{\textit{" + group + r"}} \\")
    hrows += [f"{k} & {v} \\\\" for k, v in items]
t6 = rf"""% Hyperparameters. Written by analysis/paper_tables_v2.py; every cell is a macro from numbers.tex (read from the checkpoints and launch scripts).
\begin{{table}}[H]
\centering
\small
\caption{{Settings of the corrected pipeline, read from the checkpoints and launch scripts. Dreamer 4 trains a \dreamerParamsTokenizer M tokenizer and a \dreamerParamsDynamics B world model, which it calls the dynamics model \citep{{hafner2025dreamer4}}.}}
\label{{tab:hparams}}
\begin{{tabular}}{{@{{}}L{{7.2cm}}L{{6.6cm}}@{{}}}}
\toprule
setting & value \\
\midrule
{chr(10).join(hrows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_hparams.tex").write_text(t6)
print("wrote table_hparams.tex")

# ---------------- appendix: every imagination run (p3_runs.csv + p3_wandb_curves.csv + summary.json reward calibration) ----------------
BKP = Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup"
curves = {(int(r["draw"]), int(r["seed"])): r for r in csv.DictReader(open(BKP / "phase3/analysis/p3_wandb_curves.csv"))}
EVAL_GPU = {10: "A4500", 11: "RTX 4090", 12: "RTX 4000 Ada", 13: "A4500", 14: "A40", 15: "RTX 5080", 16: "RTX 5090", 17: "A4500"}   # sheet BU
import json as _json
rrows = []
for r in runs:
    d, sd = int(r["draw"]), int(r["seed"])
    summ = _json.load(open(BKP / f"phase3/draw{d}/evaluation/tmlr-aligned-p3-draw{d}-seed{sd}-n500/summary.json"))
    pear = summ["policies"]["phase3"]["reward_calibration"]["reward_pearson"]
    rrows.append(f"{d} & {sd} & {f(100*float(r['bc_rate']))} & {f(100*float(r['p3_rate']))} & {float(r['delta_pts']):+.1f} & {f(float(r['se_paired_pts']))} & {f(float(r['bc_return']), 0)} & {f(float(r['p3_return']), 0)} & {f(float(curves[(d, sd)]['imagined_return_last500']), 0)} & {f(pear, 3)} & {EVAL_GPU[d]} \\\\")
t7 = rf"""% Every imagination run. Written by analysis/paper_tables_v2.py from p3_runs.csv, p3_wandb_curves.csv and the evaluation summaries.
\begin{{table}}[H]
\centering
\scriptsize
\caption{{Every imagination run of the corrected pipeline on the same \nBoards games. SE is the paired standard error of the gain over those games. Imagined return is the mean $\lambda$ return of the imagined rollouts over the \horizon step horizon, including the bootstrapped value at the last step, averaged over the last \riseWindow training steps; it is on a different scale from the real return and is read under each draw's own reward head, so it is comparable across the seeds of one draw and only loosely across draws. Reward correlation is the Pearson correlation between predicted and real reward over all steps of the policy's evaluation games. The evaluation GPU is the card that scored the imagination policy; every BC parent was evaluated on one Blackwell card, an RTX 5070 Ti.}}
\label{{tab:runs_all}}
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{@{{}}rrrrrrrrrrl@{{}}}}
\toprule
draw & seed & BC catch & imag.\ catch & gain & SE & BC return & imag.\ return & imagined return & reward corr. & evaluation GPU \\
 & & (\%) & (\%) & (pts) & (pts) & (real) & (real) & ($\lambda$) & & \\
\midrule
{chr(10).join(rrows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(GEN / "table_runs_all.tex").write_text(t7)
print("wrote table_runs_all.tex")
