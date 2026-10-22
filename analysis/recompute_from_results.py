"""Recompute the paper's main numbers from the files in this repository and compare them with the paper.

Reads only evaluation/*/episodes.csv, evaluation/*/summary.json and results_corrected/*. Prints one line per
number (recomputed value, the paper's value, OK or MISMATCH) and exits non zero on any mismatch.
usage: python analysis/recompute_from_results.py
"""
import csv, json, math, sys, statistics as st
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; E = ROOT / "evaluation"; R = ROOT / "results_corrected"
EXPECTED = {'gainMeanExact': '27.15', 'gainLoExact': '22.20', 'gainHiExact': '32.10', 'nRunsPositive': '24', 'nRuns': '24', 'bcCatchExact': '54.55', 'imagCatchExact': '81.70', 'gainRunMin': '13.0', 'drawMeanMin': '15.2', 'drawMeanMax': '33.5', 'sdBetween': '5.70', 'sdWithin': '2.76', 'looWithoutLow': '3.33', 'looLowDraw': '10', 'oldAnchorGain': '5.9', 'oldAnchorLo': '1.5', 'oldAnchorHi': '10.4', 'oldAnchorRuns': '6', 'oldRetrainWorst': '-32.6', 'oldRetrainBest': '24.2', 'oldRetrainThird': '-8.3', 'oldBcCatchMin': '35.6', 'oldBcCatchMax': '44.8', 'randomFloor': '9.4', 'bcArgmax': '11.6', 'bcMeanReadout': '11.3', 'imagArgmax': '26.5', 'imagMeanReadout': '31.6', 'imagSampledSeedOne': '84.2', 'mirageNew': '0', 'missedNew': '437', 'mirageOld': '91', 'missedOld': '173', 'mirageOldHealthy': '3', 'missedOldHealthy': '530', 'mirageOldParent': '0', 'missedOldParent': '54', 'maxBeliefMissedOld': '978', 'rewCorrNewMin': '0.972', 'rewCorrNewMax': '0.985', 'rewCorrOldMin': '0.61', 'rewCorrOldMax': '0.71', 'rewCorrOldParent': '0.98', 'errOldSmallStep': '0.0069', 'errOldShortcutRatioMin': '76', 'errOldShortcutRatioMax': '109', 'errNewMin': '0.0058', 'errNewMax': '0.0215', 'errOldNoTopMin': '13', 'errOldNoTopMax': '24', 'timingNewDrawsAmin': '64', 'timingNewDrawsAmax': '66', 'timingNewDrawsZmin': '10.6', 'timingOldReleasedA': '-1.8', 'timingOldReleasedB': '-20.7'}
out, bad = [], []
def rep(name, value, nd=None):
    exp = EXPECTED[name]; shown = f"{value:.{nd}f}" if nd is not None else str(value)
    ok = shown == exp or (nd is not None and abs(float(shown) - float(exp)) < 10 ** (-nd) * 0.51)
    out.append(f"{name:24s} recomputed {shown:>10s}   paper {exp:>10s}   {'OK' if ok else 'MISMATCH'}")
    if not ok: bad.append(name)
def eps(path, policy):
    return {int(r["seed"]): int(r["caught"]) for r in csv.DictReader(open(path)) if r["policy"] == policy}
def rate(d): return 100 * sum(d.values()) / len(d)

# ---- Table 1 and the headline, from the per game files
draws, seeds = range(10, 18), (1, 2, 3)
bc = {d: eps(E / f"tmlr-aligned-bc-seed{d}-n500/episodes.csv", "bc") for d in draws}
p3 = {(d, s): eps(E / f"tmlr-aligned-p3-draw{d}-seed{s}-n500/episodes.csv", "phase3") for d in draws for s in seeds}
assert all(len(v) == 500 for v in bc.values()) and all(len(v) == 500 for v in p3.values())
gain = {(d, s): rate(p3[d, s]) - rate(bc[d]) for d in draws for s in seeds}
runs_csv = {(int(r["draw"]), int(r["seed"])): float(r["delta_pts"]) for r in csv.DictReader(open(R / "p3_runs.csv"))}
assert all(abs(gain[k] - runs_csv[k]) < 1e-6 for k in gain), "episodes.csv and p3_runs.csv disagree"
table = np.array([[gain[d, s] for s in seeds] for d in draws]); dm = table.mean(axis=1)
half = stats.t.ppf(0.975, len(dm) - 1) * dm.std(ddof=1) / math.sqrt(len(dm))
rep("gainMeanExact", dm.mean(), 2); rep("gainLoExact", dm.mean() - half, 2); rep("gainHiExact", dm.mean() + half, 2)
rep("nRunsPositive", int((table > 0).sum())); rep("nRuns", table.size)
rep("bcCatchExact", np.mean([rate(bc[d]) for d in draws]), 2); rep("imagCatchExact", np.mean([rate(p3[k]) for k in p3]), 2)
rep("gainRunMin", table.min(), 1); rep("drawMeanMin", dm.min(), 1); rep("drawMeanMax", dm.max(), 1)
# one way random effects (balanced ANOVA, as in the paper)
a, n = table.shape; gmeans = table.mean(axis=1)
msb = n * ((gmeans - table.mean()) ** 2).sum() / (a - 1); msw = ((table - gmeans[:, None]) ** 2).sum() / (a * (n - 1))
rep("sdBetween", math.sqrt(max(0.0, (msb - msw) / n)), 2); rep("sdWithin", math.sqrt(msw), 2)
low = list(draws)[int(np.argmin(dm))]; rep("looLowDraw", low)
t2 = np.delete(table, int(np.argmin(dm)), axis=0); g2 = t2.mean(axis=1)
msb2 = n * ((g2 - t2.mean()) ** 2).sum() / (t2.shape[0] - 1); msw2 = ((t2 - g2[:, None]) ** 2).sum() / (t2.shape[0] * (n - 1))
rep("looWithoutLow", math.sqrt(max(0.0, (msb2 - msw2) / n)), 2)

# ---- the defective pipeline: the first model's six runs and the three retrains
bc0 = eps(E / "tmlr-n500-categorical/episodes.csv", "bc")
anchor = [eps(E / "tmlr-n500-categorical/episodes.csv", "phase3")]
for s in (1, 2, 3):
    m = eps(E / f"tmlr-cat-seed{s}-n200/episodes.csv", "phase3"); m.update(eps(E / f"tmlr-cat-seed{s}-ext300/episodes.csv", "phase3")); anchor.append(m)
for s in (4, 5): anchor.append(eps(E / f"tmlr-cat-seed{s}-n500/episodes.csv", "phase3"))
assert all(len(m) == 500 for m in anchor)
ag = np.array([rate(m) - rate(bc0) for m in anchor]); h = stats.t.ppf(0.975, len(ag) - 1) * ag.std(ddof=1) / math.sqrt(len(ag))
rep("oldAnchorRuns", len(ag)); rep("oldAnchorGain", ag.mean(), 1); rep("oldAnchorLo", ag.mean() - h, 1); rep("oldAnchorHi", ag.mean() + h, 1)
old_bc = {d: eps(E / f"tmlr-bc-seed{d}-n500/episodes.csv", "bc") for d in (11, 12, 13)}
retr = {d: np.mean([rate(eps(E / f"tmlr-fact-bc{d}-rl{r}-n500/episodes.csv", "phase3")) - rate(old_bc[d]) for r in rs]) for d, rs in ((11, (21, 22)), (12, (23, 24)), (13, (25, 26)))}
rep("oldRetrainWorst", retr[11], 1); rep("oldRetrainBest", retr[12], 1); rep("oldRetrainThird", retr[13], 1)
allold = [rate(bc0)] + [rate(v) for v in old_bc.values()]; rep("oldBcCatchMin", min(allold), 1); rep("oldBcCatchMax", max(allold), 1)
rep("randomFloor", rate(eps(E / "tmlr-n500-categorical/episodes.csv", "random")), 1)

# ---- readout ablation
rt = list(csv.DictReader(open(R / "readout_table.csv")))
med = lambda k: st.median(float(r[k]) for r in rt)
rep("bcArgmax", med("bc_argmax"), 1); rep("bcMeanReadout", med("bc_mean"), 1); rep("imagArgmax", med("p3_argmax"), 1); rep("imagMeanReadout", med("p3_mean"), 1); rep("imagSampledSeedOne", med("p3_sampled"), 1)

# ---- reward calibration: correlations from summary.json, mirage counts from perstep_metrics.csv
cal = lambda p, pol: json.load(open(p))["policies"][pol]["reward_calibration"]["reward_pearson"]
newc = [cal(E / f"tmlr-aligned-p3-draw{d}-seed{s}-n500/summary.json", "phase3") for d in draws for s in seeds]
oldc = [cal(E / f"tmlr-fact-bc11-rl{r}-n500/summary.json", "phase3") for r in (21, 22)]
rep("rewCorrNewMin", min(newc), 3); rep("rewCorrNewMax", max(newc), 3); rep("rewCorrOldMin", min(oldc), 2); rep("rewCorrOldMax", max(oldc), 2)
rep("rewCorrOldParent", cal(E / "tmlr-bc-seed11-n500/summary.json", "bc"), 2)
ps = list(csv.DictReader(open(R / "perstep_metrics.csv")))
grp = {"new": [r for r in ps if r["pipeline"] == "new" and r["policy"] == "phase3"],
       "col": [r for r in ps if r["run"].startswith("old_bc11") and r["policy"] == "phase3"],
       "hea": [r for r in ps if r["pipeline"] == "old" and not r["run"].startswith("old_bc11") and r["policy"] == "phase3"]}
sm = lambda g, k: sum(int(r[k]) for r in grp[g])
rep("mirageNew", sm("new", "mirage")); rep("missedNew", sm("new", "missed")); rep("mirageOld", sm("col", "mirage")); rep("missedOld", sm("col", "missed"))
rep("mirageOldHealthy", sm("hea", "mirage")); rep("missedOldHealthy", sm("hea", "missed"))
par = [r for r in ps if r["run"] == "old_bc11_rl21" and r["policy"] == "bc"][0]; rep("mirageOldParent", int(par["mirage"])); rep("missedOldParent", int(par["missed"]))
rep("maxBeliefMissedOld", max(float(r["max_belief_missed"]) for r in grp["col"]), 0)

# ---- error by step size and signal level (old Phase 1 model at epoch 320 against the corrected one)
def dbucket(tag): return {float(r["d_value"]): float(r["latent_mse"]) for r in csv.DictReader(open(R / f"evaldyn_{tag}/d_bucket_metrics.csv"))}
o, nw = dbucket("OLD_e320"), dbucket("JOINT_e320"); dmin = min(o)
sc = [o[d] for d in o if d != dmin]
rep("errOldSmallStep", o[dmin], 4); rep("errOldShortcutRatioMin", min(sc) / o[dmin], 0); rep("errOldShortcutRatioMax", max(sc) / o[dmin], 0)
rep("errNewMin", min(nw.values()), 4); rep("errNewMax", max(nw.values()), 4)
def notop(tag):
    acc = {}
    for r in csv.DictReader(open(R / f"evaldyn_{tag}/tau_d_joint_metrics.csv")):
        if float(r["tau_end"]) <= 0.8 + 1e-6 and int(r["count"]) > 0:
            d = float(r["d_value"]); s, c = acc.get(d, (0.0, 0)); acc[d] = (s + float(r["latent_mse"]) * int(r["count"]), c + int(r["count"]))
    return {d: s / c for d, (s, c) in acc.items()}
no = notop("OLD_e320"); rat = [no[d] / no[dmin] for d in no if d != dmin and d < 1.0]; assert len(rat) == 5
rep("errOldNoTopMin", min(rat), 0); rep("errOldNoTopMax", max(rat), 0)

# ---- timing test (the summary the paper's timing numbers come from)
ts = list(csv.DictReader(open(R / "timing_summary.csv")))
newA = [r for r in ts if r["model"].startswith("NEW P2") and r["cond"] == "A"]; assert len(newA) == 8
rep("timingNewDrawsAmin", min(float(r["km1_pct"]) for r in newA), 0); rep("timingNewDrawsAmax", max(float(r["km1_pct"]) for r in newA), 0)
rep("timingNewDrawsZmin", min(float(r["km1_z"]) for r in ts if r["model"].startswith("NEW P2")), 1)
oldA = [r for r in ts if "OLD" in r["model"] and r["cond"] == "A"]; oldB = [r for r in ts if "OLD" in r["model"] and r["cond"] == "B"]
rep("timingOldReleasedA", float(oldA[0]["km1_pct"]), 1); rep("timingOldReleasedB", float(oldB[0]["km1_pct"]), 1)

print("\n".join(out)); print(f"\n{len(out) - len(bad)} of {len(out)} numbers match the paper" + (f"; MISMATCH: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
