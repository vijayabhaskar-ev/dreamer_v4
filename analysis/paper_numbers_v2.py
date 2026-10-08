#!/usr/bin/env python
"""Regenerate every number the v2 (corrected-pipeline) paper cites, from the raw evaluation outputs, as LaTeX macros.

Writes  <paper>/generated/numbers.tex   \\newcommand macros (one per number; \\input by main.tex)
        <paper>/generated/NUMBERS.md    macro -> value -> meaning -> claim -> source, plus a check against the
                                        values written in paper/CLAIMS_LIST.md (EXPECT column)

Inputs  --backup  corrected-pipeline results (default ~/Documents/Projects/dreamer_v4_phase1_aligned_backup):
                  phase3/analysis/{p3_runs.csv,p3_wandb_curves.csv,perstep_metrics.csv}, phase3/draw*/evaluation/*,
                  det_readout_check/readout_table.csv, timing_new_draws/timing_summary.csv, old_draw_forensics/timing_tests.log,
                  eval_held_out/{final_checks.out,timing_control_OLD_e040.log,evaldyn_*}
        this repo evaluation/tmlr-*  (old, defective pipeline: the "before" arm), evaluation/tmlr-aligned-bc-seed*-n500,
                  evaluation/wm-split-check; ball_in_cup_catch.npz if present (demonstration catch rate)

Three numbers come from wandb training logs and are NOT regenerated here (constants block at the end; sheet BK/BO).
Usage:  python -m analysis.paper_numbers_v2 [--backup DIR] [--paper DIR]
"""
import argparse, csv, itertools, json, math, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REG = []  # (macro, value string, meaning, claim, source, expect)


def add(name, value, meaning, claim, source, expect=None):
    assert re.fullmatch(r"[A-Za-z]+", name), f"macro names are letters only: {name}"
    assert name not in {r[0] for r in REG}, f"duplicate macro {name}"
    REG.append((name, value, meaning, claim, source, expect))


def f(x, nd=1):
    """Fixed decimals, ties rounded half-up on the DECIMAL value (27.15 -> 27.2, not the float artefact 27.1).
    A real minus sign comes from math mode (the macro wraps the value in \\ensuremath)."""
    out = str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    return out.lstrip("-") if float(out) == 0 else out


def tci(v, level=0.95):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return m, m - h, m + h, m / se


def one_way_random(table, alpha=0.05):
    """Balanced one-way random-effects ANOVA (= REML when positive); Williams-Tukey CI for the between component."""
    y = np.asarray(table, float); a, n = y.shape
    gm, gmeans = y.mean(), y.mean(axis=1)
    msb = n * ((gmeans - gm) ** 2).sum() / (a - 1)
    msw = ((y - gmeans[:, None]) ** 2).sum() / (a * (n - 1))
    n1, n2 = a - 1, a * (n - 1)
    sb2 = max(0.0, (msb - msw) / n)
    fu, fl = stats.f.ppf(1 - alpha / 2, n1, n2), stats.f.ppf(alpha / 2, n1, n2)
    cu, cl = stats.chi2.ppf(1 - alpha / 2, n1) / n1, stats.chi2.ppf(alpha / 2, n1) / n1
    lo, hi = max(0.0, (msb - msw * fu) / (n * cu)), max(0.0, (msb - msw * fl) / (n * cl))
    w_lo, w_hi = n2 * msw / stats.chi2.ppf(1 - alpha / 2, n2), n2 * msw / stats.chi2.ppf(alpha / 2, n2)
    return dict(msb=msb, msw=msw, sb=math.sqrt(sb2), sb_lo=math.sqrt(lo), sb_hi=math.sqrt(hi), sw=math.sqrt(msw),
                sw_lo=math.sqrt(w_lo), sw_hi=math.sqrt(w_hi), icc=sb2 / (sb2 + msw))


def load_eps(path, policy):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["policy"] == policy:
                out[int(r["seed"])] = int(r["caught"])
    return out


def parse_timing(path):
    """Blocks printed by analysis/phase1_timing_control.py -> rows (model, cond A/B, err_correct, late %, late z, argmin k)."""
    cond_key = {"training-style noise": "A", "next frame from pure noise": "B"}
    rows, model, cur = [], None, None
    for line in open(path):
        if m := re.match(r"===== (.+?) =====", line.strip()):
            model, cur = m.group(1), None; continue
        for key, c in cond_key.items():
            if line.strip().startswith(key) and (m := re.search(r"error at k=0 \(correct timing\) = ([0-9.]+)", line)):
                cur = dict(model=model, cond=c, err=float(m.group(1))); rows.append(cur)
        if cur is None:
            continue
        if m := re.search(r"STALE \(previous row\)\s+([+-][0-9.]+)% vs correct\s+\(episode-clustered z = ([+-][0-9.]+)\)", line):
            cur["late"], cur["z"] = float(m.group(1)), float(m.group(2))
        if m := re.search(r"shuffled\s+([+-][0-9.]+)% vs correct", line):
            cur["shuf"] = float(m.group(1))
        if m := re.search(r"lowest error at k=([+-]?\d+)", line):
            cur["argmin"] = int(m.group(1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default=str(Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup"))
    ap.add_argument("--paper", default=str(ROOT / "paper"))
    a = ap.parse_args()
    BK, PAPER = Path(a.backup), Path(a.paper)
    P3 = BK / "phase3"

    # ───────────── C1: does imagination beat its BC parent? (8 draws x 3 seeds, 500 paired boards) ─────────────
    runs = list(csv.DictReader(open(P3 / "analysis/p3_runs.csv")))
    draws = sorted({int(r["draw"]) for r in runs}); seeds = sorted({int(r["seed"]) for r in runs})
    tab = np.array([[next(float(r["delta_pts"]) for r in runs if int(r["draw"]) == d and int(r["seed"]) == s) for s in seeds] for d in draws])
    dm = tab.mean(axis=1)
    m, lo, hi, t = tci(dm)
    src = "phase3/analysis/p3_runs.csv"
    add("nDraws", str(len(draws)), "Phase-2 retrains (draws)", "C1", src, "8")
    add("nSeeds", str(len(seeds)), "imagination seeds per draw", "C1", src, "3")
    add("nRuns", str(tab.size), "imagination runs", "C1", src, "24")
    add("nBoards", "500", "evaluation games per policy", "C1", "evaluate_env --num-episodes", "500")
    add("gainMean", f(m), "mean catch-rate gain over the BC parent, points (mean of draw means)", "C1", src, "27.2")
    add("gainLo", f(lo), "95% CI lower (t, df 7)", "C1", src, "22.2")
    add("gainHi", f(hi), "95% CI upper", "C1", src, "32.1")
    add("gainT", f(t, 2), "t statistic, df 7", "C1", src, "12.98")
    add("gainMeanExact", f(m, 2), "the same mean gain to two decimals (use where the two catch rates are also shown)", "C1", src, "27.15")
    add("gainLoExact", f(lo, 2), "95% CI lower, two decimals", "C1", src, "22.20")
    add("gainHiExact", f(hi, 2), "95% CI upper, two decimals", "C1", src, "32.10")
    add("nRunsPositive", str(int((tab > 0).sum())), "runs with a positive gain", "C1", src, "24")
    add("nDrawsPositive", str(int((dm > 0).sum())), "draw means with a positive gain", "C1", src, "8")
    add("gainRunMin", f(tab.min()), "smallest single-run gain, points", "C1", src, "13.0")
    add("gainRunMax", f(tab.max()), "largest single-run gain, points", "C1", src)
    se = np.array([float(r["se_paired_pts"]) for r in runs])
    add("runSEmin", f(se.min()), "per-run paired standard error, smallest (points)", "C1", src, "2.5")
    add("runSEmax", f(se.max()), "per-run paired standard error, largest", "C1", src, "3.0")
    bc_rate = np.array([next(float(r["bc_rate"]) for r in runs if int(r["draw"]) == d) for d in draws])
    bc_ret = np.array([next(float(r["bc_return"]) for r in runs if int(r["draw"]) == d) for d in draws])
    p3_rate = np.array([float(r["p3_rate"]) for r in runs]); p3_ret = np.array([float(r["p3_return"]) for r in runs])
    add("bcCatch", f(100 * bc_rate.mean()), "BC parents' mean catch rate, % (8 draws); exactly 54.55, a rounding tie", "C1", src, "54.6")
    add("bcCatchExact", f(100 * bc_rate.mean(), 2), "BC parents' mean catch rate to two decimals", "C1", src, "54.55")
    add("imagCatchExact", f(100 * p3_rate.mean(), 2), "imagination policies' mean catch rate to two decimals", "C1", src, "81.70")
    add("bcCatchMin", f(100 * bc_rate.min()), "lowest BC parent catch rate, %", "C1", src)
    add("bcCatchMax", f(100 * bc_rate.max()), "highest BC parent catch rate, %", "C1", src)
    add("imagCatch", f(100 * p3_rate.mean()), "imagination policies' mean catch rate, % (24 runs)", "C1", src, "81.7")
    add("imagCatchMin", f(100 * p3_rate.min()), "lowest imagination-run catch rate, %", "C1", src, "68.8")
    add("imagCatchMax", f(100 * p3_rate.max()), "highest imagination-run catch rate, %", "C1", src, "90.6")
    add("bcReturn", f(bc_ret.mean(), 0), "BC parents' mean return", "C1", src, "292")
    add("imagReturn", f(p3_ret.mean(), 0), "imagination policies' mean return", "C1", src, "521")

    # ───────────── C2: how much does the retrain matter? ─────────────
    v = one_way_random(tab)
    add("sdBetween", f(v["sb"], 2), "between-draw sd of the gain, points", "C2", src, "5.70")
    add("sdBetweenLo", f(v["sb_lo"], 2), "Williams-Tukey 95% lower", "C2", src, "3.43")
    add("sdBetweenHi", f(v["sb_hi"], 2), "Williams-Tukey 95% upper", "C2", src, "11.95")
    add("sdWithin", f(v["sw"], 2), "within-draw (seed-to-seed) sd, points", "C2", src, "2.76")
    add("sdWithinLo", f(v["sw_lo"], 2), "95% lower", "C2", src, "2.05")
    add("sdWithinHi", f(v["sw_hi"], 2), "95% upper", "C2", src, "4.19")
    add("icc", f(v["icc"], 2), "intra-class correlation (raw; keep out of the abstract)", "C2", src, "0.81")
    add("drawMeanMin", f(dm.min()), "smallest draw-mean gain, points", "C2", src, "15.2")
    add("drawMeanMax", f(dm.max()), "largest draw-mean gain, points", "C2", src, "33.5")
    loo = {d: one_way_random(np.delete(tab, i, axis=0)) for i, d in enumerate(draws)}
    low_d = draws[int(np.argmin(dm))]
    others = [loo[d]["sb"] for d in draws if d != low_d]
    add("looLowDraw", str(low_d), "the draw with the smallest mean gain", "C2", src, "10")
    add("looWithoutLow", f(loo[low_d]["sb"], 2), "between-draw sd with that draw left out", "C2", src, "3.33")
    add("looOtherMin", f(min(others), 2), "between-draw sd leaving out any other draw, smallest", "C2", src, "5.50")
    add("looOtherMax", f(max(others), 2), "... largest", "C2", src, "6.17")
    binom = np.mean(p3_rate * (1 - p3_rate) / 500) * 1e4
    add("binomShare", f(100 * binom / v["msw"], 0), "share of within-draw variance that is binomial evaluation noise, %", "C2", src, "39")

    # ───────────── C3: stop rule (epoch 15 minus epoch 10, seed 1 of each draw) ─────────────
    def catch(d, tag):
        s = json.load(open(P3 / f"draw{d}/evaluation/tmlr-aligned-p3-draw{d}-seed1{tag}-n500/summary.json"))
        return 100 * s["policies"]["phase3"]["caught_rate"]
    diffs = [catch(d, "") - catch(d, "-e10") for d in draws]
    m3, lo3, hi3, _ = tci(diffs)
    src3 = "phase3/draw*/evaluation/*-seed1[-e10]-n500/summary.json"
    add("stopDiff", f(m3, 2), "catch rate at epoch 15 minus epoch 10, points (8 draws, seed 1)", "C3", src3, "-0.15")
    add("stopLo", f(lo3, 2), "95% CI lower", "C3", src3, "-1.94")
    add("stopHi", f(hi3, 2), "95% CI upper", "C3", src3, "1.64")
    cur = list(csv.DictReader(open(P3 / "analysis/p3_wandb_curves.csv")))
    last = np.array([float(r["imagined_return_last500"]) for r in cur]); prev = np.array([float(r["imagined_return_prev500"]) for r in cur])
    add("imagRisePct", f(100 * (last.mean() / prev.mean() - 1)), "rise of imagined return over the last 500 steps, %", "C3", "phase3/analysis/p3_wandb_curves.csv", "3.0")
    add("imagRiseRuns", str(int((last > prev).sum())), "runs whose imagined return was still rising", "C3", "phase3/analysis/p3_wandb_curves.csv", "23")

    # ───────────── C4: the defective pipeline (before arm) — this repo's evaluation/tmlr-* ─────────────
    E = ROOT / "evaluation"
    bc0 = load_eps(E / "tmlr-n500-categorical/episodes.csv", "bc")
    old = {"original": load_eps(E / "tmlr-n500-categorical/episodes.csv", "phase3")}
    for s in (1, 2, 3):
        mm = load_eps(E / f"tmlr-cat-seed{s}-n200/episodes.csv", "phase3"); mm.update(load_eps(E / f"tmlr-cat-seed{s}-ext300/episodes.csv", "phase3")); old[f"seed{s}"] = mm
    for s in (4, 5):
        old[f"seed{s}"] = load_eps(E / f"tmlr-cat-seed{s}-n500/episodes.csv", "phase3")
    boards = sorted(bc0); assert len(boards) == 500 and all(len(r) == 500 for r in old.values())
    bc0_rate = np.mean([bc0[b] for b in boards])
    dc = [100 * (np.mean([r[b] for b in boards]) - bc0_rate) for r in old.values()]
    m4, lo4, hi4, _ = tci(dc)
    src4 = "evaluation/tmlr-n500-categorical, tmlr-cat-seed*"
    add("oldAnchorGain", f(m4), "OLD pipeline, anchor draw: mean gain over its BC parent, points (6 runs)", "C4", src4, "5.9")
    add("oldAnchorLo", f(lo4), "95% CI lower (t, df 5)", "C4", src4, "1.5")
    add("oldAnchorHi", f(hi4), "95% CI upper", "C4", src4, "10.4")
    add("oldAnchorRuns", str(len(dc)), "anchor imagination runs", "C4", src4, "6")
    add("oldAnchorPositive", str(sum(x > 0 for x in dc)), "anchor runs with a positive gain", "C4", src4, "6")
    old_bc = {"anchor": bc0_rate}; fact = {}
    for d, rls in ((11, (21, 22)), (12, (23, 24)), (13, (25, 26))):
        b = load_eps(E / f"tmlr-bc-seed{d}-n500/episodes.csv", "bc"); br = np.mean(list(b.values())); old_bc[d] = br
        fact[d] = np.mean([100 * (np.mean(list(load_eps(E / f"tmlr-fact-bc{d}-rl{r}-n500/episodes.csv", "phase3").values())) - br) for r in rls])
    src4b = "evaluation/tmlr-bc-seed*, tmlr-fact-*"
    add("oldRetrainWorst", f(fact[11]), "OLD pipeline: mean gain on the collapsing retrain (seed 11), points", "C4", src4b, "-32.6")
    add("oldRetrainBest", f(fact[12]), "OLD pipeline: mean gain on retrain seed 12", "C4", src4b, "24.2")
    add("oldRetrainThird", f(fact[13]), "OLD pipeline: mean gain on retrain seed 13", "C4", src4b, "-8.3")
    add("oldBcCatchMin", f(100 * min(old_bc.values())), "OLD BC parents: lowest catch rate, %", "C4", src4b, "35.6")
    add("oldBcCatchMax", f(100 * max(old_bc.values())), "OLD BC parents: highest catch rate, %", "C4", src4b, "44.8")
    rnd = load_eps(E / "tmlr-n500-categorical/episodes.csv", "random")
    add("randomFloor", f(100 * np.mean(list(rnd.values()))), "catch rate of a random policy, %", "C12", src4, "9.4")

    # ───────────── C5: reward model on the policies' own games ─────────────
    def cal(path, pol):
        c = json.load(open(path))["policies"][pol]["reward_calibration"]; return c["reward_mae"], c["reward_pearson"]
    new = [cal(P3 / f"draw{d}/evaluation/tmlr-aligned-p3-draw{d}-seed{s}-n500/summary.json", "phase3") for d in draws for s in seeds]
    par = [cal(E / f"tmlr-aligned-bc-seed{d}-n500/summary.json", "bc") for d in draws]
    oldc = [cal(E / f"tmlr-fact-bc11-rl{r}-n500/summary.json", "phase3") for r in (21, 22)]
    src5 = "summary.json reward_calibration"
    add("rewMaeNewMin", f(min(x[0] for x in new), 3), "reward-model error per step, new imagination runs, smallest", "C5", src5, "0.038")
    add("rewMaeNewMax", f(max(x[0] for x in new), 3), "... largest", "C5", src5, "0.061")
    add("rewCorrNewMin", f(min(x[1] for x in new), 3), "reward-model correlation, new runs, smallest", "C5", src5, "0.972")
    add("rewCorrNewMax", f(max(x[1] for x in new), 3), "... largest", "C5", src5, "0.985")
    add("rewMaeParMin", f(min(x[0] for x in par), 3), "reward-model error, new BC parents, smallest", "C5", src5, "0.029")
    add("rewMaeParMax", f(max(x[0] for x in par), 3), "... largest", "C5", src5, "0.031")
    add("rewMaeOldMin", f(min(x[0] for x in oldc), 2), "reward-model error, OLD collapsing draw, smaller of 2 runs", "C5", src5, "0.17")
    add("rewMaeOldMax", f(max(x[0] for x in oldc), 2), "... larger", "C5", src5, "0.26")
    add("rewCorrOldMin", f(min(x[1] for x in oldc), 2), "reward-model correlation, OLD collapsing draw, smaller", "C5", src5, "0.61")
    add("rewCorrOldMax", f(max(x[1] for x in oldc), 2), "... larger", "C5", src5, "0.71")
    ps = list(csv.DictReader(open(P3 / "analysis/perstep_metrics.csv")))
    grp = {"new": [r for r in ps if r["pipeline"] == "new" and r["policy"] == "phase3"],
           "par": [r for r in ps if r["pipeline"] == "new" and r["policy"] == "bc"],
           "col": [r for r in ps if r["run"].startswith("old_bc11") and r["policy"] == "phase3"],
           "hea": [r for r in ps if r["pipeline"] == "old" and not r["run"].startswith("old_bc11") and r["policy"] == "phase3"]}
    sm = lambda g, k: sum(int(r[k]) for r in grp[g]); ph = lambda g: np.array([float(r["phantom"]) for r in grp[g]])
    src5b = "phase3/analysis/perstep_metrics.csv (boards 0-99)"
    add("mirageNew", str(sm("new", "mirage")), "mirage games (never caught, believed > 100), new runs", "C5", src5b, "0")
    add("missedNew", str(sm("new", "missed")), "never-caught games, new runs", "C5", src5b, "437")
    add("mirageOld", str(sm("col", "mirage")), "mirage games, OLD collapsing draw", "C5", src5b, "91")
    add("missedOld", str(sm("col", "missed")), "never-caught games, OLD collapsing draw", "C5", src5b, "173")
    add("mirageOldHealthy", str(sm("hea", "mirage")), "mirage games, OLD healthy runs", "C5", src5b, "3")
    add("missedOldHealthy", str(sm("hea", "missed")), "never-caught games, OLD healthy runs", "C5", src5b, "530")
    add("maxBeliefMissedNew", f(max(float(r["max_belief_missed"]) for r in grp["new"]), 0), "highest predicted return on a never-caught game, new runs", "C5", src5b, "46")
    add("maxBeliefMissedOld", f(max(float(r["max_belief_missed"]) for r in grp["col"]), 0), "... OLD collapsing draw", "C5", src5b)
    add("phantomChildMed", f(np.median(ph("new")), 3), "phantom reward per zero-reward step, new children, median", "C5", src5b, "0.030")
    add("phantomParentMed", f(np.median(ph("par")), 3), "... new BC parents, median", "C5", src5b, "0.011")
    add("phantomOldMin", f(ph("col").min(), 3), "... OLD collapsing draw, smaller", "C5", src5b, "0.196")
    add("phantomOldMax", f(ph("col").max(), 3), "... larger", "C5", src5b, "0.291")

    # ───────────── C6: timing-preference test (late actions vs correct actions, % change in prediction error) ─────────────
    tn = list(csv.DictReader(open(BK / "timing_new_draws/timing_summary.csv")))
    nd = [r for r in tn if r["model"].startswith("NEW")]; oe = [r for r in tn if r["model"].startswith("OLD")]
    rng_ = lambda rows, c, k: [float(r[k]) for r in rows if r["cond"] == c]
    srcT = "timing_new_draws/timing_summary.csv"
    add("timingNewDrawsAmin", f(min(rng_(nd, "A", "km1_pct")), 0), "8 new Phase-2 draws, training-style setup: late-action penalty, smallest, %", "C6", srcT, "64")
    add("timingNewDrawsAmax", f(max(rng_(nd, "A", "km1_pct")), 0), "... largest", "C6", srcT, "66")
    add("timingNewDrawsBmin", f(min(rng_(nd, "B", "km1_pct")), 0), "8 new draws, pure-noise setup: late-action penalty, smallest, %", "C6", srcT, "177")
    add("timingNewDrawsBmax", f(max(rng_(nd, "B", "km1_pct")), 0), "... largest", "C6", srcT, "181")
    add("timingNewDrawsCorrect", str(sum(int(r["argmin_k"]) == 0 for r in nd)), "new-draw setups (of 16) with lowest error at correct timing", "C6", srcT, "16")
    add("timingOldReleasedA", f(rng_(oe, "A", "km1_pct")[0]), "OLD released Phase-1 (epoch 320), training-style: change with late actions, %", "C6", srcT, "-1.8")
    add("timingOldReleasedB", f(rng_(oe, "B", "km1_pct")[0]), "... pure-noise setup, %", "C6", srcT, "-20.7")
    p1 = parse_timing(BK / "eval_held_out/final_checks.out")
    g = lambda rows, c: next(r for r in rows if r["cond"] == c)
    add("timingNewPoneA", f(g(p1, "A")["late"], 0), "corrected Phase-1 (epoch 320), training-style: late-action penalty, %", "C6", "eval_held_out/final_checks.out", "53")
    add("timingNewPoneB", f(g(p1, "B")["late"], 0), "... pure-noise setup, %", "C6", "eval_held_out/final_checks.out", "121")
    od = parse_timing(BK / "old_draw_forensics/timing_tests.log")
    late = [r["late"] for r in od]
    add("timingOldDrawsMin", f(min(late)), "4 OLD Phase-2 draws, both setups: change with late actions, most negative, %", "C6", "old_draw_forensics/timing_tests.log", "-3.3")
    add("timingOldDrawsMax", f(max(late)), "... least negative, %", "C6", "old_draw_forensics/timing_tests.log", "-1.0")
    add("timingOldDrawsLate", str(sum(r["argmin"] == -1 for r in od)), "old-draw setups (of 8) with lowest error at LATE timing", "C6", "old_draw_forensics/timing_tests.log", "8")
    # C8: the same 4 old draws still react strongly to shuffled actions, so a shuffle probe passes on models that read actions late
    shuf = lambda c: [r["shuf"] for r in od if r["cond"] == c]
    add("shuffleOldDrawsAmin", f(min(shuf("A")), 0), "4 OLD Phase-2 draws, training-style: error increase with shuffled vs correctly timed actions, smallest, %", "C8", "old_draw_forensics/timing_tests.log", "22")
    add("shuffleOldDrawsAmax", f(max(shuf("A")), 0), "... largest, %", "C8", "old_draw_forensics/timing_tests.log", "52")
    add("shuffleOldDrawsBmin", f(min(shuf("B")), 0), "4 OLD Phase-2 draws, pure-noise setup: error increase with shuffled actions, smallest, %", "C8", "old_draw_forensics/timing_tests.log", "52")
    add("shuffleOldDrawsBmax", f(max(shuf("B")), 0), "... largest, %", "C8", "old_draw_forensics/timing_tests.log", "92")
    e40 = parse_timing(BK / "eval_held_out/timing_control_OLD_e040.log")
    add("timingOldEarlyA", f(g(e40, "A")["late"]), "OLD Phase-1 epoch 40, training-style: change with late actions, %", "C6", "eval_held_out/timing_control_OLD_e040.log", "-11.5")
    add("nHeldOut", "24", "held-out episodes behind every world-model test", "C6", "OfflineDataset split=val", "24")

    # ───────────── C7: prediction error by step size; paired rollout ─────────────
    def dbucket(tag):
        return {float(r["d_value"]): float(r["latent_mse"]) for r in csv.DictReader(open(BK / f"eval_held_out/evaldyn_{tag}/d_bucket_metrics.csv"))}
    o320, o40, n320 = dbucket("OLD_e320"), dbucket("OLD_e040"), dbucket("JOINT_e320")
    dmin = min(o320); sc = [o320[d] for d in o320 if d != dmin]
    srcD = "eval_held_out/evaldyn_*/d_bucket_metrics.csv"
    add("errOldSmallStep", f(o320[dmin], 4), "OLD released model: 1-step prediction error at the smallest step (1/64)", "C7", srcD, "0.0069")
    add("errOldShortcutMin", f(min(sc), 2), "OLD released model: error at the six shortcut steps, smallest", "C7", srcD, "0.52")
    add("errOldShortcutMax", f(max(sc), 2), "... largest", "C7", srcD, "0.75")
    add("errOldShortcutRatioMin", f(min(sc) / o320[dmin], 0), "shortcut-step error as a multiple of the smallest-step error, smallest", "C7", srcD, "76")
    add("errOldShortcutRatioMax", f(max(sc) / o320[dmin], 0), "... largest", "C7", srcD, "109")
    add("errNewMin", f(min(n320.values()), 4), "corrected model: error over all seven step sizes, smallest", "C7", srcD, "0.0058")
    add("errNewMax", f(max(n320.values()), 4), "... largest", "C7", srcD, "0.0215")
    summ = lambda tag: json.load(open(BK / f"eval_held_out/evaldyn_{tag}/summary.json"))
    add("errOldReleased", f(summ("OLD_e320")["overall_latent_mse"], 4), "OLD released (epoch 320): held-out 1-step error, all step sizes", "C7", "evaldyn_OLD_e320/summary.json", "0.5261")
    add("errOldEarly", f(summ("OLD_e040")["overall_latent_mse"], 4), "OLD epoch 40: held-out 1-step error", "C7", "evaldyn_OLD_e040/summary.json", "0.0143")
    add("errNew", f(summ("JOINT_e320")["overall_latent_mse"], 4), "corrected (epoch 320): held-out 1-step error", "C7", "evaldyn_JOINT_e320/summary.json", "0.0096")
    # C9: error by step size x signal level at tau 0.6-0.8. Frame 0 is pinned at tau = 0.9 (evaluate_dynamics.py:489),
    # so these cells hold no frame-0 entries; the tau 0.8-1.0 cells for d >= 1/4 hold ONLY frame 0.
    def tdcell(tag):
        rows = csv.DictReader(open(BK / f"eval_held_out/evaldyn_{tag}/tau_d_joint_metrics.csv"))
        return {float(r["d_value"]): float(r["latent_mse"]) for r in rows if abs(float(r["tau_start"]) - 0.6) < 1e-6 and int(r["count"]) > 0}
    to_, tn_ = tdcell("OLD_e320"), tdcell("JOINT_e320")
    dsm = min(to_); srcTD = "eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv"
    add("tdOldShortMin", f(min(v for d, v in to_.items() if d != dsm), 2), "OLD released: error at signal level 0.6-0.8 (no frame-0 entries), shortcut steps, smallest", "C9", srcTD, "0.49")
    add("tdOldShortMax", f(max(v for d, v in to_.items() if d != dsm), 2), "... largest", "C9", srcTD, "0.57")
    add("tdOldSmall", f(to_[dsm], 4), "... at the smallest step 1/64", "C9", srcTD, "0.0037")
    add("tdNewMin", f(min(tn_.values()), 4), "corrected: error at signal level 0.6-0.8, all step sizes present, smallest", "C9", srcTD, "0.0039")
    add("tdNewMax", f(max(tn_.values()), 4), "... largest", "C9", srcTD, "0.0049")
    add("errOldOverNew", f(summ("OLD_e320")["overall_latent_mse"] / summ("JOINT_e320")["overall_latent_mse"], 0), "OLD released error as a multiple of the corrected error (same evaluator, same held-out episodes): the absolute anchor the old train-vs-held-out check lacked", "C8", "evaldyn_OLD_e320 + evaldyn_JOINT_e320 summary.json", "55")
    add("rollOldReleased", f(summ("OLD_e320")["rollout_overall_mse"], 4), "OLD released: 4-step rollout error", "C7", "evaldyn_OLD_e320/summary.json", "0.5678")
    txt = open(BK / "eval_held_out/final_checks.out").read()
    add("rollPairedOld", re.search(r"OLD  e040:.*average ([0-9.]+)", txt).group(1), "paired 4-step rollout error, OLD epoch 40 (1,280 shared windows)", "C7", "eval_held_out/final_checks.out", "0.0303")
    add("rollPairedNew", re.search(r"NEW  e320:.*average ([0-9.]+)", txt).group(1), "paired 4-step rollout error, corrected epoch 320", "C7", "eval_held_out/final_checks.out", "0.0189")
    mp = re.search(r"NEW e320 minus OLD e040: mean difference [-0-9.]+ \(([-0-9.]+)%\).*z = ([-0-9.]+)", txt)
    add("rollPairedPct", f(float(mp.group(1))), "paired rollout change, corrected vs OLD epoch 40, %", "C7", "eval_held_out/final_checks.out", "-37.6")
    add("rollPairedZ", f(float(mp.group(2))), "its episode-clustered z", "C7", "eval_held_out/final_checks.out", "-12.0")

    # ───────────── C8: why the checks passed ─────────────
    sh = list(csv.DictReader(open(BK / "eval_held_out/evaldyn_OLD_e040/action_shuffle_sensitivity.csv")))
    tr_, sf = np.mean([float(r["mse_true_actions"]) for r in sh]), np.mean([float(r["mse_shuffled_actions"]) for r in sh])
    add("shuffleOld", f(100 * (sf / tr_ - 1), 0), "OLD epoch 40: error increase with shuffled actions, % (the shuffle probe passes)", "C8", "evaldyn_OLD_e040/action_shuffle_sensitivity.csv", "166")
    wt = json.load(open(E / "wm-split-check/train/summary.json"))["overall_latent_mse"]; wv = json.load(open(E / "wm-split-check/val/summary.json"))["overall_latent_mse"]
    add("wmTrain", f(wt, 3), "OLD released model: world-model error on training episodes", "C8", "evaluation/wm-split-check", "0.531")
    add("wmHeldOut", f(wv, 3), "... on held-out episodes (the 'no gap' check)", "C8", "evaluation/wm-split-check", "0.528")
    add("wmRatio", f(wv / summ("OLD_e040")["overall_latent_mse"], 0), "held-out error as a multiple of the same run's epoch-40 error (latent MSE)", "C8", "derived", "37")
    add("nChecks", "12", "checks in the v1 verification matrix", "C8", "old/main_v1.tex Table 1", "12")

    # ───────────── C10: imagined return vs real catch (exploratory) ─────────────
    imag = {(int(r["draw"]), int(r["seed"])): float(r["imagined_return_last500"]) for r in cur}
    real = {(int(r["draw"]), int(r["seed"])): 100 * float(r["p3_rate"]) for r in runs}
    x = np.array([[imag[(d, s)] for s in seeds] for d in draws]); y = np.array([[real[(d, s)] for s in seeds] for d in draws])
    rho = stats.spearmanr(x.ravel(), y.ravel())[0]
    rng = np.random.default_rng(0); bs = []
    for _ in range(20000):
        di = rng.integers(0, 8, 8); si = rng.integers(0, 3, (8, 3))
        xs = x[di[:, None], si].ravel(); ys = y[di[:, None], si].ravel()
        if np.ptp(xs) > 0 and np.ptp(ys) > 0: bs.append(stats.spearmanr(xs, ys)[0])
    xm, ym = x.mean(1), y.mean(1); rb = stats.spearmanr(xm, ym)[0]
    perm = [stats.spearmanr(xm, np.array(pp))[0] for pp in itertools.permutations(ym)]
    order = np.argsort(-x.ravel())
    src10 = "p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py)"
    add("rhoAll", f(rho, 2), "Spearman, imagined return vs real catch, 24 runs", "C10", src10, "0.14")
    add("rhoLo", f(np.percentile(bs, 2.5), 2), "draw-then-seed bootstrap 95% lower", "C10", src10, "-0.60")
    add("rhoHi", f(np.percentile(bs, 97.5), 2), "... upper", "C10", src10, "0.88")
    add("rhoDraws", f(rb, 2), "Spearman over the 8 draw means", "C10", src10, "0.26")
    add("rhoDrawsP", f(np.mean(np.abs(perm) >= abs(rb) - 1e-12), 2), "its exact permutation p", "C10", src10, "0.54")
    add("pickTopCatch", f(y.ravel()[order[0]]), "real catch of the run with the highest imagined return, %", "C10", src10, "85.4")

    # ───────────── C12: deterministic readouts ─────────────
    rt = list(csv.DictReader(open(BK / "det_readout_check/readout_table.csv")))
    col = lambda k: np.array([float(r[k]) for r in rt])
    det = np.concatenate([col("bc_argmax"), col("bc_mean"), col("p3_argmax"), col("p3_mean")])
    srcR = "det_readout_check/readout_table.csv"
    add("detMedian", f(np.median(det)), "median deterministic catch over 32 evaluations, %", "C12", srcR, "18.3")
    add("sampledMedian", f(np.median(np.concatenate([col("bc_sampled"), col("p3_sampled")]))), "median sampled catch over the same 16 policies, %", "C12", srcR, "62.7")
    add("bcArgmax", f(np.median(col("bc_argmax"))), "BC parents, argmax readout, median catch %", "C12", srcR, "11.6")
    add("bcMeanReadout", f(np.median(col("bc_mean"))), "BC parents, mean readout, median catch %", "C12", srcR, "11.3")
    add("imagArgmax", f(np.median(col("p3_argmax"))), "imagination children (seed 1), argmax readout, median catch %", "C12", srcR, "26.5")
    add("imagMeanReadout", f(np.median(col("p3_mean"))), "imagination children, mean readout, median catch %", "C12", srcR, "31.6")
    add("imagSampledSeedOne", f(np.median(col("p3_sampled"))), "the same 8 children with sampled actions, median catch %", "C12", srcR, "84.2")
    add("nReadoutEvals", str(det.size), "deterministic evaluations (16 policies x 2 readouts)", "C12", srcR, "32")

    # ───────────── demonstrations ─────────────
    npz = ROOT / "ball_in_cup_catch.npz"
    if npz.exists():
        rew = np.asarray(np.load(npz, mmap_mode="r")["rewards"], float)
        add("demoCatch", f(100 * (rew.sum(axis=1) > 0).mean()), "share of the demonstration episodes that catch, % (unpaired)", "C1", "ball_in_cup_catch.npz rewards", "84.2")
        add("nDemos", str(rew.shape[0]), "demonstration episodes", "C1", "ball_in_cup_catch.npz", "240")
        # C8: why the shuffle probe passes with late actions: consecutive stored actions are correlated (row 0 is padding)
        act = np.asarray(np.load(npz, mmap_mode="r")["actions"], float)[:, 1:]
        lag1 = [np.corrcoef(act[:, :-1, k].ravel(), act[:, 1:, k].ravel())[0, 1] for k in range(act.shape[-1])]
        add("lateCorrMin", f(min(lag1), 2), "correlation of each stored action with the next one in its episode (a one-step-late action vs the correct one), smaller of the two action dimensions", "C8", "ball_in_cup_catch.npz actions, rows 1-500", "0.47")
        add("lateCorrMax", f(max(lag1), 2), "... larger of the two", "C8", "ball_in_cup_catch.npz actions, rows 1-500", "0.49")
        # C9: a model-free tell of the file's convention: nothing 'led to' frame 0, so a 'led to' file has an empty row 0
        row0 = np.asarray(np.load(npz, mmap_mode="r")["actions"][:, 0], float)
        add("nRowZero", str(int((row0 == 0).all(axis=1).sum())), "episodes whose action row 0 is exactly zero in the original file (model-free tell of the 'led to' convention)", "C9", "ball_in_cup_catch.npz actions, row 0", "240")
    else:
        print("WARNING: ball_in_cup_catch.npz not found -> demoCatch / nDemos not generated")

    # ───────────── setup: settings read from the checkpoints, launch scripts and data actually used ─────────────
    import torch
    p2 = torch.load(BK / f"phase2/seed{draws[0]}/checkpoints-phase2-aligned-seed{draws[0]}/dynamics_epoch_040.pt", map_location="cpu", weights_only=False)
    p3 = torch.load(BK / f"phase3/draw{draws[0]}/checkpoints-phase3-aligned-draw{draws[0]}-seed1/final.pt", map_location="cpu", weights_only=False)
    dc, ic = p2["dynamics_cfg"], p3["imagination_cfg"]
    dc = dc if isinstance(dc, dict) else vars(dc); ic = ic if isinstance(ic, dict) else vars(ic)
    run1 = open(BK / "provenance_joint/run_phase1.sh").read()
    arg = lambda txt, k: re.search(rf"--{k}[ =]([^ \\\n]+)", txt).group(1)
    srcC = "checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt)"
    add("policyBins", str(dc["policy_num_bins"]), "bins per action dimension of the categorical policy head", "setup", srcC, "41")
    add("kMax", str(dc["K_max"]), "finest shortcut grid: smallest step size is 1/kMax", "setup", srcC, "64")
    add("kSample", str(ic["K_imagination"]), "denoising steps per generated frame in imagination (step size 1/kSample)", "setup", srcC, "4")
    add("ctxSignal", f(1 - dc["tau_ctx"], 1), "signal level of context frames (1 minus tau_ctx)", "setup", srcC, "0.9")
    add("ctxFrames", str(ic["num_context_frames"]), "real context frames before each imagined rollout", "setup", srcC, "4")
    add("horizon", str(ic["imagination_horizon"]), "imagination horizon, steps", "setup", srcC, "15")
    add("pmpoAlpha", f(ic["pmpo_alpha"], 1), "PMPO weight between positive and negative advantages", "setup", srcC, "0.5")
    add("pmpoBeta", f(ic["pmpo_beta"], 1), "PMPO weight of the reverse KL to the frozen BC prior", "setup", srcC, "0.3")
    add("discount", str(ic["gamma"]), "discount factor", "setup", srcC, "0.997")
    add("lambdaRet", str(ic["lambda_"]), "lambda of the TD(lambda) returns", "setup", srcC, "0.95")
    add("epochsPThree", str(ic["epochs"]), "imagination training epochs", "setup", srcC, "15")
    add("stepsPThree", str(ic["epochs"] * ic["steps_per_epoch"]), "imagination training steps in total", "setup", srcC, "3000")
    add("batchPThree", str(ic["batch_size"]), "imagination batch size", "setup", srcC, "48")
    add("epochsPTwo", str(p2["epoch"]), "Phase 2 finetuning epochs", "setup", srcC, "40")
    add("stepsPTwo", str(p2["global_step"]), "Phase 2 finetuning steps in total", "setup", srcC, "20000")
    add("batchPTwo", arg(open(BK / f"phase2/seed{draws[0]}/provenance/run_phase2.sh").read(), "batch-size"), "Phase 2 batch size", "setup", "phase2/seed*/provenance/run_phase2.sh", "32")
    add("batchPOne", arg(run1, "batch-size"), "Phase 1 batch size", "setup", "provenance_joint/run_phase1.sh", "48")
    add("epochsPOne", "320", "Phase 1 epochs (checkpoint used: epoch 320 of a 600-epoch cosine schedule)", "setup", "CONSTANT: sheet BN / AV (both Phase 1 runs stopped at epoch 320)", "320")
    add("cosineEpochsPOne", arg(run1, "epochs"), "length of the Phase 1 cosine learning rate schedule, epochs", "setup", "provenance_joint/run_phase1.sh", "600")
    z = np.load(ROOT / "ball_in_cup_catch_aligned.npz", mmap_mode="r")
    add("imgSize", str(z["frames"].shape[2]), "image side length, pixels", "setup", "ball_in_cup_catch_aligned.npz frames shape", "128")
    add("episodeSteps", str(z["actions"].shape[1] - 1), "control steps per episode", "setup", "ball_in_cup_catch_aligned.npz actions shape", "500")
    add("nTrainEps", str(z["frames"].shape[0] - 24), "training episodes (the other 24 are held out)", "setup", "OfflineDataset split: val_fraction 0.1, split_seed 0", "216")
    add("nExpertDemos", "20", "expert episodes in the dataset", "setup", "CONSTANT: dataset README / sheet BJ", "20")
    add("nMixedSmallDemos", "20", "episodes in the dataset's mixed-small split", "setup", "CONSTANT: dataset README / sheet BJ", "20")
    add("nMixedLargeDemos", "200", "episodes in the dataset's mixed-large split", "setup", "CONSTANT: dataset README / sheet BJ", "200")
    add("nNoisyDemos", "220", "episodes from noise-injected rollouts", "setup", "CONSTANT: dataset README / sheet BJ", "220")
    add("epochEarly", "10", "the earlier checkpoint compared with the final epoch in the training-length check", "C3", "CONSTANT: checkpoints saved at epochs 5, 10, 15; sheet BR addendum", "10")
    add("stopThreshold", "2", "stopping rule: extend training if epoch 15 beats epoch 10 by more than this many points", "C3", "CONSTANT: sheet BR addendum (stop rule)", "2")

    # ───────────── constants from wandb training logs (NOT regenerated; sheet BK / BO) ─────────────
    add("bootArtifactMedian", "838", "OLD Phase 1: median raw bootstrap loss, steps 4,500-17,000 (wandb)", "C7", "CONSTANT: sheet BK", "838")
    add("bootArtifactEnd", "288{,}600", "OLD Phase 1: raw bootstrap loss at the end of training (wandb)", "C7", "CONSTANT: sheet BO", "288{,}600")
    add("dreamerMouseClasses", "121", "Dreamer 4 (Minecraft): classes of the joint categorical for mouse movement (11 x 11)", "setup", "CONSTANT: Dreamer 4 paper text, 'mouse actions as a categorical with 121 classes'", "121")
    add("bootSignal", "0.0004", "genuine raw bootstrap loss once the artifact is removed (wandb)", "C7", "CONSTANT: sheet BK", "0.0004")

    # ───────────── write ─────────────
    out = PAPER / "generated"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "numbers.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by analysis/paper_numbers_v2.py (code repo) — do not edit by hand; rerun the script.\n")
        fh.write("% Every macro prints a number in math mode and handles the following space (xspace).\n")
        claim = None
        for name, val, meaning, c, source, _ in REG:
            if c != claim:
                fh.write(f"\n% ---- {c} ----\n"); claim = c
            fh.write(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{val}}}\\xspace}}  % {meaning} [{source}]\n")
    bad = [(n, v, e) for n, v, _, _, _, e in REG if e is not None and v != e]
    with open(out / "NUMBERS.md", "w") as fh:
        fh.write("# Number macros for the paper (auto-generated: `python -m analysis.paper_numbers_v2` in the code repo)\n\n")
        fh.write("Type the macro where the number goes, e.g. `a gain of \\gainMean points [\\gainLo, \\gainHi]`. Signs: negative values carry\n")
        fh.write("their own minus; write `+\\gainMean` yourself when you want a plus. `CHECK` compares with the value in CLAIMS_LIST.md.\n\n")
        fh.write("**Rounding tie in the headline.** The exact values are BC 54.55%, imagination 81.70%, gain 27.15 points. To one decimal they\n")
        fh.write("round to 54.6, 81.7 and 27.2, and 81.7 - 54.6 reads as 27.1. Wherever the two catch rates and the gain appear together\n")
        fh.write("(the main results sentence, the main table), use `\\bcCatchExact`, `\\imagCatchExact`, `\\gainMeanExact` [`\\gainLoExact`, `\\gainHiExact`].\n\n")
        fh.write("| macro | value | meaning | claim | source | CHECK |\n|---|---|---|---|---|---|\n")
        for name, val, meaning, c, source, e in REG:
            chk = "" if e is None else ("ok" if val == e else f"**MISMATCH: claims list says {e}**")
            fh.write(f"| `\\{name}` | {val.replace('{,}', ',')} | {meaning} | {c} | {source} | {chk} |\n")
    print(f"wrote {len(REG)} macros -> {out/'numbers.tex'}")
    for n, v, e in bad:
        print(f"  MISMATCH {n}: computed {v}, claims list {e}")
    print(f"{len(bad)} mismatches vs CLAIMS_LIST.md")


if __name__ == "__main__":
    main()
