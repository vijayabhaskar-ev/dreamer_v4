"""Reproduce every statistic in the paper's results section from the raw eval CSVs.

Inputs (all produced by dynamics/evaluate_env.py; ckpt provenance in each dir's
summary.json -> data_facts):
    evaluation/tmlr-n500-categorical/         BC + original phase3, eval seeds 0-499
    evaluation/tmlr-cat-seed{1,2,3}-n200/     seeds 1-3, eval seeds 0-199
    evaluation/tmlr-cat-seed{1,2,3}-ext300/   seeds 1-3, eval seeds 200-499
    evaluation/tmlr-cat-seed{4,5}-n500/       seeds 4-5, eval seeds 0-499

Usage:  python -m analysis.paper_stats
"""
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 20_000
SEED = 0


def load(path, policy):
    out = {}
    with open(ROOT / path) as f:
        for r in csv.DictReader(f):
            if r["policy"] == policy:
                out[int(r["seed"])] = (float(r["return"]), int(r["caught"]))
    return out


def sign_test(b, c):
    """Exact two-sided sign test on discordant counts."""
    m = b + c
    if m == 0:
        return 1.0
    p = sum(math.comb(m, k) for k in range(max(b, c), m + 1)) / 2 ** m * 2
    return min(1.0, p)


def boot_ci(x, rng, n_boot=N_BOOT):
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    bc = load("evaluation/tmlr-n500-categorical/episodes.csv", "bc")
    runs = {"original": load("evaluation/tmlr-n500-categorical/episodes.csv", "phase3")}
    for s in (1, 2, 3):
        m = load(f"evaluation/tmlr-cat-seed{s}-n200/episodes.csv", "phase3")
        m.update(load(f"evaluation/tmlr-cat-seed{s}-ext300/episodes.csv", "phase3"))
        runs[f"seed{s}"] = m
    for s in (4, 5):
        runs[f"seed{s}"] = load(f"evaluation/tmlr-cat-seed{s}-n500/episodes.csv", "phase3")

    boards = sorted(bc.keys())
    n = len(boards)
    assert n == 500 and all(len(r) == n for r in runs.values()), "missing boards"
    bc_c = np.array([bc[s][1] for s in boards], float)
    bc_r = np.array([bc[s][0] for s in boards], float)

    print(f"BC baseline: catch {bc_c.mean():.3f}, return {bc_r.mean():.1f}  (n={n})")
    print("\n== Table: per-run paired comparison vs BC ==")
    d_catch, d_ret, per_run_evalvar = [], [], []
    Cs, Rs = [], []
    for name, d in runs.items():
        c = np.array([d[s][1] for s in boards], float)
        r = np.array([d[s][0] for s in boards], float)
        Cs.append(c); Rs.append(r)
        dc, dr = c - bc_c, r - bc_r
        b = int(((c == 1) & (bc_c == 0)).sum())
        e = int(((c == 0) & (bc_c == 1)).sum())
        lo, hi = boot_ci(dc, rng, 8000)
        print(f"  {name:9s} catch={c.mean():.3f}  d_catch={dc.mean():+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}] sign-p={sign_test(b, e):.3f}   d_return={dr.mean():+.1f}")
        d_catch.append(dc.mean()); d_ret.append(dr.mean())
        per_run_evalvar.append(dc.var(ddof=1) / n)

    k = len(d_catch)
    print("\n== Run-level inference (each training run = one observation) ==")
    for label, v in (("d_catch", d_catch), ("d_return", d_ret)):
        m_, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        se = sd / math.sqrt(k)
        t = m_ / se
        tcrit = 2.571  # t_{0.975, df=5}
        print(f"  {label}: mean {m_:+.4f}  sd {sd:.4f}  t({k-1})={t:.2f}  "
              f"95% CI [{m_ - tcrit * se:+.4f}, {m_ + tcrit * se:+.4f}]  ({sum(x > 0 for x in v)}/{k} positive)")

    print("\n== Committee estimate (per-board mean of runs, paired vs BC, board bootstrap) ==")
    Cbar, Rbar = np.mean(Cs, axis=0), np.mean(Rs, axis=0)
    Dc, Dr = Cbar - bc_c, Rbar - bc_r
    lo, hi = boot_ci(Dc, rng)
    print(f"  d_catch  = {Dc.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    lo, hi = boot_ci(Dr, rng)
    print(f"  d_return = {Dr.mean():+.1f}   95% CI [{lo:+.1f}, {hi:+.1f}]")

    print("\n== Return-gain channel decomposition ==")
    pb, pp = bc_c.mean(), Cbar.mean()
    mu_b, mu_p = bc_r.sum() / bc_c.sum(), Rbar.sum() / Cbar.sum()
    tot = Dr.mean()
    print(f"  rate channel        (dp * mu_p): {(pp - pb) * mu_p:+.1f}  ({(pp - pb) * mu_p / tot:.0%})")
    print(f"  conditional channel (p_b * dmu): {pb * (mu_p - mu_b):+.1f}  ({pb * (mu_p - mu_b) / tot:.0%})")
    print(f"  return-per-success: {mu_b:.0f} -> {mu_p:.0f}")

    print("\n== Training-seed variance decomposition ==")
    obs_sd = float(np.std(d_catch, ddof=1))
    eval_se = math.sqrt(float(np.mean(per_run_evalvar)))
    train_sd = math.sqrt(max(obs_sd ** 2 - eval_se ** 2, 0.0))
    print(f"  across-run sd {obs_sd:.4f} | per-run eval SE {eval_se:.4f} "
          f"| implied training-seed sd {train_sd:.4f}")

    print("\n== Multiplicity ==")
    print(f"  6 paired tests vs one baseline; Bonferroni alpha = {0.05/6:.4f}; "
          f"runs at p<=0.002 remain significant after correction.")

    print("\n== BC vs random (whole-stack integration check) ==")
    rnd = load("evaluation/tmlr-n500-categorical/episodes.csv", "random")
    rn_c = np.array([rnd[s][1] for s in boards], float)
    b = int(((bc_c == 1) & (rn_c == 0)).sum())
    e = int(((bc_c == 0) & (rn_c == 1)).sum())
    print(f"  BC {bc_c.mean():.3f} vs random {rn_c.mean():.3f}; "
          f"paired sign test p = {sign_test(b, e):.2e}")

    print("\n== Readout ablation (run 1, n=500 each, shared eval seeds) ==")
    for r_ in ("mean", "argmax"):
        d = load(f"evaluation/tmlr-readout-{r_}-n500/episodes.csv", "phase3")
        c = np.array([d[s][1] for s in boards], float)
        ret = np.array([d[s][0] for s in boards], float)
        se = math.sqrt(c.mean() * (1 - c.mean()) / n)
        print(f"  {r_:7s} catch {c.mean():.3f} +-{1.96*se:.3f}  return {ret.mean():.1f}"
              f"  (return|catch {ret.sum()/max(c.sum(),1):.0f})")
    r1c = np.array([runs['original'][s][1] for s in boards], float)
    r1r = np.array([runs['original'][s][0] for s in boards], float)
    print(f"  sample  catch {r1c.mean():.3f}         return {r1r.mean():.1f}"
          f"  (return|catch {r1r.sum()/r1c.sum():.0f})")
    print(f"  random floor for comparison: {rn_c.mean():.3f}")

    print("\n== Gaussian pipeline variant (appendix; single run per arm — CONFOUNDED, "
          "no run-level inference) ==")
    g_bc = load("evaluation/tmlr-n500-stoch/episodes.csv", "bc")
    g_p3 = load("evaluation/tmlr-n500-stoch/episodes.csv", "phase3")
    gb = np.array([g_bc[s][1] for s in boards], float)
    gp = np.array([g_p3[s][1] for s in boards], float)
    print(f"  Gaussian pipeline:    BC {gb.mean():.3f}  imagination-RL {gp.mean():.3f}")
    print(f"  categorical pipeline: BC {bc_c.mean():.3f}  imagination-RL {r1c.mean():.3f}")
    print(f"  pipeline-level gaps (BC arms {bc_c.mean()-gb.mean():+.3f}, "
          f"RL arms {r1c.mean()-gp.mean():+.3f}) — one Phase-2 finetune per arm; "
          f"head type confounded with Phase-2 training variance.")

    print("\n== Closed-loop calibration (from summary.json of the categorical n=500 eval) ==")
    import json
    with open(ROOT / "evaluation/tmlr-n500-categorical/summary.json") as f:
        summ = json.load(f)
    p3s = summ["policies"]["phase3"]
    rc, cc = p3s["reward_calibration"], p3s["continue_calibration"]
    print(f"  reward: Pearson {rc['reward_pearson']:.3f}, event-AUC {rc['reward_event_auc']:.4f}"
          f" (n={rc['n']})")
    print(f"  continue: Brier {cc['brier']:.4f} ({cc['done_positive_count']} real terminations)")


if __name__ == "__main__":
    main()
