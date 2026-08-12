"""Reproduce every statistic in the paper's results section from the raw eval CSVs.

Inputs (all produced by dynamics/evaluate_env.py; ckpt provenance in each dir's
summary.json -> data_facts):
    evaluation/tmlr-n500-categorical/         BC + original phase3, eval seeds 0-499
    evaluation/tmlr-cat-seed{1,2,3}-n200/     seeds 1-3, eval seeds 0-199
    evaluation/tmlr-cat-seed{1,2,3}-ext300/   seeds 1-3, eval seeds 200-499
    evaluation/tmlr-cat-seed{4,5}-n500/       seeds 4-5, eval seeds 0-499
    evaluation/tmlr-bc-seed{11,12,13}-n500/   BC retrains (sec 6.3)
    evaluation/tmlr-fact-*/                   factorial cells (sec 6.4)
    ball_in_cup_catch.npz                     demo ceilings (sec 6.4)

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

    # ── §6.3: the baseline is itself a draw ─────────────────────────────
    print("\n== BC seed study (sec 6.3) ==")
    bc_runs = {"anchor": (bc_c, bc_r)}
    for s in (11, 12, 13):
        d = load(f"evaluation/tmlr-bc-seed{s}-n500/episodes.csv", "bc")
        bc_runs[f"seed{s}"] = (np.array([d[b_][1] for b_ in boards], float),
                               np.array([d[b_][0] for b_ in boards], float))
        print(f"  BC seed{s}: catch {bc_runs[f'seed{s}'][0].mean():.3f}")
    reps = np.array([bc_runs[f"seed{s}"][0].mean() for s in (11, 12, 13)])
    s_obs = float(np.std(reps, ddof=1))
    binom = float(np.mean([p * (1 - p) / n for p in reps]))
    s_dec = math.sqrt(max(s_obs**2 - binom, 0.0))
    # chi-square CI on an sd with df=2: chi2_{0.975,2}=7.3778, chi2_{0.025,2}=0.050636
    lo_sd, hi_sd = s_obs * math.sqrt(2 / 7.3778), s_obs * math.sqrt(2 / 0.050636)
    print(f"  replicas: mean {reps.mean():.3f}, observed sd {100*s_obs:.1f}pp "
          f"(chi2 95% CI [{100*lo_sd:.1f}, {100*hi_sd:.1f}]), deconvolved {100*s_dec:.1f}pp")
    print(f"  anchor ({bc_c.mean():.3f}) is the lowest of the 4 draws")
    print("  baseline swap (mean d_catch of the 6 RL runs vs each BC draw):")
    for name, (c_, _) in bc_runs.items():
        ds = [float((np.array([r[s_][1] for s_ in boards], float) - c_).mean())
              for r in runs.values()]
        print(f"    vs {name:7s} ({c_.mean():.3f}): {100*np.mean(ds):+.1f}pp, "
              f"{sum(x > 0 for x in ds)}/6 positive")

    # ── §6.4: the factorial (2 RL seeds x 3 fresh P2 draws) ─────────────
    print("\n== Factorial (sec 6.4) ==")
    cells = [("bc11-rl21", 11), ("bc11-rl22", 11), ("bc12-rl23", 12),
             ("bc12-rl24", 12), ("bc13-rl25", 13), ("bc13-rl26", 13)]
    deltas, by_draw = [], {}
    for cell, par in cells:
        d = load(f"evaluation/tmlr-fact-{cell}-n500/episodes.csv", "phase3")
        c_ = np.array([d[s_][1] for s_ in boards], float)
        pc = bc_runs[f"seed{par}"][0]
        dc = float((c_ - pc).mean())
        b_ = int(((c_ == 1) & (pc == 0)).sum())
        e_ = int(((c_ == 0) & (pc == 1)).sum())
        lo, hi = boot_ci(c_ - pc, rng, 8000)
        deltas.append(dc)
        by_draw.setdefault(par, []).append(dc)
        print(f"  {cell}: catch {c_.mean():.3f} vs parent {pc.mean():.3f} -> "
              f"{100*dc:+.1f}pp  CI [{100*lo:+.1f},{100*hi:+.1f}]  "
              f"McNemar p={sign_test(b_, e_):.1e}")
    draw_means = np.array([np.mean(v) for v in by_draw.values()])
    grand = float(np.mean(deltas))
    ms_b = 2 * float(np.sum((draw_means - grand) ** 2)) / 2      # k-1 = 2
    ms_w = float(sum((v[0] - v[1]) ** 2 / 2 for v in by_draw.values())) / 3
    var_b = max((ms_b - ms_w) / 2, 0.0)
    icc = var_b / (var_b + ms_w)
    F = ms_b / ms_w
    p_F = (1 + 2 * F / 3) ** -1.5                                # F(2,3) closed form
    s_draw = float(np.std(draw_means, ddof=1))
    tcrit2 = 4.303                                               # t_{0.975, df=2}
    se_d = s_draw / math.sqrt(3)
    print(f"  draw-level deltas: {[f'{100*x:+.1f}' for x in draw_means]}")
    print(f"  variance components: sigma_between {100*math.sqrt(var_b):.1f}pp "
          f"(sd of draw means {100*s_draw:.1f}, chi2 CI "
          f"[{100*s_draw*math.sqrt(2/7.3778):.1f}, {100*s_draw*math.sqrt(2/0.050636):.1f}]), "
          f"sigma_within {100*math.sqrt(ms_w):.1f}pp")
    print(f"  ICC {icc:.3f}, F(2,3) = {F:.1f}, p = {p_F:.1e}")
    print(f"  draw-level mean {100*grand:+.1f}pp, 95% CI "
          f"[{100*(grand - tcrit2*se_d):+.1f}, {100*(grand + tcrit2*se_d):+.1f}] "
          f"(not estimable at n=3)")
    demo = np.load(ROOT / "ball_in_cup_catch.npz", mmap_mode="r")
    rew = np.asarray(demo["rewards"], float)
    # per-frame rewards carry action-repeat-2 sums (values in {0,1,2}), so the
    # per-step normalizer is 2*T, matching the env's ~1000-scale returns
    print(f"  demo ceilings: catch {(rew.sum(axis=1) > 0).mean():.3f}, "
          f"normalized return {rew.sum(axis=1).mean() / (2 * rew.shape[1]):.3f}, "
          f"hold fraction {(rew > 0).mean():.3f}")

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
