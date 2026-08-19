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

    print(f"BC baseline: catch {bc_c.mean():.3f}, return {bc_r.mean():.1f}, "
          f"zero-return episodes {(bc_r == 0).mean():.1%}  (n={n})")
    rnd0 = load("evaluation/tmlr-n500-categorical/episodes.csv", "random")
    rr = np.array([rnd0[s_][0] for s_ in boards], float)
    print(f"random floor: catch {np.array([rnd0[s_][1] for s_ in boards]).mean():.3f}, "
          f"return {rr.mean():.1f}")
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
        u = np.linspace(abs(t), abs(t) + 60, 400_000)
        dens = 2 / (math.sqrt(5 * math.pi) * 1.329340388) * (1 + u**2 / 5) ** -3
        p_t = 2 * np.trapezoid(dens, u)
        print(f"  {label}: p = {p_t:.3f} (two-sided, df={k-1})")
        print(f"  {label}: mean {m_:+.4f}  sd {sd:.4f}  t({k-1})={t:.2f}  "
              f"95% CI [{m_ - tcrit * se:+.4f}, {m_ + tcrit * se:+.4f}]  ({sum(x > 0 for x in v)}/{k} positive)")

    print("\n== Committee estimate (per-board mean of runs, paired vs BC, board bootstrap) ==")
    Cbar, Rbar = np.mean(Cs, axis=0), np.mean(Rs, axis=0)
    print(f"  committee catch {Cbar.mean():.4f} (vs BC {bc_c.mean():.4f}); "
          f"committee return {Rbar.mean():.1f} (vs BC {bc_r.mean():.1f})")
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
    a2, b2 = (pp - pb) * mu_b, pp * (mu_p - mu_b)
    print(f"  alternative attribution (interaction to conditional channel): "
          f"{a2/tot:.0%}/{b2/tot:.0%}")

    print("\n== Training-seed variance decomposition (sec 6.2 convention) ==")
    obs_sd = float(np.std(d_catch, ddof=1))
    # binomial-only correction: the shared baseline's eval noise is constant
    # across the six deltas and cancels in their spread (paper footnote 1)
    rl_rates = [float(np.array([d[s][1] for s in boards]).mean()) for d in runs.values()]
    binom_se = math.sqrt(float(np.mean([p_ * (1 - p_) / n for p_ in rl_rates])))
    train_sd = math.sqrt(max(obs_sd ** 2 - binom_se ** 2, 0.0))
    print(f"  across-run sd {obs_sd:.4f} | per-run binomial se {binom_se:.4f} "
          f"| deconvolved training-seed sd {train_sd:.4f}")

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
    cell_ses = []
    for cell, par in cells:
        d = load(f"evaluation/tmlr-fact-{cell}-n500/episodes.csv", "phase3")
        dc_b = np.array([d[s_][1] for s_ in boards], float) - bc_runs[f"seed{par}"][0]
        cell_ses.append(dc_b.std(ddof=1) / math.sqrt(n))
    print(f"  per-cell eval noise floor: {100*float(np.mean(cell_ses)):.1f}pp")
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
    print(f"  ICC {icc:.4f}, F(2,3) = {F:.1f}  (parametric p = {p_F:.1e}; "
          f"NOT quoted in the paper)")
    # Searle F-based CI for the ICC (n=2 per group): F crits from closed forms
    # are not available for (2,3)/(3,2); use scipy-free constants
    Fc23, Fc32 = 16.0441, 39.1655   # F_{0.975}(2,3), F_{0.975}(3,2)
    FL, FU = F / Fc23, F * Fc32
    print(f"  ICC 95% CI (Searle): [{(FL-1)/(FL+1):.4f}, {(FU-1)/(FU+1):.5f}]")
    # exact enumeration: all 15 ways to pair the six cell deltas
    import itertools
    def F_of(groups):
        gm = np.array([np.mean(g) for g in groups])
        msb_ = 2 * float(((gm - gm.mean()) ** 2).sum()) / 2
        msw_ = float(sum((g[0] - g[1]) ** 2 / 2 for g in groups)) / 3
        return msb_ / msw_ if msw_ > 0 else float("inf")
    seen, Fs = set(), []
    for perm in itertools.permutations(range(6)):
        key = tuple(sorted(tuple(sorted(perm[i:i+2])) for i in (0, 2, 4)))
        if key not in seen:
            seen.add(key)
            Fs.append(F_of([[deltas[i] * 100 for i in pr] for pr in key]))
    Fs.sort(reverse=True)
    print(f"  enumeration: {len(Fs)} possible pairings; max F {Fs[0]:.1f} "
          f"(the by-draw grouping), runner-up {Fs[1]:.1f} "
          f"(ratio {Fs[0]/Fs[1]:.1f}x); exact permutation p = 1/{len(Fs)}")
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

    print("\n== Closed-loop calibration echo (every released summary.json) ==")
    import json
    import glob
    rows = []
    for sj in sorted(glob.glob(str(ROOT / "evaluation/tmlr-*/summary.json"))):
        with open(sj) as f:
            summ = json.load(f)
        for pol, ps in summ.get("policies", {}).items():
            rc = ps.get("reward_calibration")
            if not rc:
                continue  # random floor never queries the model
            cc = ps.get("continue_calibration", {})
            rows.append((sj.split("evaluation/")[-1].split("/")[0], pol,
                         rc.get("reward_mae"), rc["reward_pearson"],
                         rc["reward_event_auc"], cc.get("brier")))
            print(f"  {rows[-1][0]:32s} {pol:7s} MAE {rows[-1][2]:.4f}  "
                  f"pearson {rows[-1][3]:.3f}  AUC {rows[-1][4]:.3f}  brier {rows[-1][5]:.4f}")
    p3 = [r for r in rows if r[1] == "phase3" and "fact-" not in r[0] and "readout" not in r[0]
          and "stoch" not in r[0] and "mj337" not in r[0]]
    print(f"  six-run spread (anchor-study phase3): MAE {min(r[2] for r in p3):.4f}-"
          f"{max(r[2] for r in p3):.4f}, pearson {min(r[3] for r in p3):.4f}-"
          f"{max(r[3] for r in p3):.4f}")


def sec7_block():
    """Section 7 twins: every number quoted in sec7 that is not already
    printed above. Sources: paper/probe_artifacts/{kl_probe_results,
    training_finals,weight_forensics}.json + evaluation/*/episodes.csv."""
    import csv
    import itertools
    import json
    from math import comb, sqrt

    pa = ROOT / "paper/probe_artifacts"
    finals = json.load(open(pa / "training_finals.json"))
    probe = json.load(open(pa / "kl_probe_results.json"))
    wf = json.load(open(pa / "weight_forensics.json"))

    print("\n== sec7.1: training-side blindness (factorial) ==")
    for k, v in finals["factorial"].items():
        print(f"  {k}: imagined {v['imagined_return']:7.2f}  per-step reward "
              f"{v['mean_reward']:.3f}  mean_continue {v['mean_continue']:.5f}  "
              f"real catch {v['real_catch']:.3f}")
    ds = finals["expert_dataset"]["per_step_reward"]
    print(f"  expert npz per-step reward mean {ds['mean']} on [{ds['min']}, {ds['max']}]")
    lo = [finals["factorial"][k]["real_catch"] for k in ("bc11-rl21", "bc11-rl22")]
    hi = [finals["factorial"][k]["real_catch"] for k in ("bc12-rl23", "bc12-rl24")]
    print(f"  real-catch factor (midpoints): {(sum(hi)/2)/(sum(lo)/2):.2f}")
    print("  within-draw pair diffs vs binomial noise (n=500/cell):")
    cells = {"bc11": (0.126, 0.118), "bc12": (0.648, 0.704), "bc13": (0.288, 0.294)}
    for k, (a, b) in cells.items():
        se = sqrt(a * (1 - a) / 500 + b * (1 - b) / 500)
        print(f"    {k}: diff {a-b:+.3f}  se_diff {se:.4f}  ({abs(a-b)/se:.2f} sigma)")

    print("\n== sec7.1/7.5: anchor within-draw ranking, 6 runs -> 15 pairs ==")
    a6 = [(v["imagined_return"], v["real_catch"]) for v in finals["anchor"].values()]
    agree = sum((x[0] > y[0]) == (x[1] > y[1]) for x, y in itertools.combinations(a6, 2))
    p = sum(comb(15, k) for k in range(agree, 16)) / 2 ** 15
    imag = [x[0] for x in a6]
    real = [x[1] for x in a6]
    print(f"  concordant {agree}/15, exact one-sided p = {p:.4f}")
    print(f"  imagined band {min(imag):.2f}-{max(imag):.2f} "
          f"(relative {100*(max(imag)-min(imag))/min(imag):.1f}%)  real band "
          f"{min(real):.3f}-{max(real):.3f} (relative "
          f"{100*(max(real)-min(real))/min(real):.1f}%, {100*(max(real)-min(real)):.1f} points)")

    print("\n== sec7.1/7.2: closed-loop probe (kl_probe_results.json) ==")
    for e in probe:
        pe = e["per_ep"]
        n10 = e["prefix_pearson"].get("10", {})
        print(f"  {e['tag']}: probe n={e['repro']['n']}  MAE@10 {n10.get('mae', float('nan')):.4f}  "
              f"pred {pe['pred_return_mean']:.1f} vs actual {pe['actual_return_mean']:.1f} "
              f"(gap {pe['pred_minus_actual_mean']:+.1f})  frac(pred>100 & actual=0) "
              f"{pe['frac_eps_pred_gt_100_actual_0']:.2f}")
        bc = e["bc_pearson_nbc"]
        print(f"      BC parent: MAE {bc['mae']:.4f} over n_eps={bc['n_eps']} "
              f"(pearson {bc['pearson']})")
    # hallucinated-success contrast: 13/50 (exploiting) vs 0/25 and 0/15
    def _fisher(a, b, c, d):
        n, r1, c1 = a + b + c + d, a + b, a + c
        return sum(comb(r1, x) * comb(n - r1, c1 - x)
                   for x in range(a, min(r1, c1) + 1)) / comb(n, c1)
    print(f"  hallucinated-success 13/50 vs 0/25 p={_fisher(13, 37, 0, 25):.4f}, "
          f"vs 0/15 p={_fisher(13, 37, 0, 15):.4f}, vs pooled 0/40 "
          f"p={_fisher(13, 37, 0, 40):.5f}  (0/40 exact 95% upper bound "
          f"{1 - 0.05 ** (1/40):.3f})")
    maes = {"exploit": 0.29305, "record": 0.02783, "declining": 0.02402,
            "bc11": 0.02534, "bc12": 0.03895}
    print(f"  separation: exploiting 0.293 vs worst healthy 0.039 -> "
          f"{0.29305/0.03895:.1f}x at the closest point (no overlap)")

    print("\n== sec7.3: KL + weight forensics ==")
    for e in probe:
        own = e["kl"]["p3_states"]["kl_p3_to_bc_mean"]
        par = e["kl"]["bc_states"]["kl_p3_to_bc_mean"]
        print(f"  {e['tag']}: KL own-route {own:.4f}  parent-route {par:.4f}  "
              f"ratio {own/par:.2f}")
    print(f"  compounding: 0.3186*500 = {0.3186*500:.0f} nats (exploit), "
          f"0.1112*500 = {0.1112*500:.0f} nats (record)")
    for k, v in wf["children"].items():
        own = v["own_parent"].replace("seed", "")
        print(f"  {k}: L2 own {v['l2_vs_seed' + own]:.3f} (norm {v['norm']:.1f}, "
              f"{100*v['l2_vs_seed' + own]/v['norm']:.1f}%)  cos own "
              f"{v['cos_vs_seed' + own]:.6f}")

    print("\n== sec7.4: same-board readout comparison (episodes.csv) ==")

    def _load(p):
        return {int(r["seed"]): (float(r["return"]), int(r["success"]))
                for r in csv.DictReader(open(p)) if r["policy"] == "phase3"}

    det = _load(ROOT / "evaluation/tmlr-readout-mean-n500/episodes.csv")
    samp = _load(ROOT / "evaluation/tmlr-n500-categorical/episodes.csv")
    wins = sorted(s for s in det if det[s][1] == 1)
    rets = sorted(det[s][0] for s in wins)
    mid = rets[len(rets)//2 - 1:len(rets)//2 + 1]
    print(f"  mean-readout catches {len(wins)}: return|catch mean "
          f"{sum(rets)/len(rets):.1f}  median {sum(mid)/2:.0f}  "
          f">=900: {sum(r >= 900 for r in rets)}/{len(rets)}")
    sw = [samp[s] for s in wins]
    print(f"  sampling on those boards: return {sum(r for r, _ in sw)/len(sw):.1f}  "
          f"caught {sum(c for _, c in sw)}/{len(sw)} "
          f"({100*sum(c for _, c in sw)/len(sw):.0f}% vs "
          f"{100*sum(c for _, c in samp.values())/len(samp):.0f}% overall)")

    print("\n== sec7.5: BC parents' training-side metrics ==")
    for k, v in finals["bc_parents"].items():
        print(f"  {k}: train {v['train_loss']}  val {v['val_loss']}  BC catch {v['bc_catch']}")
    ds = finals["expert_dataset"]
    print(f"  dataset: {ds['file']}  {ds['bytes']} bytes  md5 {ds['md5']}")


if __name__ == "__main__":
    main()
    sec7_block()
