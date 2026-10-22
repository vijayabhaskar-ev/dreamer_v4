#!/usr/bin/env python
"""Imagined return (the Phase-3 training score) vs real catch rate, 24 corrected runs (8 draws x 3 seeds).

Exploratory, not pre-registered. Numbers -> RESULTS_SHEET.md "BU addendum 3".
Sections 1-7 were first run inline on 2026-09-27 and are saved here with the same logic and the
same random-number stream. Section 8 was added the same day: it checks the formula-based bar and p-value
against actually shuffling, and computes the exact n=8 bar (0.738) that the sheet quotes but section 1 does
not compute (section 1 gives the t-approximation, 0.707).
Inputs: p3_runs.csv (real catch rate, 500 boards per run), p3_wandb_curves.csv (imagined_return_last500).
"""
import csv, itertools, os
import numpy as np
from scipy import stats

A = os.path.dirname(os.path.abspath(__file__))   # p3_runs.csv and p3_wandb_curves.csv sit beside this script
real = {(int(r["draw"]), int(r["seed"])): 100 * float(r["p3_rate"]) for r in csv.DictReader(open(f"{A}/p3_runs.csv"))}
imag = {(int(r["draw"]), int(r["seed"])): float(r["imagined_return_last500"]) for r in csv.DictReader(open(f"{A}/p3_wandb_curves.csv"))}
D = list(range(10, 18)); S = [1, 2, 3]
x = np.array([[imag[(d, s)] for s in S] for d in D])   # 8 x 3: one row per draw, one column per seed
y = np.array([[real[(d, s)] for s in S] for d in D])

# 1. significance bar: smallest |rho| the t-approximation calls significant (two-sided 5%)
for n in (24, 8):
    t = stats.t.ppf(0.975, n - 2); print(f"critical |rho| at n={n}: {t/np.sqrt(n-2+t*t):.3f}")

# 2. all 24 runs treated as independent: rho, formula p, Fisher-z 95% CI (1.06 = Spearman's SE inflation)
r, p = stats.spearmanr(x.ravel(), y.ravel()); z = np.arctanh(r); se = 1.06 / np.sqrt(24 - 3)
print(f"pooled rho {r:+.3f}, naive p {p:.2f}, naive 95% CI [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}]")

# 3. hierarchical bootstrap: resample draws, then seeds within each drawn draw
rng = np.random.default_rng(0); bs = []
for _ in range(20000):
    di = rng.integers(0, 8, 8); si = rng.integers(0, 3, (8, 3))
    xs = x[di[:, None], si].ravel(); ys = y[di[:, None], si].ravel()
    if np.ptp(xs) > 0 and np.ptp(ys) > 0: bs.append(stats.spearmanr(xs, ys)[0])
print(f"hierarchical bootstrap 95% CI for pooled rho: [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

# 4. between draws: 8 draw means, exact permutation test over all 8! orderings
xm, ym = x.mean(1), y.mean(1); rb = stats.spearmanr(xm, ym)[0]
perm = [stats.spearmanr(xm, np.array(pp))[0] for pp in itertools.permutations(ym)]
print(f"between-draw rho {rb:+.2f}, exact permutation p {np.mean(np.abs(perm) >= abs(rb)-1e-12):.2f} (8! = {len(perm)} orders)")

# 5. within draws: subtract each draw's mean, then shuffle seeds only inside their own draw
xc = x - x.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True); rw = stats.pearsonr(xc.ravel(), yc.ravel())[0]
null = []
for _ in range(20000):
    yp = np.array([rng.permutation(row) for row in yc]); null.append(stats.pearsonr(xc.ravel(), yp.ravel())[0])
print(f"within-draw r {rw:+.2f}, within-draw permutation p {np.mean(np.abs(null) >= abs(rw)):.2f}")

# 6. how much of the within-draw real-catch spread is measurement noise (500 boards per run)?
noise_sd = np.sqrt(np.mean([(v/100) * (1 - v/100) / 500 for v in y.ravel()])) * 100
wsd = np.sqrt(((y - y.mean(1, keepdims=True))**2).sum() / (8 * 2))
true_sd = np.sqrt(max(wsd**2 - noise_sd**2, 0))
print(f"within-draw sd of real catch {wsd:.2f} pts; binomial noise per run ~{noise_sd:.2f} pts -> true seed-to-seed sd ~{true_sd:.2f}, reliability ~{true_sd**2/wsd**2:.2f}")
print(f"within-draw spread of imagined return: sd {np.sqrt((xc**2).sum()/16):.2f} (between-draw sd of means {xm.std(ddof=1):.2f})")

# 7. decision view: pick the run with the highest imagined return
order = np.argsort(-x.ravel()); yr = y.ravel()
print(f"picking by imagined return: top-1 real {yr[order[0]]:.1f}%, top-3 mean {yr[order[:3]].mean():.1f}% | all-run mean {yr.mean():.1f}% | best run {yr.max():.1f}% | worst {yr.min():.1f}%")

# 8. checks: does the section-1/2 formula agree with actually shuffling? exact bar at n=8?
rng8 = np.random.default_rng(0)   # separate generator, so sections 3 and 5 keep the inline run's stream
rx, ry = stats.rankdata(x.ravel()), stats.rankdata(y.ravel())
shuf = np.array([np.corrcoef(rx, rng8.permutation(ry))[0, 1] for _ in range(100_000)])
print(f"shuffle check, 24 runs (100k shuffles): middle 95% [{np.percentile(shuf, 2.5):+.3f}, {np.percentile(shuf, 97.5):+.3f}], "
      f"bar {np.percentile(np.abs(shuf), 95):.3f}, p {np.mean(np.abs(shuf) >= abs(r)-1e-12):.3f} (formula: p {p:.3f})")
r8 = np.arange(1, 9)
vals = np.array([1 - 6 * ((r8 - np.array(q))**2).sum() / (8 * 63) for q in itertools.permutations(r8)])
tail = lambda c: np.mean(np.abs(vals) >= c - 1e-12)
crit8 = next(c for c in np.unique(np.abs(vals)) if tail(c) <= 0.05)
print(f"exact bar at n=8 (all {len(vals)} orderings): {crit8:.3f} (tail prob {tail(crit8):.3f}; formula gives 0.707)")
