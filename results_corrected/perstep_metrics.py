#!/usr/bin/env python
"""Exploitation measures from the per-step re-evaluation (phase3/perstep_reeval/<run>/per_step_<policy>.npz).

For every policy (boards 0-99, sampled readout, same boards for all):
  phantom      = mean reward the reward head predicts on steps whose realized reward is 0
  mirage share = fraction of NEVER-CAUGHT games whose predicted return (sum of predicted rewards) exceeds 100
                 (threshold from the old paper's sec 7.1 statistic: 13 of 50 probed games in the collapsing draw)
  belief/real  = mean predicted return per game vs mean realized return per game
Measures defined 2026-09-25 (external review) before the per-step data existed. Old runs = defective pipeline (before arm);
old_bc11_* = the collapsing draw, i.e. the positive control.
"""
import csv, glob, os, sys
import numpy as np

R = os.path.expanduser("<results-backup>/phase3/perstep_reeval")
OUT = os.path.expanduser("<results-backup>/phase3/analysis")
THRESH = 100.0

rows = []
for run_dir in sorted(glob.glob(f"{R}/*/")):
    run = os.path.basename(run_dir.rstrip("/"))
    for f in sorted(glob.glob(f"{run_dir}/per_step_*.npz")):
        pol = os.path.basename(f)[len("per_step_"):-4]
        z = np.load(f)
        seeds, L = z["seed"], z["ep_len"]
        keep = seeds < 100                      # boards 0-99 for every run
        idx = np.r_[0, np.cumsum(L)]
        pr, ar = z["pred_reward"], z["actual_reward"]
        belief = np.array([pr[idx[i]:idx[i + 1]].sum() for i in range(len(L))])[keep]
        real = np.array([ar[idx[i]:idx[i + 1]].sum() for i in range(len(L))])[keep]
        caught = z["caught"][keep]
        step_mask = np.concatenate([np.full(L[i], keep[i]) for i in range(len(L))])
        zero = (ar == 0) & step_mask
        missed = ~caught
        rows.append(dict(
            run=run, policy=pol, pipeline="old" if run.startswith("old") else "new",
            games=int(keep.sum()), catch_rate=float(caught.mean()),
            phantom=float(pr[zero].mean()),
            belief=float(belief.mean()), real=float(real.mean()),
            missed=int(missed.sum()),
            mirage=int(((belief > THRESH) & missed).sum()),
            mirage_share=float(((belief > THRESH) & missed).sum() / missed.sum()) if missed.any() else float("nan"),
            max_belief_missed=float(belief[missed].max()) if missed.any() else float("nan"),
        ))

with open(f"{OUT}/perstep_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def show(sel, title):
    print(f"\n{title}")
    print(f"{'run':<22}{'pol':<7}{'catch':>6}{'phantom':>9}{'belief':>8}{'real':>7}{'missed':>7}{'mirage':>7}{'max belief(missed)':>20}")
    for r in sel:
        print(f"{r['run']:<22}{r['policy']:<7}{100*r['catch_rate']:>5.0f}%{r['phantom']:>9.4f}{r['belief']:>8.0f}{r['real']:>7.0f}"
              f"{r['missed']:>7}{r['mirage']:>7}{r['max_belief_missed']:>20.1f}")

new3 = [r for r in rows if r["pipeline"] == "new" and r["policy"] == "phase3"]
newbc = [r for r in rows if r["pipeline"] == "new" and r["policy"] == "bc"]
old3 = [r for r in rows if r["pipeline"] == "old" and r["policy"] == "phase3"]
oldbc = [r for r in rows if r["pipeline"] == "old" and r["policy"] == "bc"]
show(old3 + oldbc, "OLD (defective) pipeline — old_bc11_* is the collapsing draw (positive control)")
show(new3 + newbc, "NEW (corrected) pipeline")

def agg(sel, name):
    ph = np.array([r["phantom"] for r in sel]); mi = sum(r["mirage"] for r in sel); ms = sum(r["missed"] for r in sel)
    mx = np.nanmax([r["max_belief_missed"] for r in sel])
    print(f"{name:<34} n={len(sel):>2}  phantom {ph.min():.4f}-{ph.max():.4f} (median {np.median(ph):.4f})   "
          f"mirage games {mi}/{ms} missed   highest belief on a missed game {mx:.1f}")
print("\nSUMMARY")
agg(new3, "new imagination runs")
agg(newbc, "new parents (bc)")
agg([r for r in old3 if not r["run"].startswith("old_bc11")], "old imagination runs, healthy draws")
agg([r for r in old3 if r["run"].startswith("old_bc11")], "old imagination runs, collapsing draw")
agg(oldbc, "old parents (bc)")
print(f"\nsaved -> {OUT}/perstep_metrics.csv")
