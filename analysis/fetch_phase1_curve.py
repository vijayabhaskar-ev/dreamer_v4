"""Export per-epoch means of the Phase-1 flow-matching loss and learning rate from wandb.

Writes the CSV read by paper/figs/make_fig_phase1.py. Entity/project/run ids are
arguments on purpose (nothing identifying is hardcoded). Usage:

  python -m analysis.fetch_phase1_curve --entity E --project P \
      --run first=RUN_ID_A --run resumed=RUN_ID_B --first-max-epoch 40 --out phase1_flow_per_epoch.csv
"""
import argparse, csv, collections
import numpy as np
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True); ap.add_argument("--project", required=True)
ap.add_argument("--run", action="append", required=True, help="segment=run_id (repeatable, in order)")
ap.add_argument("--first-max-epoch", type=int, default=40, help="last epoch taken from the first segment")
ap.add_argument("--out", required=True)
a = ap.parse_args()
api = wandb.Api(timeout=120)
rows = []
for spec in a.run:
    seg, rid = spec.split("=", 1)
    run = api.run(f"{a.entity}/{a.project}/{rid}")
    acc = collections.defaultdict(list)
    for r in run.scan_history(keys=["epoch", "train/flow", "train/lr"], page_size=5000):
        if r.get("epoch") is None or r.get("train/flow") is None: continue
        acc[int(r["epoch"])].append((float(r["train/flow"]), float(r["train/lr"])))
    for e in sorted(acc):
        if seg == "first" and e > a.first_max_epoch: continue
        v = np.array(acc[e]); rows.append((e, seg, v[:, 0].mean(), v[:, 1].mean(), len(v)))
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["epoch", "segment", "flow_loss_epoch_mean", "lr_epoch_mean", "n_logged_steps"]); w.writerows(rows)
print(f"wrote {len(rows)} epochs -> {a.out}")
