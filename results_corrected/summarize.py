"""Parse timing_tests.log (run_timing.py) into timing_summary.csv and score the predictions pre-registered in
paper/RESULTS_SHEET.md entry CA: P1 every new draw's lowest error at k = 0 in both conditions; P2 late-vs-correct
difference positive with episode-clustered z >= +2 in both conditions; P3 released old e320 lowest at k = -1 in both."""
import csv, os, re

D = os.path.dirname(os.path.abspath(__file__))
COND = {"training-style noise": "A", "next frame from pure noise": "B"}
rows, model, cond = [], None, None
for line in open(os.path.join(D, "timing_tests.log")):
    if m := re.match(r"\n?===== (.+?) =====(.*)", line.strip("\n")):
        model = m.group(1); cond = None
        if "FAILED" in m.group(2):
            rows.append(dict(model=model, cond="-", failed=True))
        continue
    for key, c in COND.items():
        if line.strip().startswith(key) and (m := re.search(r"error at k=0 \(correct timing\) = ([0-9.]+)", line)):
            cond = dict(model=model, cond=c, err_correct=float(m.group(1)), failed=False); rows.append(cond)
    if cond is None:
        continue
    if m := re.search(r"(two rows back|STALE \(previous row\)|next row|shuffled)\s+([+-][0-9.]+)% vs correct\s+\(episode-clustered z = ([+-][0-9.]+)\)", line):
        k = {"two rows back": "km2", "STALE (previous row)": "km1", "next row": "kp1", "shuffled": "shuf"}[m.group(1)]
        cond[f"{k}_pct"], cond[f"{k}_z"] = float(m.group(2)), float(m.group(3))
    if m := re.search(r"lowest error at k=([+-]?\d+)", line):
        cond["argmin_k"] = int(m.group(1))

keys = ["model", "cond", "failed", "err_correct", "argmin_k", "km1_pct", "km1_z", "km2_pct", "km2_z", "kp1_pct", "kp1_z", "shuf_pct", "shuf_z"]
with open(os.path.join(D, "timing_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

print(f"{'model':<58}{'cond':>5}{'err k=0':>9}{'argmin':>7}{'late %':>9}{'late z':>8}{'k=+1 %':>9}{'shuf %':>9}")
for r in rows:
    if r.get("failed"):
        print(f"{r['model']:<58}  FAILED"); continue
    print(f"{r['model'][:57]:<58}{r['cond']:>5}{r['err_correct']:>9.5f}{r.get('argmin_k', 99):>+7d}{r.get('km1_pct', float('nan')):>+9.2f}"
          f"{r.get('km1_z', float('nan')):>+8.1f}{r.get('kp1_pct', float('nan')):>+9.2f}{r.get('shuf_pct', float('nan')):>+9.2f}")

new = [r for r in rows if r["model"].startswith("NEW") and not r.get("failed")]
old = [r for r in rows if r["model"].startswith("OLD") and not r.get("failed")]
n_new_models = len({r["model"] for r in rows if r["model"].startswith("NEW")})
p1 = sum(r.get("argmin_k") == 0 for r in new)
p2 = sum(r.get("km1_pct", -1) > 0 and r.get("km1_z", -1) >= 2 for r in new)
p3 = sum(r.get("argmin_k") == -1 for r in old)
print(f"\nnew draws parsed: {n_new_models} models, {len(new)} condition rows (expected 8 / 16)")
print(f"P1 argmin at correct timing (k=0):            {p1}/{len(new)}  -> {'HOLDS' if len(new) == 16 and p1 == 16 else 'FAILS'}")
print(f"P2 late worse than correct with z >= +2:       {p2}/{len(new)}  -> {'HOLDS' if len(new) == 16 and p2 == 16 else 'FAILS'}")
print(f"P3 released old e320 argmin at late (k=-1):    {p3}/{len(old)}  -> {'HOLDS' if len(old) == 2 and p3 == 2 else 'FAILS'}")
if new:
    for c in "AB":
        v = sorted(r["km1_pct"] for r in new if r["cond"] == c and "km1_pct" in r)
        zz = sorted(r["km1_z"] for r in new if r["cond"] == c and "km1_z" in r)
        if v:
            print(f"new draws, condition {c}: late penalty {v[0]:+.1f}% to {v[-1]:+.1f}%, z {zz[0]:+.1f} to {zz[-1]:+.1f}")
