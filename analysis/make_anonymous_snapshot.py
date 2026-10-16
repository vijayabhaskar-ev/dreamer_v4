"""Build the anonymized code snapshot for double-blind review.  Run from the code repo root:
    python analysis/make_anonymous_snapshot.py            -> release/code_anonymous.zip (release/ is git-ignored)

What it does: `git archive HEAD` (tracked files only, so no data, checkpoints, logs or the nested paper repo), then
  - drops files that exist only for the public, named release: CITATION.cff, README.md, assets/ (site artwork),
    analysis/render_hero.py (personal-site figure), analysis/paper_lint_v2.py (its banned-word list names the author)
  - rewrites the LICENSE copyright holder to "The authors"
  - rewrites the machine-local checkpoint paths recorded in evaluation/tmlr-aligned-*/summary.json (they contain the
    user name) to "<results-backup>/..."; nothing else in those files changes
  - writes a short anonymous README
  - scans every text file for identity strings and refuses to zip if any remain
Camera-ready: use the tagged public repository instead of this bundle."""
import io, os, re, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release" / "code_anonymous.zip"
DROP = ["CITATION.cff", "README.md", "assets/", "analysis/render_hero.py", "analysis/paper_lint_v2.py",
        "analysis/make_anonymous_snapshot.py",   # this script carries the identity list it scans for
        ".gitignore"]                            # its comments mention the public release and local tool folders   # this script carries the identity list it scans for
HOME_PREFIX = "/home/vijay/Documents/Projects/dreamer_v4_phase1_aligned_backup/"
IDENTITY = ["vijay", "eswaran", "outlook.com", "gmail.com", "github", "huggingface", "hf.co/", "hf hub", "hf-downloaded",
            "public repo", "personal site", ".claude", ".codex", "jarvis", "/home/", "wandb.ai/", "+05:30", " ist)", "runpod", "~/documents", "path.home()"]
ALLOWED = ["github.com/nicklashansen/dreamer4"]    # the public dataset source, not the authors
TEXT = {".py", ".sh", ".md", ".txt", ".cff", ".json", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".gitignore", ".log", ".out", ""}

README = """# Code for "Does Imagination Training Improve on Behavior Cloning?" (anonymous review copy)

PyTorch reimplementation of Dreamer 4 (tokenizer, world model with shortcut forcing, Phase 2 finetune with
reward, continue and BC heads, Phase 3 imagination training with PMPO) on `ball_in_cup` catch, plus the
analysis scripts behind every number in the paper.

## Layout
- `tokenizer/`, `dynamics/`, `imagination/`, `heads.py`: the three training phases and the heads
- `convert_hansen_to_npz.py`: builds the dataset file; the per-episode left shift of actions (defect 1 fix) is here
- `dynamics/trainer.py`: the shortcut loss as one normalized term (defect 2 fix)
- `dynamics/evaluate_env.py`: closed-loop evaluation in the real simulator (paired games, sampled actions)
- `analysis/phase1_timing_control.py`: the timing preference test; `analysis/phase1_paired_rollout.py`: paired rollouts
- `analysis/paper_numbers_v2.py`, `analysis/paper_tables_v2.py`: compute the numbers and tables of the paper. They were
  run on the authors' results folder, whose layout differs from this copy, so they do not run here unchanged. Their
  inputs are included where size allows: `results_corrected/` holds the analysis files flat and
  `evaluation/tmlr-aligned-p3-*` holds the evaluation of each run. Settings and parameter counts were read from the
  checkpoints and the demonstration statistics from the dataset file, neither of which is included, and a few numbers
  come from training logs
- `evaluation/tmlr-*/`: per-game results (`episodes.csv`) and settings (`summary.json`) of every evaluated policy:
  `tmlr-aligned-bc-*` are the 8 corrected BC parents, `tmlr-aligned-p3-*` the 24 imagination policies (and their
  epoch 10 checkpoints, `-e10-`), the rest are the defective pipeline's policies
- `results_corrected/`: the per-run table behind Table 1 (`p3_runs.csv`), the training curves summary
  (`p3_wandb_curves.csv`), the per-game reward analysis (`perstep_metrics.csv` and its script), the timing test runner
  and its summary (`run_timing.py`, `summarize.py`, `timing_summary.csv`), the error-by-step-size and signal-level
  tables of the old and corrected Phase 1 models (`evaldyn_*`), the deterministic readout table, and the logs behind the
  timing and shuffle numbers (`final_checks.out`, `timing_control_OLD_e040.log`, `timing_tests.log` for the eight corrected
  draws and the old Phase 1 model, read by `summarize.py`, `timing_tests_old_draws.log` for the four defective pipeline draws,
  `evaldyn_OLD_e040/action_shuffle_sensitivity.csv`). Scripts that read the authors' local result folders have those paths
  replaced by the placeholders `<repo>` and `<results-backup>`; `imagined_vs_real.py` and `summarize.py` read the files beside them
- `launch/`: the launch scripts of the corrected pipeline's three training phases (`phase1.sh`, `phase2.sh`, `phase3.sh`);
  the tokenizer's settings are in the paper's hyperparameter table. `run_seed_study.sh` and `run_bc_seeds.sh` at the
  top level are the defective pipeline's scripts
- `tests/`: the unit tests cited in the appendix (`python -m tests.<name>`)

## Setup
`pip install -r requirements.txt`; for closed-loop evaluation also `pip install dm_control mujoco`.
Evaluation flags are recorded per policy in `evaluation/tmlr-*/summary.json` (`opts`); training flags are in the launch scripts under `launch/` and in the paper's appendix.

Checkpoints and the dataset are not included (size); they will be released with the camera-ready version.
"""

def main():
    tmp = Path(tempfile.mkdtemp(prefix="snapshot_"))
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar, check=True)
    for d in DROP:
        p = tmp / d
        if p.is_dir():
            for f in sorted(p.rglob("*"), reverse=True): f.unlink() if f.is_file() else f.rmdir()
            p.rmdir()
        elif p.exists(): p.unlink()
    # corrected-pipeline results and scripts that live outside the repo (backup dir): copied in, data and scripts only
    BK = Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup"
    import shutil
    for d in sorted(BK.glob("phase3/draw*/evaluation/tmlr-aligned-p3-*")):
        dst = tmp / "evaluation" / d.name; dst.mkdir(parents=True, exist_ok=True)
        for f in ("episodes.csv", "summary.json"): shutil.copy(d / f, dst / f)
    rc = tmp / "results_corrected"; rc.mkdir()
    for f in ("phase3/analysis/p3_runs.csv", "phase3/analysis/p3_wandb_curves.csv", "phase3/analysis/perstep_metrics.csv",
              "phase3/analysis/scripts/perstep_metrics.py", "phase3/analysis/scripts/imagined_vs_real.py",
              "timing_new_draws/run_timing.py", "timing_new_draws/summarize.py", "timing_new_draws/timing_summary.csv",
              "det_readout_check/readout_table.csv"):
        shutil.copy(BK / f, rc / Path(f).name)
    for tag in ("OLD_e320", "JOINT_e320", "OLD_e040"):
        for f in ("d_bucket_metrics.csv", "tau_d_joint_metrics.csv", "summary.json"):
            src = BK / f"eval_held_out/evaldyn_{tag}" / f
            if src.exists(): (rc / f"evaldyn_{tag}").mkdir(exist_ok=True); shutil.copy(src, rc / f"evaldyn_{tag}" / f)
    for f in ("eval_held_out/final_checks.out", "eval_held_out/timing_control_OLD_e040.log", "timing_new_draws/timing_tests.log"):
        shutil.copy(BK / f, rc / Path(f).name)
    shutil.copy(BK / "old_draw_forensics/timing_tests.log", rc / "timing_tests_old_draws.log")
    shutil.copy(BK / "eval_held_out/evaldyn_OLD_e040/action_shuffle_sensitivity.csv", rc / "evaldyn_OLD_e040" / "action_shuffle_sensitivity.csv")
    # launch scripts that actually ran (provenance_joint/run_phase1.sh did NOT run; see SETUP_FACTS "Exact command")
    ln = tmp / "launch"; ln.mkdir()
    import json
    args = json.load(open(BK / "wandb_logs/wandb/run-20260920_083236-4bupwxn9/files/wandb-metadata.json"))["args"]
    eff, i = {}, 0                                   # later duplicate flags win, first-appearance order kept
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"): eff[args[i]] = args[i + 1]; i += 2
        else: eff[args[i]] = None; i += 1
    assert eff.pop("--dynamics-loss-norm") == "joint"
    eff.pop("--wandb-name", None)                    # the run name named the cloud vendor and the card; left out (said in the header)
    body = " \\\n  ".join(k if v is None else f"{k} {v}" for k, v in eff.items())
    (ln / "phase1.sh").write_text(
        "#!/bin/bash\n# Corrected Phase 1 run: the effective command, rebuilt from the arguments the run recorded\n"
        "# (where a flag was given twice, the later value won). The run also passed --dynamics-loss-norm joint; in this\n"
        "# code one joint normalizer is the only mode, so that flag no longer exists. The run's --wandb-name is left out;\n# it named the cloud vendor.\n"
        f"python -m dynamics.train_dynamics \\\n  {body}\n")
    p2 = (BK / "phase2/seed11/provenance/run_phase2.sh").read_text()
    p2 = p2.replace("Command = run_bc_seeds.sh (the paper's published categorical pipeline),", "Command = run_bc_seeds.sh (the first build's categorical pipeline),")
    assert "published" not in p2
    (ln / "phase2.sh").write_text(p2.replace("set -u\n", "# Draws 15 to 17 ran this same script with SEEDS set to their seeds.\nset -u\n", 1))
    p3 = (BK / "scratchpad_provenance_2026-09-29/run_phase3.sh").read_text()
    assert p3.count("--policies phase3,bc") == 2 and "eval = the paper's evaluate_env command on boards 0-499 with --policies phase3,bc." in p3   # comment + command
    p3 = p3.replace("eval = the paper's evaluate_env command on boards 0-499 with --policies phase3,bc.",
                    "eval = evaluate_env on boards 0-499. The evaluations\n# reported in the paper were rerun with --policies phase3 (the opts in each\n# evaluation/tmlr-aligned-p3-*/summary.json); the BC parents were evaluated separately.")
    (ln / "phase3.sh").write_text(p3.replace("--policies phase3,bc", "--policies phase3"))
    lic = tmp / "LICENSE"; lic.write_text(re.sub(r"Copyright \(c\) (\d{4}) .*", r"Copyright (c) \1 The authors", lic.read_text()))
    n_scrub = 0
    SCRUB = [(HOME_PREFIX, "<results-backup>/"), ("/home/vijay/Documents/Projects/dreamer_v4/", "<repo>/"), ("/home/vijay", "<home>"),
         ('A = os.path.expanduser("~/Documents/Projects/dreamer_v4_phase1_aligned_backup/phase3/analysis")', 'A = os.path.dirname(os.path.abspath(__file__))   # p3_runs.csv and p3_wandb_curves.csv sit beside this script'),
         ("~/Documents/Projects/dreamer_v4_phase1_aligned_backup", "<results-backup>"), ('Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup/phase3/analysis/p3_runs.csv"', 'Path("<results-backup>/phase3/analysis/p3_runs.csv")'),
         ('str(Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup")', '"<results-backup>"'), ('Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup"', 'Path("<results-backup>")'),
         ("~/Documents/Projects/dreamer_v4", "<repo>"), ("~/.conda/envs/dreamer_v4/bin/python", "python"), ("# override on RunPod:", "# override on a pod:"),
         ("2026-09-27 (14:55 IST)", "2026-09-27"), ("2026-09-27T21:17:16+05:30", "2026-09-27"),
             ("# TMLR paper experiments — CATEGORICAL pipeline (the published artifact:\n# GitHub/HF agent_bc.pt + agent_imagination_rl.pt were stripped from these dirs).",
              "# Defective pipeline (first build), categorical heads."),
             ("#   Stage 0: n=500 closed-loop eval of the PUBLISHED categorical pipeline", "#   Stage 0: n=500 closed-loop eval of the first build's categorical pipeline"),
             ("(phase3 vs bc vs random) -> the paper's headline table.", "(phase3 vs bc vs random)."),
             ("THIS script is the paper's main pipeline.", "THIS script was the first build's main pipeline."),
             ("# ── Stage 0: headline n=500 on the published categorical pipeline ──", "# ── Stage 0: n=500 on the first build's categorical pipeline ──"),
             ("# Stage 5 — CategoricalPolicyHead (the shipped head behind the published\n# numbers).", "# Stage 5 — CategoricalPolicyHead (the head behind the paper's\n# numbers)."),
             ("# Overridable for pod runs (HF-downloaded paths). Weight-identical to local files\n# (verified: tokenizer weights md5 d1c70f9a... in both stripped and full checkpoints).",
              "# Paths are overridable for pod runs.")]
    for f in sorted(tmp.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT: continue
        t = f.read_text(errors="ignore"); t2 = t
        for a_, b_ in SCRUB: t2 = t2.replace(a_, b_)
        if t2 != t: n_scrub += sum(t.count(a_) for a_, _ in SCRUB); f.write_text(t2)
    (tmp / "README.md").write_text(README)
    bad = []
    for f in sorted(tmp.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT: continue
        try: t = f.read_text(errors="ignore").lower()
        except Exception: continue
        for a in ALLOWED: t = t.replace(a, "")
        for w in IDENTITY:
            if w in t: bad.append((str(f.relative_to(tmp)), w))
    if bad:
        for b in bad: print("IDENTITY:", *b)
        sys.exit("refusing to zip")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file(): z.write(f, f"dreamer4_code/{f.relative_to(tmp)}")
    n = sum(1 for f in tmp.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} files, {n_scrub} paths scrubbed, identity scan clean)")

if __name__ == "__main__":
    main()
