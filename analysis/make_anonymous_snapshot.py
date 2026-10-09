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
        "analysis/make_anonymous_snapshot.py"]   # this script carries the identity list it scans for
HOME_PREFIX = "/home/vijay/Documents/Projects/dreamer_v4_phase1_aligned_backup/"
IDENTITY = ["vijay", "eswaran", "outlook.com", "gmail.com", "github.com/", "huggingface.co", "hf.co/", "jarvis", "/home/", "wandb.ai/"]
ALLOWED = ["github.com/nicklashansen/dreamer4"]    # the public dataset source, not the authors
TEXT = {".py", ".sh", ".md", ".txt", ".cff", ".json", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".gitignore", ""}

README = """# Code for "One Retrain Is Not Enough" (anonymous review copy)

PyTorch reimplementation of Dreamer 4 (tokenizer, world model with shortcut forcing, Phase 2 finetune with
reward, continue and BC heads, Phase 3 imagination training with PMPO) on `ball_in_cup` catch, plus the
analysis scripts behind every number in the paper.

## Layout
- `tokenizer/`, `dynamics/`, `imagination/`, `heads.py`: the three training phases and the heads
- `convert_hansen_to_npz.py`: builds the dataset file; the per-episode left shift of actions (defect 1 fix) is here
- `dynamics/trainer.py`: the shortcut loss as one normalized term (defect 2 fix)
- `dynamics/evaluate_env.py`: closed-loop evaluation in the real simulator (paired boards, sampled actions)
- `analysis/phase1_timing_control.py`: the timing preference test; `analysis/phase1_paired_rollout.py`: paired rollouts
- `analysis/paper_numbers_v2.py`, `analysis/paper_tables_v2.py`: regenerate every number and table in the paper
  from the raw evaluation outputs
- `evaluation/tmlr-*/`: per-game results (`episodes.csv`) and settings (`summary.json`) of every evaluated policy,
  old pipeline and corrected pipeline
- `tests/`: the unit tests cited in the appendix (`python -m tests.<name>`)

## Setup
`pip install -r requirements.txt`; for closed-loop evaluation also `pip install dm_control mujoco`.
Training commands and flags are recorded per run in `evaluation/tmlr-*/summary.json` (`opts`) and in the paper's appendix.

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
    lic = tmp / "LICENSE"; lic.write_text(re.sub(r"Copyright \(c\) (\d{4}) .*", r"Copyright (c) \1 The authors", lic.read_text()))
    n_scrub = 0
    for f in sorted((tmp / "evaluation").glob("tmlr-aligned-*/summary.json")):
        s = f.read_text(); c = s.count(HOME_PREFIX); n_scrub += c
        f.write_text(s.replace(HOME_PREFIX, "<results-backup>/"))
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
