"""Pre-upload checks for the v2 paper. Run from the code repo root:  python analysis/paper_lint_v2.py
Reports (never edits): hand-typed digits in prose, a bare % after a macro or digit, banned words and patterns,
hyphenated compounds, colons in prose, identity strings (anonymity), unknown citation keys, page count."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; PAPER = ROOT / "paper"
files = sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex", PAPER / "figures" / "fig_beforeafter.tex"] + sorted((PAPER / "generated").glob("table_*.tex"))
bib_keys = set(k.strip() for k in re.findall(r"@\w+\{\s*([^,\s]+)\s*,", (PAPER / "refs.bib").read_text(), re.I))
BANNED = ["pre-registered", "preregistered", "converged", "contradicts", "before the runs", "plan-specified", "pre-specified",
          "novel", "significant", "beats the demonstrations", "released model", "to the best of our knowledge", "state of the art",
          "vijay", "eswaran", "github.com", "huggingface", "wandb", "runpod", "@gmail", "@outlook"]
PROSE_ONLY = [PAPER / "sections" / f"{n}.tex" for n in ("00_abstract", "01_intro", "02_setup", "03_results", "05_defects", "06_diagnostics", "07_discussion", "08_related", "09_limitations")]
problems = 0

def strip(line):
    line = re.sub(r"\\(cite[pt]?|Cref|cref|ref|label|input|includegraphics)\{[^}]*\}", "", line)
    line = re.sub(r"\$[^$]*\$", "", line)                     # inline math
    line = line.replace("95\\%", "").replace("1,000 evaluation episodes", "")   # the CI level; Dreamer 4's own number
    line = re.sub(r"\\[A-Za-z]+", "", line)                    # macros
    return line

for f in files:
    for n, raw in enumerate(f.read_text().split("\n"), 1):
        if raw.lstrip().startswith("%"): continue
        code = re.sub(r"(?<!\\)%.*$", "", raw)                  # drop a real comment, keep \%
        tag = f"{f.relative_to(PAPER)}:{n}"
        if re.search(r"(\\[A-Za-z]+|\d)%", code):               # bare % swallows the rest of the line
            print(f"BARE %        {tag}: {code[:90]}"); problems += 1
        for key in re.findall(r"\\cite[pt]?\{([^}]*)\}", code):
            for k in key.split(","):
                if k.strip() not in bib_keys: print(f"UNKNOWN KEY   {tag}: {k.strip()}"); problems += 1
        text = strip(code)
        low = text.lower()
        for w in BANNED:
            if w in low: print(f"BANNED '{w}'  {tag}: {text.strip()[:90]}"); problems += 1
        if f in PROSE_ONLY:
            for m in re.finditer(r", not ", text): print(f"X-NOT-Y       {tag}: ...{text[max(0,m.start()-40):m.end()+30].strip()}"); problems += 1
            for m in re.finditer(r"\b[A-Za-z]+-[A-Za-z]+\b", text): print(f"HYPHEN        {tag}: {m.group(0)}"); problems += 1
            if ":" in text: print(f"COLON         {tag}: ...{text[max(0,text.find(':')-50):text.find(':')+30].strip()}"); problems += 1
            for m in re.finditer(r"\d[\d,.]*", text):
                ctx = text[max(0, m.start()-12):m.end()+8]
                if re.search(r"(Phase|Dreamer|frame|defect|row|Eq\.|step|Figure|Table|Section) *$", text[:m.start()]): continue
                print(f"DIGIT         {tag}: ...{ctx.strip()}..."); problems += 1
pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(PAPER / "main.pdf")], capture_output=True, text=True).stdout)
print(f"PAGES         {pages.group(1) if pages else '?'} (TMLR fast track: main text <= 12)")
print(f"{problems} items to look at")
