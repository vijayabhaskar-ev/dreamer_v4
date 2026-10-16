"""Strip the optimizer state from the corrected pipeline's checkpoints and write inference-only copies for release.

usage: python analysis/release_corrected_checkpoints.py [--out release/corrected_checkpoints]
Reads the tokenizer from this repo and the Phase 1, Phase 2 and Phase 3 checkpoints from the results backup
(the folder behind the paper's numbers). Every output is re-loaded and its tensor keys compared with the source.
"""
import argparse, hashlib, json, torch
from pathlib import Path
BK = Path.home() / "Documents/Projects/dreamer_v4_phase1_aligned_backup"
REPO = Path(__file__).resolve().parents[1]
DROP = ("optimizer", "scaler", "optimizer_state_dict")

def sources():
    yield "tokenizer.pt", REPO / "checkpoints/checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt"
    yield "world_model_phase1_epoch320.pt", BK / "checkpoints-dynamics-phase1-aligned-joint/dynamics/dynamics_epoch_320.pt"
    for d in range(10, 18):
        yield f"phase2/draw{d}_epoch40.pt", BK / f"phase2/seed{d}/checkpoints-phase2-aligned-seed{d}/dynamics_epoch_040.pt"
        for s in (1, 2, 3):
            yield f"phase3/draw{d}_seed{s}_final.pt", BK / f"phase3/draw{d}/checkpoints-phase3-aligned-draw{d}-seed{s}/final.pt"

def tensor_keys(ck):
    out = []
    for k, v in ck.items():
        if isinstance(v, dict):
            out += [f"{k}.{kk}" for kk, vv in v.items() if torch.is_tensor(vv)]
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(REPO / "release/corrected_checkpoints")); a = ap.parse_args()
    out = Path(a.out) / "ball_in_cup/corrected"; manifest = []
    for rel, src in sources():
        assert src.exists(), src
        ck = torch.load(src, map_location="cpu", weights_only=False)
        dropped = [k for k in DROP if k in ck]
        for k in dropped: del ck[k]
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); torch.save(ck, dst)
        back = torch.load(dst, map_location="cpu", weights_only=False)
        src_keys = tensor_keys({k: v for k, v in torch.load(src, map_location="cpu", weights_only=False).items() if k not in DROP})
        assert tensor_keys(back) == src_keys, rel
        manifest.append({"file": f"ball_in_cup/corrected/{rel}", "source": str(src.relative_to(BK) if src.is_relative_to(BK) else src.relative_to(REPO)),
                         "dropped": dropped, "size_mb": round(dst.stat().st_size / 1e6, 1), "md5": hashlib.md5(dst.read_bytes()).hexdigest()})
        print(f"{rel}: {manifest[-1]['size_mb']} MB, dropped {dropped}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} files, {sum(m['size_mb'] for m in manifest)/1000:.2f} GB ->", out)

if __name__ == "__main__":
    main()
