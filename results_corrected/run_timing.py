"""Timing-preference test (analysis/phase1_timing_control.py, unchanged) on the 8 NEW Phase-2 world models (seeds 10-17,
the models behind the 24-run result) and the released OLD Phase-1 epoch 320. Old data file, held-out split: rows are taken
from the OLD (unshifted) npz so every model is probed in the same coordinates (k=0 = physically correct timing).
Pre-registered in paper/RESULTS_SHEET.md entry CA (committed and pushed before this script ran)."""
import hashlib, os, sys, traceback
R = os.path.expanduser("<repo>"); os.chdir(R); sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "analysis"))
import phase1_timing_control as T, action_alignment_ask_the_weights as A
from tokenizer.dataset import OfflineDataset

BK = os.path.expanduser("<results-backup>")
MODELS = [(f"NEW P2 draw {d} (aligned data + joint loss, epoch 40)",
           f"{BK}/phase2/seed{d}/checkpoints-phase2-aligned-seed{d}/dynamics_epoch_040.pt") for d in range(10, 18)]
MODELS.append(("OLD Phase-1 epoch 320 (released; old Phase 2 started from it)",
               "checkpoints/checkpoints-dynamics-iter46-phase1-bf16-optim/dynamics/dynamics_epoch_320.pt"))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


device = A.resolve_device("cuda")
print(f"data ball_in_cup_catch.npz md5 {md5('ball_in_cup_catch.npz')}")
ds = OfflineDataset("ball_in_cup_catch.npz", seq_len=A.T, batch_size=A.B, steps_per_epoch=1, split="val")
print(f"held-out episodes: {len(ds.allowed_idx)}   windows: {A.NB} batches x {A.B}, T={A.T}", flush=True)
for tag, ckpt in MODELS:
    try:
        print(f"\n[{tag}] {ckpt}  md5 {md5(ckpt)}", flush=True)
        T.run(tag, ckpt, ds, device)
    except Exception:
        print(f"\n===== {tag} ===== FAILED"); traceback.print_exc()
print("\nALL TIMING TESTS DONE", flush=True)
