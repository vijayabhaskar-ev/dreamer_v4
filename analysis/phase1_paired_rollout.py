"""Paired rollout comparison: same held-out windows, same noise, each model fed the actions of ITS OWN training file
through the loader's slice. K=4 sampling, 4 context frames, 4 generated frames. Episode-clustered statistics."""
import os, sys, numpy as np, torch
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(S)); sys.path.insert(0, S)
import action_alignment_ask_the_weights as A
from dynamics.evaluate_dynamics import autoregressive_rollout
from tokenizer.dataset import OfflineDataset
T,C,H,NBAT,BS=8,4,4,40,32
dev=A.resolve_device("cuda")
ds=OfflineDataset("ball_in_cup_catch_aligned.npz", seq_len=T, batch_size=BS, steps_per_epoch=1, split="val")
acts={"aligned":np.asarray(ds.actions), "old":np.load("ball_in_cup_catch.npz")["actions"]}
rng=np.random.default_rng(7)
plan=[(ds.allowed_idx[rng.integers(0,len(ds.allowed_idx),BS)], rng.integers(0, ds.episode_len-T+1, BS)) for _ in range(NBAT)]
models=[("OLD  e040","checkpoints/checkpoints-dynamics-iter46-phase1-bf16/dynamics/dynamics_epoch_040.pt","old"),
        ("NEW  e240",sys.argv[1]+"/dynamics_epoch_240.pt","aligned"),
        ("NEW  e320",sys.argv[1]+"/dynamics_epoch_320.pt","aligned")]
res={}; epi=np.concatenate([e for e,_ in plan])
for name,ck,which in models:
    tr,dcfg=A.load_phase1(ck,dev); out=[]
    with torch.no_grad():
        for bi,(eps,starts) in enumerate(plan):
            fr=torch.from_numpy(np.stack([ds.frames[e,s:s+T] for e,s in zip(eps,starts)])).permute(0,1,4,2,3).float().div(255).to(dev)
            a=torch.from_numpy(np.stack([acts[which][e,s:s+T-1] for e,s in zip(eps,starts)]).copy()).float().to(dev)   # the loader's slice
            z=tr.model.encode_frames(fr)
            torch.manual_seed(1000+bi)                                   # identical noise for every model
            zr,_=autoregressive_rollout(tr.model, tr.tokenizer, z, a, C, H, dcfg.K_inference, dcfg.tau_ctx, dcfg.context_length, dcfg.K_max)
            out.append(((zr-z[:,C:C+H])**2).mean(dim=(2,3)).cpu().numpy())     # (BS,H)
    res[name]=np.concatenate(out); del tr; torch.cuda.empty_cache()
    r=res[name]; print(f"{name}: rollout error by generated frame " + "  ".join(f"{v:.4f}" for v in r.mean(0)) + f"   | average {r.mean():.4f}")
def paired(a,b,label):
    d=res[a].mean(1)-res[b].mean(1); em=np.array([d[epi==e].mean() for e in np.unique(epi)])
    se=em.std(ddof=1)/np.sqrt(len(em)); z=em.mean()/se
    print(f"{label}: mean difference {d.mean():+.5f} ({100*d.mean()/res[b].mean():+.1f}%), 95% CI [{em.mean()-2.07*se:+.5f}, {em.mean()+2.07*se:+.5f}], episode-clustered z = {z:+.2f}, better in {100*(d<0).mean():.0f}% of windows")
print(f"\n{len(epi)} shared windows from {len(np.unique(epi))} held-out episodes")
paired("NEW  e320","OLD  e040","NEW e320 minus OLD e040")
paired("NEW  e240","OLD  e040","NEW e240 minus OLD e040")
paired("NEW  e320","NEW  e240","NEW e320 minus NEW e240")
