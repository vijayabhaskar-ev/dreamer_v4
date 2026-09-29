"""Positive control: which action timing does a checkpoint predict best with?
Rows are taken from the OLD (unshifted) npz so both models are probed in the same coordinates:
   k = -1  slot j <- old row start+j-1  = the move that produced the PREVIOUS frame (stale; what the old pipeline fed)
   k =  0  slot j <- old row start+j    = the move that PRODUCED frame j (physically correct; what the aligned file + loader feed)
Episode-clustered paired statistics (24 held-out episodes)."""
import os, sys, numpy as np, torch
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(S)); sys.path.insert(0, S)
import action_alignment_ask_the_weights as A
from dynamics.flow_matching import add_noise, sample_tau_and_d
from tokenizer.dataset import OfflineDataset
KS=[-2,-1,0,1]
def run(tag, ckpt, ds, device):
    tr,dcfg=A.load_phase1(ckpt, device); rng=np.random.default_rng(0); torch.manual_seed(0)
    T,Bt=A.T,A.B; r1={k:[] for k in KS+["shuf"]}; r2={k:[] for k in KS+["shuf"]}; epi=[]
    with torch.no_grad():
        for _ in range(A.NB):
            eps=ds.allowed_idx[rng.integers(0,len(ds.allowed_idx),Bt)]; starts=rng.integers(3, ds.episode_len-T-3, Bt)
            fr=torch.from_numpy(np.stack([ds.frames[e,s:s+T] for e,s in zip(eps,starts)])).permute(0,1,4,2,3).float().div(255).to(device)
            stored=torch.from_numpy(np.stack([ds.actions[e,s-3:s+T+3] for e,s in zip(eps,starts)]).copy()).float().to(device); epi.append(eps)
            z=tr.model.encode_frames(fr)
            tau1,d1=sample_tau_and_d(Bt,T,K_max=dcfg.K_max,device=device); tau1[:,0]=1-dcfg.tau_ctx; d1[:,0]=1/dcfg.K_max
            zn1,_=add_noise(z,tau1)
            tau2=torch.full((Bt,T),1-dcfg.tau_ctx,device=device); d2=torch.full((Bt,T),1/dcfg.K_max,device=device); tau2[:,-1]=0.0; d2[:,-1]=0.25
            zn2,_=add_noise(z,tau2); perm=torch.randperm(Bt,device=device)
            for k in KS+["shuf"]:
                a=A.feed(stored,-1)[perm] if k=="shuf" else A.feed(stored,k)
                r1[k].append(((tr.model(zn1,a,tau1,d1).z_hat-z)**2)[:,1:].mean(dim=(1,2,3)).cpu().numpy())
                r2[k].append(((tr.model(zn2,a,tau2,d2).z_hat-z)**2)[:,-1].mean(dim=(1,2)).cpu().numpy())
    epi=np.concatenate(epi)
    print(f"\n===== {tag} =====")
    for name,r in (("training-style noise, frames 1..7",r1),("next frame from pure noise (d=1/4)",r2)):
        r={k:np.concatenate(v) for k,v in r.items()}; ref=r[0]
        print(f"  {name}:   error at k=0 (correct timing) = {ref.mean():.5f}")
        for k in (-2,-1,1,"shuf"):
            dlt=r[k]-ref; ep_means=np.array([dlt[epi==e].mean() for e in np.unique(epi)])
            z=ep_means.mean()/(ep_means.std(ddof=1)/np.sqrt(len(ep_means)))
            lab={-2:"two rows back",-1:"STALE (previous row)",1:"next row","shuf":"shuffled"}[k]
            print(f"      {lab:<22} {100*dlt.mean()/ref.mean():+8.2f}% vs correct   (episode-clustered z = {z:+.1f})")
        best=min(KS,key=lambda kk:r[kk].mean()); print(f"      --> lowest error at k={best:+d}  ({'CORRECT timing' if best==0 else 'stale timing' if best==-1 else 'other'})")
    del tr; torch.cuda.empty_cache()
if __name__=="__main__":
    device=A.resolve_device("cuda")
    ds=OfflineDataset("ball_in_cup_catch.npz", seq_len=A.T, batch_size=A.B, steps_per_epoch=1, split="val")
    run("NEW run, epoch 320 FINAL (aligned data, joint loss)", sys.argv[1], ds, device)
    pass
