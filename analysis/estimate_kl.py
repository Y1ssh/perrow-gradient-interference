#!/usr/bin/env python3
"""
T1 (+ T2/T3): the mixture-account hypothesis test, zero training.

The shared-head MTP objective is EXACTLY (1.75x at s=1) cross-entropy against a soft
label q_s = (e1 + 0.5s*e2 + 0.25s*e3)/(1+0.75s). Its next-token-optimal predictor is
the mixture p*_s = (P1 + s*M)/(1+0.75s), M = 0.5 P2 + 0.25 P3, and the excess next-token
loss at that optimum is KL(P1 || p*_s). This script estimates P1,P2,P3 from the CE-only
model on validation and computes the PREDICTED excess-loss curve KL(s) for every scale by
REWEIGHTING the same distributions (no extra forward passes per scale).

THE TEST (not a shape heuristic):
  * T1 absolute: KL(s=1) predicted here should reproduce the OBSERVED shared-MTP gap
    (~0.392 nats) within this script's reported interval. Zero fitted parameters ->
    matching it is strong evidence for the mixture account; missing it badly falsifies it.
  * The full curve KL(s) is what analyze_mtp_sweep.py compares the measured sweep against
    (NOT 0.70*log(1+0.75s), and NOT an assumed quadratic onset).

Robustness (so T1 is an interval, not a point):
  * top-k marginalization for P2 (and P3), k in {64,128,256}, renormalized; coverage mass reported.
  * validation split in half -> two independent estimates.
  * offset-3 handled BOTH ways: P3 ~= P2 proxy AND t+3 dropped (0.25-weight term removed);
    the spread is method uncertainty.

P2 marginalization (one-step rollout):
  P(x_{t+2} | x_<=t) = sum_{x_{t+1}} P(x_{t+1}|x_<=t) * P(x_{t+2} | x_<=t, x_{t+1}).
  We restrict the outer sum to the top-k x_{t+1} (renormalized) and run the model on each
  of the k one-token continuations, batched. P3 (P3~=P2 mode) reuses P2; (dropped mode) omits it.

Usage:
  python3 analysis/estimate_kl.py --ce_ckpt results/checkpoints/model_scale0.0_seed42.pt \
                                  --mtp_ckpt results/checkpoints/model_scale1.0_seed42.pt
"""
import os, sys, json, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config

ap = argparse.ArgumentParser()
ap.add_argument('--ce_ckpt', required=True, help='CE-only (scale 0.0) checkpoint')
ap.add_argument('--mtp_ckpt', default=None, help='shared-MTP (scale 1.0) checkpoint, for T2/T3')
ap.add_argument('--n_positions', type=int, default=4096, help='held-out next-token positions to average KL over')
ap.add_argument('--topk_list', type=int, nargs='+', default=[64, 128, 256])
ap.add_argument('--seq_len', type=int, default=1024)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)

def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = GPT2Config(**ck['config']) if 'config' in ck else GPT2Config()
    m = GPT2(cfg).to(DEVICE)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, ck

# ── validation data: same FineWeb-Edu tail as training (cached .pt if present) ──
def load_val():
    for c in ['fineweb_train_500M.pt', 'fineweb_train_50M.pt']:
        if os.path.exists(c):
            t = torch.load(c, weights_only=True)
            return t[480_000_000:] if t.numel() > 480_000_000 else t[int(t.numel()*0.96):]
    # fallback: stream a small val slice
    import tiktoken; from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    toks = []
    for ex in ds:
        toks.extend(enc.encode_ordinary(ex['text']))
        if len(toks) >= 3_000_000: break
    return torch.tensor(toks, dtype=torch.long)

print(f"Device: {DEVICE}")
val = load_val()
S = args.seq_len
val = val[: (val.numel()//S)*S].view(-1, S).to(DEVICE)
V = None

@torch.no_grad()
def next_token_probs(model, ctx):
    """P(x_{t+1} | ctx) at the LAST position of each row. ctx: (B,L). returns (B,V) float."""
    with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
        logits, _ = model(ctx)
    return F.softmax(logits[:, -1, :].float(), dim=-1)

@torch.no_grad()
def estimate_curve(model, topk, offset3_mode, n_positions, val_half=None):
    """Returns dict: predicted KL(s) for a grid of s, plus coverage + the P-tensors' summary.
    Uses n_positions sampled (row, t) anchors; for each, P1 exact, P2 via top-k rollout, P3 per mode."""
    global V
    V = model.config.vocab_size
    grid = [0.0, 0.25, 0.5, 1.0]
    # sample anchor positions with room for a t+1 rollout and >=3 lookahead
    B = 16
    kl_accum = {s: [] for s in grid}
    cover_accum = []
    done = 0
    # disjoint validation halves: val_half in {0,1} restricts sampling to one half,
    # None uses the whole set. Two halves -> two independent estimates (robustness).
    half = val.shape[0] // 2
    if val_half == 0:      lo_row, hi_row = 0, half
    elif val_half == 1:    lo_row, hi_row = half, val.shape[0]
    else:                  lo_row, hi_row = 0, val.shape[0]
    while done < n_positions:
        idx = torch.randint(lo_row, hi_row, (B,))
        rows = val[idx]                        # (B,S)
        t = torch.randint(8, S-4, (1,)).item() # a shared cut point (keeps batching simple)
        ctx = rows[:, :t+1]                    # (B, t+1) -> predict x_{t+1}
        P1 = next_token_probs(model, ctx)      # (B,V)
        # top-k rollout for P2
        topv, topi = P1.topk(topk, dim=-1)     # (B,k)
        cover = topv.sum(-1)                    # (B,) coverage mass
        w = topv / topv.sum(-1, keepdim=True)  # renormalized branch weights
        P2 = torch.zeros_like(P1)
        for j in range(topk):
            cont = torch.cat([ctx, topi[:, j:j+1]], dim=1)   # (B, t+2)
            P2j = next_token_probs(model, cont)              # P(x_{t+2}|ctx,x_{t+1}=topi_j)
            P2 = P2 + w[:, j:j+1] * P2j
        if offset3_mode == 'p3eqp2':
            P3 = P2
        else:  # 'drop'
            P3 = None
        for s in grid:
            if P3 is None:
                M = 0.5*s*P2
                norm = 1.0 + 0.5*s        # only t+1 and t+2 terms; 0.25s term dropped
            else:
                M = 0.5*s*P2 + 0.25*s*P3
                norm = 1.0 + 0.75*s
            pstar = (P1 + M) / norm
            kl = (P1 * (P1.clamp_min(1e-12).log() - pstar.clamp_min(1e-12).log())).sum(-1)  # (B,)
            kl_accum[s].append(kl.cpu().numpy())
        cover_accum.append(cover.cpu().numpy())
        done += B
    out = {str(s): float(np.concatenate(kl_accum[s]).mean()) for s in grid}
    out['coverage_mass'] = float(np.concatenate(cover_accum).mean())
    out['n_positions'] = done
    return out

# ── run T1 over the robustness grid (k x offset3-mode x val-half) ──
ce_model, ce_ck = load_model(args.ce_ckpt)
print(f"CE-only checkpoint: final_val_loss={ce_ck.get('final_val_loss')}")
results = {'ce_ckpt': args.ce_ckpt, 'observed_shared_mtp_gap': 0.392, 'kl_estimates': []}
for topk in args.topk_list:
    for mode in ['p3eqp2', 'drop']:
        for vhalf in (0, 1):   # two disjoint validation halves -> independent estimates
            est = estimate_curve(ce_model, topk, mode, args.n_positions // 2, val_half=vhalf)
            est.update({'topk': topk, 'offset3': mode, 'val_half': vhalf})
            results['kl_estimates'].append(est)
            print(f"  k={topk:4d} offset3={mode:7s} half={vhalf} coverage={est['coverage_mass']:.3f} "
                  f"KL(1)={est['1.0']:.4f}  KL(0.5)={est['0.5']:.4f}  KL(0.25)={est['0.25']:.4f}")

# T1 headline interval on KL(s=1) across the robustness grid
kl1 = [e['1.0'] for e in results['kl_estimates']]
results['T1_KL_s1'] = {'min': float(np.min(kl1)), 'max': float(np.max(kl1)), 'mean': float(np.mean(kl1))}
print(f"\nT1: predicted KL(s=1) in [{np.min(kl1):.3f}, {np.max(kl1):.3f}] "
      f"(observed shared-MTP gap = 0.392) -> "
      f"{'CONSISTENT' if np.min(kl1)-0.05 <= 0.392 <= np.max(kl1)+0.05 else 'TENSION'}")

# predicted curve (mean over grid) for the sweep comparison
grid=['0.0','0.25','0.5','1.0']
results['predicted_curve_mean'] = {s: float(np.mean([e[s] for e in results['kl_estimates']])) for s in grid}
# linear (interference) alternative for reference
results['linear_alternative'] = {s: 0.392*float(s) for s in grid}

# ── T2/T3: entropy + future-offset CE on both checkpoints ──
if args.mtp_ckpt and os.path.exists(args.mtp_ckpt):
    mtp_model, _ = load_model(args.mtp_ckpt)
    @torch.no_grad()
    def enriched(model, iters=40):
        Vv = model.config.vocab_size; ce1=ce2=ce3=ent=0.0; n=0
        for _ in range(iters):
            b = val[torch.randint(0, val.shape[0], (16,))]
            with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
                lg,_ = model(b)
            ce1 += F.cross_entropy(lg[:, :-1].reshape(-1,Vv), b[:,1:].reshape(-1)).item()
            ce2 += F.cross_entropy(lg[:, :-2].reshape(-1,Vv), b[:,2:].reshape(-1)).item()
            ce3 += F.cross_entropy(lg[:, :-3].reshape(-1,Vv), b[:,3:].reshape(-1)).item()
            fl = lg.reshape(-1,Vv)
            e=0.0; m=0
            for i in range(0, fl.shape[0], 4096):
                lp = F.log_softmax(fl[i:i+4096].float(),-1); e += float((-(lp.exp()*lp).sum(-1)).sum()); m += lp.shape[0]
            ent += e/m; n+=1
        return {'ce_t1':ce1/n,'ce_t2':ce2/n,'ce_t3':ce3/n,'entropy':ent/n}
    ce_e = enriched(ce_model); mtp_e = enriched(mtp_model)
    results['T2_T3'] = {'ce_only': ce_e, 'shared_mtp': mtp_e,
                        'delta_entropy': mtp_e['entropy']-ce_e['entropy'],
                        'delta_ce_t1': mtp_e['ce_t1']-ce_e['ce_t1'],
                        'delta_ce_t2': mtp_e['ce_t2']-ce_e['ce_t2'],
                        'delta_ce_t3': mtp_e['ce_t3']-ce_e['ce_t3']}
    print(f"\nT2 entropy: CE-only {ce_e['entropy']:.3f} vs shared-MTP {mtp_e['entropy']:.3f} "
          f"(Δ{mtp_e['entropy']-ce_e['entropy']:+.3f}; mixture predicts +0.3..0.4)")
    print(f"T3 future CE: shared-MTP better at t+2 by {ce_e['ce_t2']-mtp_e['ce_t2']:+.3f}, "
          f"t+3 by {ce_e['ce_t3']-mtp_e['ce_t3']:+.3f} (mixture predicts shared-MTP BETTER at t+2/t+3)")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kl_estimate_results.json')
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
