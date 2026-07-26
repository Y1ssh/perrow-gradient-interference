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
  * top-k marginalization for P2 (and P3), k in {256,1024,2048} by default, renormalized; coverage mass reported.
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
ap.add_argument('--topk_list', type=int, nargs='+', default=[256, 1024, 2048])
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
    # Position sampling uses a DEDICATED CPU generator seeded ONLY by (offset3-mode, val_half) --
    # explicitly NOT by k. This guarantees k=256/1024/2048 evaluate the IDENTICAL anchor positions,
    # so the coverage->KL(s=1) curve is a pure function of k (the ~0.004-0.009-nat k-trend the paper
    # reads is not drowned by position-sampling noise, whose half-to-half spread alone is ~0.012).
    # A separate generator (not the global RNG) means no other RNG consumer can perturb the positions.
    _sd = {'p3eqp2': 1000, 'drop': 2000}[offset3_mode] + (val_half if val_half is not None else 9)
    pos_gen = torch.Generator(device='cpu').manual_seed(args.seed + _sd)
    # sample anchor positions with room for a t+1 rollout and >=3 lookahead
    B = 16
    kl_accum = {s: [] for s in grid}
    cover_accum = []
    # T2-proper: entropy of the true next-token dist P1 vs the soft-label optimum q* at s=1,
    # computed from the SAME top-k arrays (so the truncation bias partially cancels in the difference).
    ent_p1_accum, ent_qstar_accum = [], []
    done = 0
    # disjoint validation halves: val_half in {0,1} restricts sampling to one half,
    # None uses the whole set. Two halves -> two independent estimates (robustness).
    half = val.shape[0] // 2
    if val_half == 0:      lo_row, hi_row = 0, half
    elif val_half == 1:    lo_row, hi_row = half, val.shape[0]
    else:                  lo_row, hi_row = 0, val.shape[0]
    while done < n_positions:
        idx = torch.randint(lo_row, hi_row, (B,), generator=pos_gen)  # CPU gen, k-independent
        rows = val[idx]                        # (B,S)
        t = torch.randint(8, S-4, (1,), generator=pos_gen).item() # shared cut point, k-independent
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
        # T2-proper entropies at s=1 (use whichever P3 this mode formed; q* is the soft-label optimum)
        # q* MUST match the KL loop's construction so it is a proper distribution:
        # drop mode -> M=0.5*P2, norm=1.5 ; p3eqp2 -> M=0.5*P2+0.25*P3, norm=1.75.
        if P3 is None:
            M1 = 0.5*P2;              norm1 = 1.5
        else:
            M1 = 0.5*P2 + 0.25*P3;    norm1 = 1.75
        qstar = (P1 + M1) / norm1
        ent_p1_accum.append((-(P1 * P1.clamp_min(1e-12).log()).sum(-1)).cpu().numpy())
        ent_qstar_accum.append((-(qstar * qstar.clamp_min(1e-12).log()).sum(-1)).cpu().numpy())
        done += B
    out = {str(s): float(np.concatenate(kl_accum[s]).mean()) for s in grid}
    out['coverage_mass'] = float(np.concatenate(cover_accum).mean())
    out['entropy_P1'] = float(np.concatenate(ent_p1_accum).mean())
    out['entropy_qstar_s1'] = float(np.concatenate(ent_qstar_accum).mean())
    out['delta_entropy_predicted'] = out['entropy_qstar_s1'] - out['entropy_P1']
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

# ── Per-scale BAND across the (k, offset3-mode, half) robustness grid ──
# Read the mixture prediction against the FULL band, never a single variant. Highest-k rows
# are the least-truncated; the band [min,max] over the grid is the method-uncertainty envelope.
grid = ['0.0', '0.25', '0.5', '1.0']
kmax = max(args.topk_list)
band = {}
for s in grid:
    vals = [e[s] for e in results['kl_estimates']]
    vals_kmax = [e[s] for e in results['kl_estimates'] if e['topk'] == kmax]
    band[s] = {'min': float(np.min(vals)), 'max': float(np.max(vals)),
               'min_kmax': float(np.min(vals_kmax)), 'max_kmax': float(np.max(vals_kmax))}
results['band_per_scale'] = band

# observed sweep gaps (final_val at scale s minus scale-0.0 anchor); filled from the sweep JSONs if present
import glob as _glob
obs_gap = {}
sweep_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_e')
ce0 = None
for f in _glob.glob(os.path.join(sweep_dir, 'sweep_scale*_seed42_*steps.json')):
    d = json.load(open(f)); sc = str(d.get('mtp_scale'))
    if sc == '0.0': ce0 = d.get('final_val_loss')
    obs_gap[sc] = d.get('final_val_loss')
if ce0 is not None:
    obs_gap = {k: (v - ce0) for k, v in obs_gap.items() if v is not None}
    results['observed_sweep_gap'] = obs_gap

# Noise for a single-run gap vs a single-run anchor: sd ~= sqrt(2) * per-run sd (0.015-0.020)
per_run_sd = 0.018
gap_noise = (2 ** 0.5) * per_run_sd   # ~0.025
results['gap_noise_sd'] = gap_noise

# T1 headline: is the observed s=1 gap inside the coverage-band (highest k)?
kl1_kmax = [e['1.0'] for e in results['kl_estimates'] if e['topk'] == kmax]
have_obs = ce0 is not None and obs_gap.get('1.0') is not None
results['T1_KL_s1'] = {'min': float(np.min(kl1_kmax)), 'max': float(np.max(kl1_kmax)),
                       'mean': float(np.mean(kl1_kmax)),
                       'observed': float(obs_gap['1.0']) if have_obs else None}
lo, hi = np.min(kl1_kmax), np.max(kl1_kmax)
if have_obs:
    obs1 = obs_gap['1.0']
    verdict = 'INSIDE band' if lo <= obs1 <= hi else ('ABOVE band' if obs1 > hi else 'BELOW band')
    print(f"\nT1 (k={kmax}): predicted KL(s=1) band [{lo:.3f}, {hi:.3f}]  observed gap {obs1:.3f}  -> {verdict}")
    print(f"    (gap noise sd ~= {gap_noise:.3f}; miss above band top = {max(0.0, obs1-hi):+.3f} = {max(0.0,obs1-hi)/gap_noise:.1f} sd)")
else:
    print(f"\nT1 (k={kmax}): predicted KL(s=1) band [{lo:.3f}, {hi:.3f}]  "
          f"(no sweep JSON found under {sweep_dir} -> observed gap NOT compared here)")

# Pointwise (per-scale) read against the band, at highest k -- ONLY if we have real observed gaps.
if have_obs:
    print("\nPointwise (highest-k band vs observed sweep gap):")
    for s in ['0.25', '0.5', '1.0']:
        o = obs_gap.get(s)
        if o is None:
            print(f"  s={s}: band [{band[s]['min_kmax']:.3f},{band[s]['max_kmax']:.3f}]  (no observed sweep run at this scale)")
            continue
        b = band[s]
        inside = b['min_kmax'] <= o <= b['max_kmax']
        miss = 0.0 if inside else min(abs(o-b['min_kmax']), abs(o-b['max_kmax']))
        print(f"  s={s}: band [{b['min_kmax']:.3f},{b['max_kmax']:.3f}]  obs {o:.3f}  "
              f"{'INSIDE' if inside else f'miss {miss:+.3f} ({miss/gap_noise:.1f} sd)'}")
else:
    print("\nPointwise band-vs-observed test SKIPPED (no sweep JSONs on this machine; "
          "the predicted band is in kl_scan_results.json for offline comparison).")

# coverage->KL plateau check (escalation trigger): does KL(s=1) keep rising with k?
print("\nCoverage->KL(s=1) by k (p3eqp2, half=0), for plateau/escalation decision:")
for topk in sorted(set(args.topk_list)):
    rows = [e for e in results['kl_estimates'] if e['topk']==topk and e['offset3']=='p3eqp2' and e['val_half']==0]
    if rows:
        print(f"  k={topk:5d}: coverage={rows[0]['coverage_mass']:.3f}  KL(s=1)={rows[0]['1.0']:.4f}")

results['predicted_curve_mean'] = {s: float(np.mean([e[s] for e in results['kl_estimates']])) for s in grid}
# linear-through-anchor reference (favors NEITHER hypothesis alone); uses the observed s=1 gap when
# a sweep run is present, else 0.392 (the committed 124M gap) so this never dereferences an unbound name.
_anchor1 = obs_gap['1.0'] if have_obs else 0.392
results['linear_alternative'] = {s: _anchor1*float(s) for s in grid}

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
    # T2-proper predicted entropy lift: mean of the soft-label H(q*)-H(P1) over the highest-k band rows
    # Only the p3eqp2 rows carry the FULL soft-label target (norm 1.75); drop rows are a truncated
    # target and would understate the entropy lift, so exclude them from the T2-proper prediction.
    t2_pred = [e['delta_entropy_predicted'] for e in results['kl_estimates']
               if e['topk'] == kmax and e['offset3'] == 'p3eqp2' and 'delta_entropy_predicted' in e]
    t2_pred_mean = float(np.mean(t2_pred)) if t2_pred else None
    results['T2_T3'] = {'ce_only': ce_e, 'shared_mtp': mtp_e,
                        'delta_entropy_observed': mtp_e['entropy']-ce_e['entropy'],
                        'delta_entropy_predicted_softlabel': t2_pred_mean,
                        'delta_ce_t1': mtp_e['ce_t1']-ce_e['ce_t1'],
                        'delta_ce_t2': mtp_e['ce_t2']-ce_e['ce_t2'],
                        'delta_ce_t3': mtp_e['ce_t3']-ce_e['ce_t3']}
    pred_str = f"{t2_pred_mean:+.3f}" if t2_pred_mean is not None else "n/a"
    print(f"\nT2 entropy: CE-only {ce_e['entropy']:.3f} vs shared-MTP {mtp_e['entropy']:.3f} "
          f"(observed Δ{mtp_e['entropy']-ce_e['entropy']:+.3f}; soft-label q* predicts Δ{pred_str}, "
          f"truncation-biased low)")
    print(f"T3 future CE: shared-MTP better at t+2 by {ce_e['ce_t2']-mtp_e['ce_t2']:+.3f}, "
          f"t+3 by {ce_e['ce_t3']-mtp_e['ce_t3']:+.3f} (mixture predicts shared-MTP BETTER at t+2/t+3)")

# Write to a NEW filename so the original kl_estimate_results.json (cited by committed history and
# the audit trail) is preserved; the k-scan run is self-contained and supersedes it for the paper.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kl_scan_results.json')
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}")
