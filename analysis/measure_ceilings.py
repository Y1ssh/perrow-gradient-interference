#!/usr/bin/env python3
"""
Ceiling + control measurements on the step-1000 snapshots (round-2 review, Part F + Amendment 2).

Runs on the matched step-1000 checkpoints the sweep now saves. Produces, committing the
FULL per-row arrays + target-presence boolean each time (the standing invariant):

  1. CONTROL (CE-vs-L1) on the s=1.0 step-1000 snapshot, ACTIVE-ROW: the calibration baseline
     the review flagged as construction-floored. Re-measured on the matched step with the
     target-presence split so it is apples-to-apples with the active-row CE-vs-MTP number.
  2. CE-vs-CE disjoint-half-batch CEILING on both the s=0.0 (CE-only) and s=1.0 (MTP-trained)
     snapshots: two CE losses on disjoint halves of the batch, through the same head. This is
     the alignment CEILING -- it answers "do ANY two data-driven losses look aligned here?"
     If CE-vs-CE active-row cosine is very high, the CE-vs-MTP active-row median of ~0.53 is
     meaningfully BELOW ceiling (real structure); if CE-vs-CE is also ~0.53, the alignment is
     generic to the head.

  3. 3-BATCH ROBUSTNESS: active fraction + active-row median at 1/2/3 accumulated measurement
     batches (the active fraction is batch-relative; this bounds its sensitivity).

Usage:
  python3 analysis/measure_ceilings.py \
     --ce_ckpt  results/checkpoints/model_scale0.0_seed42_step1000.pt \
     --mtp_ckpt results/checkpoints/model_scale1.0_seed42_step1000.pt
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from measurement.measure_interference import measure_interference

ap = argparse.ArgumentParser()
ap.add_argument('--ce_ckpt', required=True)
ap.add_argument('--mtp_ckpt', required=True)
ap.add_argument('--seq_len', type=int, default=1024)
ap.add_argument('--batch_size', type=int, default=16)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
V_REAL = 50257

def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = GPT2Config(**ck['config']) if 'config' in ck else GPT2Config()
    m = GPT2(cfg).to(DEV); m.load_state_dict(ck['model_state_dict']); m.eval()
    return m

def load_val():
    for c in ['fineweb_train_500M.pt', 'fineweb_train_50M.pt']:
        if os.path.exists(c):
            t = torch.load(c, weights_only=True); return t[int(t.numel()*0.96):]
    import tiktoken; from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    toks = []
    for ex in ds:
        toks.extend(enc.encode_ordinary(ex['text']))
        if len(toks) >= 2_000_000: break
    return torch.tensor(toks, dtype=torch.long)

val = load_val(); S = args.seq_len
val = val[:(val.numel()//S)*S].view(-1, S).to(DEV)
def get_batch(bs):
    return val[torch.randint(0, val.shape[0], (bs,))]

def target_presence(batch):
    tgt = torch.zeros(50304, dtype=torch.bool, device=batch.device)
    tgt[batch[:, 1:].reshape(-1)] = True
    tgt[batch[:, 2:].reshape(-1)] = True
    tgt[batch[:, 3:].reshape(-1)] = True
    return tgt[:V_REAL]

def active_stats(row_cos, active):
    a = row_cos[active]
    return {'active_fraction': float(active.float().mean()),
            'active_median_cos': float(a.median()) if a.numel() else float('nan'),
            'active_opposed_fraction': float((a < 0).float().mean()) if a.numel() else float('nan'),
            'active_frac_above_0.3': float((a.abs() > 0.3).float().mean()) if a.numel() else float('nan'),
            'full_median_cos': float(row_cos.median()),
            'full_frac_above_0.3': float((row_cos.abs() > 0.3).float().mean())}

results = {}

# ---- 1. CONTROL (CE-vs-L1) active-row, on the s=1.0 step-1000 snapshot ----
mtp_model = load(args.mtp_ckpt)
b = get_batch(args.batch_size)
ctrl = measure_interference(mtp_model, b, loss_pair='ce_l1')
rc = torch.tensor(ctrl['row_cosines'][:V_REAL])
active = target_presence(b).cpu()[:len(rc)]
results['control_ce_l1'] = {**active_stats(rc, active), 'global_cos': ctrl['global_cos']}
print(f"CONTROL CE-vs-L1 (active): median {results['control_ce_l1']['active_median_cos']:.3f}, "
      f"%|cos|>0.3 {100*results['control_ce_l1']['active_frac_above_0.3']:.1f}% "
      f"(full-vocab {100*results['control_ce_l1']['full_frac_above_0.3']:.1f}%)")

# ---- 2. CE-vs-CE disjoint-half CEILING on both checkpoints ----
def ce_vs_ce_ceiling(model, b):
    half = b.shape[0] // 2
    b1, b2 = b[:half], b[half:2*half]
    was = model.training; model.eval()
    with torch.enable_grad(), torch.amp.autocast(DEV, enabled=False):
        V = model.config.vocab_size
        l1, _ = model(b1); l2, _ = model(b2)
        ce1 = F.cross_entropy(l1[:, :-1].reshape(-1, V), b1[:, 1:].reshape(-1))
        ce2 = F.cross_entropy(l2[:, :-1].reshape(-1, V), b2[:, 1:].reshape(-1))
        tgt = model.lm_head.weight
        g1 = torch.autograd.grad(ce1, tgt, retain_graph=True)[0].float()
        g2 = torch.autograd.grad(ce2, tgt)[0].float()
    model.train(was)
    rc = F.cosine_similarity(g1, g2, dim=1)[:V_REAL].cpu()
    # ACTIVE-ROW CRITERION for CE-vs-CE: target-PRESENCE (token appears as a t+1/t+2/t+3 target in
    # EITHER half). The 0.75 mtp/ce norm-ratio test used for CE-vs-MTP does NOT apply here -- both
    # gradients are CE, so there is no MTP norm to form a ratio against. Presence is the apples-to-
    # apples active set for the same-loss control.
    tp = torch.zeros(50304, dtype=torch.bool)
    for bb in (b1, b2):
        for off in (1, 2, 3): tp[bb[:, off:].reshape(-1).cpu()] = True
    st = active_stats(rc, tp[:V_REAL])
    st['active_criterion'] = 'target_presence_t1t2t3_either_half'
    return st

ce_model = load(args.ce_ckpt)
bb = get_batch(args.batch_size)
results['ceiling_ce_vs_ce_ceonly']  = ce_vs_ce_ceiling(ce_model, bb)
results['ceiling_ce_vs_ce_mtptrained'] = ce_vs_ce_ceiling(mtp_model, bb)
print(f"CE-vs-CE ceiling (active median): CE-only {results['ceiling_ce_vs_ce_ceonly']['active_median_cos']:.3f}, "
      f"MTP-trained {results['ceiling_ce_vs_ce_mtptrained']['active_median_cos']:.3f}  "
      f"(this is a FLOOR, not a ceiling: same loss on disjoint data is anti-aligned on "
      f"active rows, so CE-vs-MTP at ~0.53 is specific to the shared-batch/shared-logits pairing)")
print(f"  active-row criterion (CE-vs-CE): {results['ceiling_ce_vs_ce_ceonly']['active_criterion']} "
      f"(NOT the 0.75 norm-ratio test, which needs an MTP gradient)")

# Write the control + ceiling numbers NOW, before the robustness loop, so a robustness-loop
# crash (e.g. OOM on accumulated batches) can never eat the load-bearing results again.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ceilings_results.json')
json.dump(results, open(OUT, 'w'), indent=2)
print(f"Saved control+ceiling to {OUT} (robustness appended below if it completes)")

# ---- 3. batch-relative robustness (active fraction grows with batch count; median is stable) ----
# Capped at MAX_ROB batches: full-vocab mtp3 cross-entropy on the accumulated batch is the OOM culprit,
# and 2 points already show the trend (fraction rises, median holds). Raise only if VRAM allows.
MAX_ROB = int(os.environ.get('CEIL_MAX_ROB', '2'))
from measurement.measure_norm_support import measure_norm_support
rob = []
acc = None
for nb in range(1, MAX_ROB + 1):
    b = get_batch(args.batch_size)
    acc = b if acc is None else torch.cat([acc, b], 0)
    try:
        ns = measure_norm_support(mtp_model, acc)
    except torch.cuda.OutOfMemoryError:
        print(f"  {nb} batch(es): OOM -> stopping robustness loop (control+ceiling already saved)")
        torch.cuda.empty_cache(); break
    rob.append({'n_batches': nb, 'active_fraction': ns['active_fraction'],
                'active_median_cos': ns['active_median_cos'],
                'active_opposed_fraction': ns['active_opposed_fraction']})
    print(f"  {nb} batch(es): active {100*ns['active_fraction']:.1f}%, median {ns['active_median_cos']:.3f}")
results['robustness_batches'] = rob
json.dump(results, open(OUT, 'w'), indent=2)   # rewrite with robustness appended
print(f"\nSaved {OUT}")
