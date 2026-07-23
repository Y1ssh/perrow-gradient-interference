#!/usr/bin/env python3
"""
run_norm_support.py — per-row norm/support measurement on a short diagnostic pass.

Trains a 124M GPT-2 for 1000 steps on the shared-head variant B objective
(CE + 0.5*MTP2 + 0.25*MTP3), then calls measure_norm_support() at step 1000 to
separate the two candidate explanations for the near-zero / negative aggregate
cosine:

    norm_profile_cos      = cos of the two per-row NORM vectors.
                            LOW (~0)  => CE and MTP load DIFFERENT rows (support divergence)
                            HIGH (~1) => CE and MTP load the SAME rows (support overlaps)
    opposed_norm_fraction = share of ||g_ce||*||g_mtp|| mass on rows where cos<0.
                            LOW  => cancellation channel is small
                            HIGH => a high-norm opposed minority sets the sign (cancellation)

This is the deciding measurement for the paper's mechanism. It faithfully
reproduces the phase_a A1 (Muon) and A2 (AdamW) recipes so the numbers are
comparable to the committed a1/a2 runs.

Usage (one run):
    python measurement/run_norm_support.py --optimizer muon  --seed 42
    python measurement/run_norm_support.py --optimizer adamw --seed 123

Output:
    results/norm_support/norm_support_{optimizer}_seed{seed}.json
    (never overwrites another optimizer/seed; the aggregator reads them all)

Runtime: ~2-3 min/run on an H100 (1000 steps + one measurement pass).
Requires: a CUDA GPU, FineWeb-Edu access (streamed), muon (for --optimizer muon).
"""

import os, sys, math, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2
from measurement.measure_norm_support import measure_norm_support

try:
    from muon import SingleDeviceMuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    HAS_MUON = False

DEVICE = 'cuda'
BATCH_SIZE = 16
SEQ_LEN = 1024
TOTAL_STEPS = 1000

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'norm_support'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

import tiktoken
from datasets import load_dataset


def load_fineweb_tokens(max_tokens=50_000_000):
    cache_path = f'fineweb_train_{max_tokens // 1_000_000}M.pt'
    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Downloading FineWeb-Edu ({max_tokens / 1e6:.0f}M tokens)...")
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    all_tokens = []
    for example in ds:
        toks = enc.encode_ordinary(example['text'])
        all_tokens.extend(toks)
        if len(all_tokens) >= max_tokens:
            break
        if len(all_tokens) % 5_000_000 < len(toks):
            print(f"    {len(all_tokens) / 1e6:.1f}M tokens...")
    all_tokens = all_tokens[:max_tokens]
    t = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(t, cache_path)
    print(f"  Cached {len(all_tokens):,} tokens -> {cache_path}")
    return t


def get_lr(step, base_lr, warmup=100, total_steps=1000):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_ratio)))


def build_optimizer(model, optimizer_name):
    """Reproduce phase_a exactly. Muon = A1 groups; AdamW = A2 dedup split."""
    if optimizer_name == 'muon':
        assert HAS_MUON, "Muon required for --optimizer muon. pip install git+https://github.com/KellerJordan/Muon"
        # Adam base LR 3e-4 (Adam groups scheduled; Muon groups flat).
        return SingleDeviceMuonWithAuxAdam(model.get_muon_param_groups()), 3e-4
    elif optimizer_name == 'adamw':
        # A2: deduplicate tied weights, decay (>=2D) vs nodecay, base LR 1e-3.
        seen = set()
        params_decay, params_nodecay = [], []
        for name, p in model.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            (params_decay if p.dim() >= 2 else params_nodecay).append(p)
        opt = torch.optim.AdamW([
            {'params': params_decay, 'lr': 1e-3, 'weight_decay': 0.01},
            {'params': params_nodecay, 'lr': 1e-3, 'weight_decay': 0.0},
        ], betas=(0.9, 0.95), eps=1e-8)
        return opt, 1e-3
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")


def set_lr(optimizer, optimizer_name, step, base_lr):
    """Muon: schedule only the Adam (non-muon) groups. AdamW: schedule all."""
    lr = get_lr(step, base_lr, warmup=100, total_steps=TOTAL_STEPS)
    for pg in optimizer.param_groups:
        if optimizer_name == 'muon' and pg.get('use_muon', False):
            continue  # Muon groups stay flat
        pg['lr'] = lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("  Norm / Support Measurement")
    print(f"  1000-step {args.optimizer} diagnostic (seed {args.seed}) -> measure_norm_support @ step 1000")
    print("=" * 70)

    # Data (shared across seeds; the seed drives model init + batch sampling)
    print("\n  Loading data...")
    tokens = load_fineweb_tokens(max_tokens=50_000_000)
    train_tokens = tokens[:48_000_000]
    val_tokens = tokens[48_000_000:]
    train_data = train_tokens[:len(train_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
    val_data = val_tokens[:len(val_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
    del tokens, train_tokens, val_tokens
    print(f"  Train: {train_data.shape[0]:,} seqs | Val: {val_data.shape[0]:,} seqs")

    # Model (seed drives init AND the batch-sampling RNG below)
    torch.manual_seed(args.seed)
    model = GPT2().to(DEVICE)
    V = model.config.vocab_size
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer, base_lr = build_optimizer(model, args.optimizer)

    print(f"\n  Training {TOTAL_STEPS} steps (variant B, {args.optimizer}, base_lr={base_lr})...")
    t0 = time.time()
    model.train()
    for step in range(1, TOTAL_STEPS + 1):
        set_lr(optimizer, args.optimizer, step, base_lr)
        batch = train_data[torch.randint(0, train_data.shape[0], (BATCH_SIZE,))]
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch)
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
            mtp2 = F.cross_entropy(logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
            mtp3 = F.cross_entropy(logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
            loss = ce + 0.5 * mtp2 + 0.25 * mtp3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == 1:
            print(f"    Step {step:>4}/{TOTAL_STEPS} | loss={loss.item():.4f} | {time.time()-t0:.0f}s")

    # ── the measurement ──
    print(f"\n  Running measure_norm_support at step {TOTAL_STEPS}...")
    val_batch = val_data[torch.randint(0, val_data.shape[0], (BATCH_SIZE,))]
    result = measure_norm_support(model, val_batch)

    npc = result['norm_profile_cos']
    onf = result['opposed_norm_fraction']
    gc = result['global_cos']
    pr03 = result['per_row_fractions'][0.3]

    print("\n  " + "=" * 50)
    print("  RESULTS - Support Divergence vs Opposed-Mass Test")
    print("  " + "=" * 50)
    print(f"  global_cos (aggregate):     {gc:+.4f}")
    print(f"  per_row |cos|>0.3:          {pr03:.1%}")
    print(f"  norm_profile_cos:           {npc:.4f}")
    print(f"  opposed_norm_fraction:      {onf:.4f}")
    print("\n  INTERPRETATION:")
    if npc < 0.3 and onf < 0.1:
        print(f"  -> SUPPORT DIVERGENCE (norm_profile_cos={npc:.3f} low, opposed={onf:.3f} low)")
    elif onf > 0.3:
        print(f"  -> CANCELLATION (high-norm opposed rows dominate; opposed={onf:.3f})")
    else:
        print(f"  -> MIXED (norm_profile_cos={npc:.3f}, opposed={onf:.3f})")
    print("  " + "=" * 50)

    save_data = {
        'step': TOTAL_STEPS,
        'seed': args.seed,
        'optimizer': args.optimizer,
        'variant': 'b',
        'model_params': sum(p.numel() for p in model.parameters()),
        'global_cos': gc,
        'per_row_fractions': result['per_row_fractions'],
        'norm_profile_cos': npc,
        'opposed_norm_fraction': onf,
        'ce_row_norms': result['ce_row_norms'].tolist(),
        'mtp_row_norms': result['mtp_row_norms'].tolist(),
        'row_cosines': result['row_cosines'].tolist(),
        'val_loss': result['val_loss'],
        'total_time': round(time.time() - t0, 1),
    }
    outpath = os.path.join(RESULTS_DIR, f'norm_support_{args.optimizer}_seed{args.seed}.json')
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved -> {outpath}")
    print(f"  (results saved LOCALLY; commit/push from your clone, or scp the file back)")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
    return save_data


if __name__ == '__main__':
    main()
