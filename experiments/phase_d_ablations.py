"""
Phase D: G_nce ablation experiments.

Screening runs (5000 steps default) to identify best hyperparameters
before committing to full 30517-step Phase B runs.

Ablation dimensions:
    D1: alpha_mult   — 0.1, 0.3 (default), 1.0, 3.0
    D2: layers       — (8,7), (9,8) default, (10,9)
    D3: K negatives  — 1, 4, 16 (default)
    D4: neg_type     — roll (default), random
    D5: nce_weights  — (0.25,0.125), (0.5,0.25) default, (1.0,0.5)
    D6: tau          — 0.05, 0.1 (default), 0.2, 0.5

Usage:
    python3 experiments/phase_d_ablations.py --alpha_mult 1.0 --seed 42
    python3 experiments/phase_d_ablations.py --layers 8 7 --K 4 --seed 42
    python3 experiments/phase_d_ablations.py --neg_type random --seed 42 --steps 30517

Full ablation sweep:
    # D1: Alpha sensitivity
    for a in 0.1 0.3 1.0 3.0; do
      python3 experiments/phase_d_ablations.py --alpha_mult $a --seed 42
    done

    # D2: Layer split
    for l in "8 7" "9 8" "10 9"; do
      python3 experiments/phase_d_ablations.py --layers $l --seed 42
    done

    # D3: K negatives
    for k in 1 4 16; do
      python3 experiments/phase_d_ablations.py --K $k --seed 42
    done

    # D4: Roll vs random
    for n in roll random; do
      python3 experiments/phase_d_ablations.py --neg_type $n --seed 42
    done

    # D5: NCE weights
    for w in "0.25 0.125" "0.5 0.25" "1.0 0.5"; do
      python3 experiments/phase_d_ablations.py --nce_weights $w --seed 42
    done

    # D6: Tau sensitivity
    for t in 0.05 0.1 0.2 0.5; do
      python3 experiments/phase_d_ablations.py --tau $t --seed 42
    done
"""

import os, sys, math, time, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from model.auxiliary_losses_ablation import GNCELossAblation

from muon import SingleDeviceMuonWithAuxAdam


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE
# ═══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Phase D: G_nce ablation')
parser.add_argument('--alpha_mult', type=float, default=0.3)
parser.add_argument('--layers', type=int, nargs=2, default=[9, 8])
parser.add_argument('--K', type=int, default=16)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--nce_weights', type=float, nargs=2, default=[0.5, 0.25])
parser.add_argument('--neg_type', type=str, default='roll', choices=['roll', 'random'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=500)
parser.add_argument('--lr_schedule_steps', type=int, default=30517,
                    help='LR cosine decay reference. 30517=Phase B schedule for screening.')
args = parser.parse_args()

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_d'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.steps

# Descriptive output filename
outfile = (f"ablation_alpha{args.alpha_mult}_layers{args.layers[0]}-{args.layers[1]}"
           f"_K{args.K}_tau{args.tau}_{args.neg_type}"
           f"_w{args.nce_weights[0]}-{args.nce_weights[1]}"
           f"_seed{args.seed}_lrs{args.lr_schedule_steps}_{args.steps}steps.json")

print("=" * 70)
print(f"  Phase D ablation: seed={args.seed}  steps={total_steps}")
print(f"  alpha_mult={args.alpha_mult}  layers={args.layers}  K={args.K}  tau={args.tau}")
print(f"  neg_type={args.neg_type}  nce_weights={args.nce_weights}")
print(f"  → {outfile}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (same as Phase B — 500M FineWeb-Edu, 480M/20M split)
# ═══════════════════════════════════════════════════════════════════════════

import tiktoken
from datasets import load_dataset


def load_fineweb_tokens(max_tokens=500_000_000):
    """Download FineWeb-Edu, tokenize with GPT-2 tokenizer, cache to disk."""
    cache_path = f'fineweb_train_{max_tokens // 1_000_000}M.pt'
    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Downloading FineWeb-Edu and tokenizing ({max_tokens / 1e6:.0f}M tokens)...")
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
    print(f"  Cached {len(all_tokens):,} tokens to {cache_path}")
    return t


print("\n  Loading data...")
tokens = load_fineweb_tokens(max_tokens=500_000_000)

train_tokens = tokens[:480_000_000]
val_tokens = tokens[480_000_000:]

SEQ_LEN = args.seq_len
train_data = train_tokens[: len(train_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
val_data = val_tokens[: len(val_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)

print(f"  Train: {train_data.shape[0]} sequences × {SEQ_LEN}")
print(f"  Val:   {val_data.shape[0]} sequences × {SEQ_LEN}")


def get_batch(data, batch_size=None):
    """Random sampling (not sequential)."""
    if batch_size is None:
        batch_size = args.batch_size
    idx = torch.randint(0, data.shape[0], (batch_size,))
    return data[idx]


# ═══════════════════════════════════════════════════════════════════════════
# EVAL + LR SCHEDULE + JSON HELPER
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_model(model, eval_iters=20):
    """Average val CE loss over eval_iters random batches."""
    model.eval()
    V = model.config.vocab_size
    losses = []
    for _ in range(eval_iters):
        batch = get_batch(val_data)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def get_lr(step, base_lr, warmup=200, total_steps=5000):
    """Linear warmup + cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_ratio)))


def save_json(obj, path):
    """Save to JSON with numpy/float-key handling."""
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, dict):
            return {str(k) if isinstance(k, float) else k: convert(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(x) for x in o]
        return o
    with open(path, 'w') as f:
        json.dump(convert(obj), f, indent=2)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train_ablation():
    """
    Train GPT-2 124M with one GNCELossAblation configuration.

    Returns results dict with eval_curve and timing.
    """
    torch.manual_seed(args.seed)
    model = GPT2().to(DEVICE)
    V = model.config.vocab_size

    # Auxiliary module with ablation config
    aux_module = GNCELossAblation(
        n_embd=768,
        layers=tuple(args.layers),
        K=args.K,
        tau=args.tau,
        alpha_mult=args.alpha_mult,
        nce_weights=tuple(args.nce_weights),
        neg_type=args.neg_type,
    ).to(DEVICE)

    # Optimizer
    param_groups = model.get_muon_param_groups() + aux_module.get_aux_param_groups()
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    n_params = sum(p.numel() for g in param_groups for p in g['params'])
    print(f"\n  Optimizer: {len(param_groups)} param groups, {n_params:,} total params")

    # Results dict
    results = {
        'variant': 'gnce_ablation',
        'seed': args.seed,
        'total_steps': total_steps,
        'total_tokens': total_steps * tokens_per_step,
        'ablation_config': {
            'alpha_mult': args.alpha_mult,
            'layers': list(args.layers),
            'K': args.K,
            'tau': args.tau,
            'nce_weights': list(args.nce_weights),
            'neg_type': args.neg_type,
            'lr_schedule_steps': args.lr_schedule_steps,
        },
        'eval_curve': [],
    }

    t0 = time.time()
    tokens_processed = 0

    model.train()

    for step in range(1, total_steps + 1):

        # ── LR schedule (Adam groups only, Muon stays flat) ──
        new_adam_lr = get_lr(step, 3e-4, warmup=200, total_steps=args.lr_schedule_steps)
        for pg in optimizer.param_groups:
            if not pg.get('use_muon', False):
                pg['lr'] = new_adam_lr

        # ── Training step ──
        batch = get_batch(train_data)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, intermediates = model(batch)

            ce_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))

            total_loss, aux_info = aux_module(
                ce_loss, intermediates, batch, model.wte)

        total_loss.backward()
        clip_params = list(model.parameters()) + list(aux_module.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()
        tokens_processed += tokens_per_step

        # ── Evaluation ──
        should_eval = (step == 1 or step % args.eval_every == 0
                       or step == total_steps)

        if should_eval:
            val_loss = eval_model(model, eval_iters=20)

            eval_entry = {
                'step': step,
                'val_loss': val_loss,
                'tokens': tokens_processed,
                'elapsed': round(time.time() - t0, 1),
                **aux_info,
            }
            results['eval_curve'].append(eval_entry)

            elapsed = time.time() - t0
            print(f"  Step {step:>6}/{total_steps} | "
                  f"val={val_loss:.4f} | "
                  f"ce={aux_info['ce_loss']:.4f} | "
                  f"nce={aux_info['nce_loss']:.4f} | "
                  f"alpha={aux_info['alpha']:.3f} | "
                  f"{elapsed:.0f}s")

        # ── Periodic checkpoint ──
        if step % 2000 == 0:
            try:
                save_json(results, os.path.join(
                    RESULTS_DIR, f'_partial_{outfile}'))
            except Exception:
                pass

        elif step % 1000 == 0 and not should_eval:
            print('.', end='', flush=True)

    total_time = time.time() - t0
    final_val = eval_model(model, eval_iters=50)

    results['total_time'] = round(total_time, 1)
    results['final_val_loss'] = final_val
    results['final_tok_per_sec'] = round(tokens_processed / total_time)

    print(f"\n  Done. {total_time:.0f}s | "
          f"final_val={final_val:.4f} | "
          f"{tokens_processed / total_time / 1000:.0f}K tok/s")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

results = {'variant': 'gnce_ablation', 'seed': args.seed, 'crashed': False}

try:
    results = train_ablation()
except Exception as e:
    print(f"\n  *** CRASHED: {e}")
    import traceback; traceback.print_exc()
    results = {
        'variant': 'gnce_ablation',
        'seed': args.seed,
        'crashed': True,
        'error': str(e),
        'ablation_config': {
            'alpha_mult': args.alpha_mult,
            'layers': list(args.layers),
            'K': args.K,
            'tau': args.tau,
            'nce_weights': list(args.nce_weights),
            'neg_type': args.neg_type,
            'lr_schedule_steps': args.lr_schedule_steps,
        },
    }
finally:
    save_json(results, os.path.join(RESULTS_DIR, outfile))

# Clean up partial file
partial = os.path.join(RESULTS_DIR, f'_partial_{outfile}')
if os.path.exists(partial):
    os.remove(partial)
    print(f"  Cleaned up {partial}")

# Git push (best-effort)
import subprocess
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    subprocess.run(['git', 'add', 'results/'], cwd=repo_root,
                   check=True, capture_output=True)
    subprocess.run(['git', 'commit', '-m',
                    f'Phase D ablation: {outfile}'],
                   cwd=repo_root, check=True, capture_output=True)
    subprocess.run(['git', 'push'], cwd=repo_root,
                   check=True, capture_output=True)
    print("  Results pushed to GitHub ✓")
except Exception:
    print(f"  Git push failed — save results manually from {RESULTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print(f"  Phase D ablation: seed={args.seed}  steps={total_steps}")
print(f"  Config: alpha={args.alpha_mult} layers={args.layers} K={args.K} "
      f"tau={args.tau} {args.neg_type} w={args.nce_weights}")
print(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
print(f"  Total time: {results.get('total_time', 0):.0f}s")
tok_sec = results.get('final_tok_per_sec', 0)
print(f"  Speed: {tok_sec / 1000:.0f}K tok/s" if tok_sec else "  Speed: N/A")
if results.get('crashed'):
    print(f"  STATUS: *** CRASHED ***")
else:
    print(f"  STATUS: Complete ✓")
print(f"{'=' * 60}")
