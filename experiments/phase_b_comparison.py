"""
Phase B: Core comparison — 5 variants × 5 seeds × 500M tokens.

The MAIN experiment of the paper. Tests whether architectural separation
(G_nce) eliminates per-row gradient interference at 124M scale.

Variants:
    a       CE only (baseline)
    b       CE + shared MTP (causes 98% interference)
    b_sg    CE + stop-grad MTP (lm_head detached for MTP — diagnostic)
    gnce    CE + NCE on intermediate layers (our fix)
    nextlat CE + latent prediction baseline

Usage (one run at a time — crash-safe, spot-instance friendly):
    python3 experiments/phase_b_comparison.py --variant gnce --seed 42
    python3 experiments/phase_b_comparison.py --variant a --seed 123

Run all 25 experiments:
    for variant in a b b_sg gnce nextlat; do
      for seed in 42 123 456 789 1337; do
        python3 experiments/phase_b_comparison.py --variant $variant --seed $seed
      done
    done

Or one variant at a time (for spot instances):
    for seed in 42 123 456 789 1337; do
      python3 experiments/phase_b_comparison.py --variant a --seed $seed
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
from model.auxiliary_losses import GNCELoss, NextLatLoss
from measurement.measure_interference import quick_measure

from muon import SingleDeviceMuonWithAuxAdam

# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE
# ═══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Phase B: core comparison')
parser.add_argument('--variant', required=True,
                    choices=['a', 'b', 'b_sg', 'gnce', 'nextlat'])
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--tokens', type=int, default=500_000_000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=500)
parser.add_argument('--measure_every', type=int, default=2000)
args = parser.parse_args()

VALID_SEEDS = [42, 123, 456, 789, 1337]
DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_b'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.tokens // tokens_per_step

print("=" * 70)
print(f"  Phase B: variant={args.variant}  seed={args.seed}")
print(f"  {args.tokens / 1e6:.0f}M tokens  |  {total_steps:,} steps  |  "
      f"batch={args.batch_size}×{args.seq_len}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
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


def get_lr(step, base_lr, warmup=200, total_steps=30517):
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

def train_variant(variant, seed):
    """
    Train GPT-2 124M with one variant/seed combination.

    Returns results dict with eval_curve, measurements, timing.
    """
    torch.manual_seed(seed)
    model = GPT2().to(DEVICE)
    V = model.config.vocab_size

    # --- Auxiliary module (if any) ---
    aux_module = None
    if variant == 'gnce':
        aux_module = GNCELoss(n_embd=768, layers=(9, 8)).to(DEVICE)
    elif variant == 'nextlat':
        aux_module = NextLatLoss(n_embd=768, pred_layer=11).to(DEVICE)

    # --- Optimizer ---
    param_groups = model.get_muon_param_groups()
    if aux_module is not None:
        param_groups = param_groups + aux_module.get_aux_param_groups()
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    n_groups = len(param_groups)
    n_params = sum(p.numel() for g in param_groups for p in g['params'])
    print(f"\n  Optimizer: {n_groups} param groups, {n_params:,} total params")

    # --- Results dict ---
    results = {
        'variant': variant,
        'seed': seed,
        'total_steps': total_steps,
        'total_tokens': total_steps * tokens_per_step,
        'eval_curve': [],
        'measurements': [],
    }

    t0 = time.time()
    tokens_processed = 0

    model.train()

    for step in range(1, total_steps + 1):

        # ── LR schedule (Adam groups only, Muon stays flat) ──
        new_adam_lr = get_lr(step, 3e-4, warmup=200, total_steps=total_steps)
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

            if variant == 'a':
                total_loss = ce_loss
                aux_info = {'ce_loss': ce_loss.item()}

            elif variant == 'b':
                mtp2 = F.cross_entropy(
                    logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
                mtp3 = F.cross_entropy(
                    logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
                total_loss = ce_loss + 0.5 * mtp2 + 0.25 * mtp3
                aux_info = {'ce_loss': ce_loss.item(),
                            'mtp2': mtp2.item(), 'mtp3': mtp3.item()}

            elif variant == 'b_sg':
                # Stop-grad on lm_head for MTP: manual matmul with detached weight.
                # h_norm values are IDENTICAL to what logits uses (same rms_norm),
                # but lm_head.weight is detached so MTP gradient trains
                # transformer blocks without touching lm_head.
                h_norm = F.rms_norm(intermediates[11], (model.config.n_embd,))
                logits_sg = h_norm @ model.lm_head.weight.detach().T
                mtp2 = F.cross_entropy(
                    logits_sg[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
                mtp3 = F.cross_entropy(
                    logits_sg[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
                total_loss = ce_loss + 0.5 * mtp2 + 0.25 * mtp3
                aux_info = {'ce_loss': ce_loss.item(),
                            'mtp2_sg': mtp2.item(), 'mtp3_sg': mtp3.item()}

            elif variant == 'gnce':
                total_loss, aux_info = aux_module(
                    ce_loss, intermediates, batch, model.wte)

            elif variant == 'nextlat':
                total_loss, aux_info = aux_module(
                    ce_loss, intermediates, batch, model.wte)

        total_loss.backward()
        clip_params = list(model.parameters())
        if aux_module is not None:
            clip_params += list(aux_module.parameters())
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
                  f"ce={aux_info.get('ce_loss', 0):.4f} | "
                  f"{tokens_processed / 1e6:.0f}M tok | "
                  f"{elapsed:.0f}s")

        # ── Interference measurement (seed 42 only) ──
        if seed == 42 and (step == 1 or step % args.measure_every == 0):
            val_batch = get_batch(val_data)
            m = quick_measure(model, val_batch)
            results['measurements'].append({
                'step': step,
                'global_cos': m['global_cos'],
                'per_row_0.3': m['per_row_0.3'],
                'discrepancy': m['discrepancy'],
            })

        # ── Periodic checkpoint ──
        if step % 5000 == 0:
            try:
                save_json(results, os.path.join(
                    RESULTS_DIR, f'_partial_{variant}_seed{seed}.json'))
            except Exception:
                pass

        elif step % 1000 == 0 and not should_eval:
            print('.', end='', flush=True)

    total_time = time.time() - t0
    final_val = eval_model(model, eval_iters=50)  # more accurate final eval

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

results = {'variant': args.variant, 'seed': args.seed, 'crashed': False}

try:
    results = train_variant(args.variant, args.seed)
except Exception as e:
    print(f"\n  *** CRASHED: {e}")
    import traceback; traceback.print_exc()
    results = {
        'variant': args.variant,
        'seed': args.seed,
        'crashed': True,
        'error': str(e),
    }
finally:
    outfile = f'{args.variant}_seed{args.seed}.json'
    save_json(results, os.path.join(RESULTS_DIR, outfile))

# Clean up partial file
partial = os.path.join(RESULTS_DIR, f'_partial_{args.variant}_seed{args.seed}.json')
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
                    f'Phase B: {args.variant} seed={args.seed}'],
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
print(f"  Phase B: {args.variant} seed={args.seed}")
print(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
print(f"  Total time: {results.get('total_time', 0):.0f}s")
tok_sec = results.get('final_tok_per_sec', 0)
print(f"  Speed: {tok_sec / 1000:.0f}K tok/s" if tok_sec else "  Speed: N/A")
if results.get('crashed'):
    print(f"  STATUS: *** CRASHED ***")
else:
    print(f"  STATUS: Complete ✓")
print(f"{'=' * 60}")
