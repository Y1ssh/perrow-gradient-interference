"""
Phase C: 350M scale measurement — R≈3.86 matched to Phase B.

Trains GPT-2 Medium (354M) with shared-head MTP (variant B) for 87,000 steps.
Measures per-row interference at lm_head every 4000 steps via two-pass backward.

Key design: R-value matches Phase B exactly.
    Phase B (124M): 480M train tokens / 124.4M params = R=3.86
    This (350M):    1,369M train tokens / 354.5M params = R=3.86

Data split (Option B — Phase B val-set matched):
    Download:  1.5B tokens from FineWeb-Edu
    Val:       positions 480M-500M (20M tokens, IDENTICAL to Phase B val)
    Train:     cat([0:480M], [500M:1389M]) = 1,369M unique tokens
    No data cycling: ~1.04 epochs (matches Phase B sampling ratio)

Usage:
    python3 experiments/phase_c_350m_r4.py --seed 42
"""

import os, sys, math, time, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2_medium import GPT2Medium
from muon import SingleDeviceMuonWithAuxAdam

parser = argparse.ArgumentParser(description='Phase C: 350M R≈3.86 matched')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--steps', type=int, default=87000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=2000)
parser.add_argument('--measure_every', type=int, default=4000)
args = parser.parse_args()

DEVICE = 'cuda'
MODEL_PARAMS = 354_549_760
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'phase_c_350m_r4'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.steps
outfile = f'scale_350m_r4_b_seed{args.seed}_{args.steps}steps.json'


# ═══════════════════════════════════════════════════════════════════════════
# DATA — 1.5B tokens, Phase B val-matched split
# ═══════════════════════════════════════════════════════════════════════════

import tiktoken
from datasets import load_dataset


def load_fineweb_tokens(max_tokens=1_500_000_000):
    """Download FineWeb-Edu, tokenize with GPT-2 tokenizer, cache to disk."""
    cache_path = f'fineweb_train_{max_tokens // 1_000_000}M.pt'
    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Downloading FineWeb-Edu and tokenizing ({max_tokens / 1e6:.0f}M tokens)...")
    print(f"  This will take 15-30 minutes...")
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


print("\n  Loading data (1.5B tokens)...")
all_tokens = load_fineweb_tokens(max_tokens=1_500_000_000)

# Option B split: val=480M-500M (Phase B matched), train=cat([0:480M],[500M:1389M])
val_tokens = all_tokens[480_000_000:500_000_000]
train_tokens = torch.cat([
    all_tokens[:480_000_000],
    all_tokens[500_000_000:1_389_000_000],
])
del all_tokens  # free ~12GB RAM
gc.collect()

SEQ_LEN = args.seq_len
train_data = train_tokens[: len(train_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
val_data = val_tokens[: len(val_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
unique_train = train_tokens.numel()
del train_tokens, val_tokens
gc.collect()

total_tokens_processed = total_steps * tokens_per_step
data_epochs = total_tokens_processed / unique_train
R_value = unique_train / MODEL_PARAMS

print(f"  Train: {train_data.shape[0]:,} sequences ({unique_train:,} tokens)")
print(f"  Val:   {val_data.shape[0]:,} sequences (20,000,000 tokens)")
print(f"  R-value: {R_value:.2f} tok/param")
print(f"  Data epochs: {data_epochs:.2f}")

print("=" * 70)
print(f"  Phase C: 350M scale measurement (R≈{R_value:.2f} matched to Phase B)")
print(f"  seed={args.seed}  steps={total_steps}")
print(f"  batch={args.batch_size}×{args.seq_len} = {tokens_per_step:,} tok/step")
print(f"  Total tokens processed: {total_tokens_processed:,}")
print(f"  Unique train tokens: {unique_train:,}")
print(f"  Val set: positions 480M-500M (Phase B matched)")
print("=" * 70)


def get_batch(data):
    return data[torch.randint(0, data.shape[0], (args.batch_size,))]


# ═══════════════════════════════════════════════════════════════════════════
# EVAL + LR + JSON
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_model(model, eval_iters=20):
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


def get_lr(step, base_lr, warmup=200, total_steps=87000):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_ratio)))


def save_json(obj, path):
    def convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, dict):
            return {str(k) if isinstance(k, float) else k: convert(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [convert(x) for x in o]
        return o
    with open(path, 'w') as f:
        json.dump(convert(obj), f, indent=2)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# INTERFERENCE MEASUREMENT (two-pass at lm_head)
# ═══════════════════════════════════════════════════════════════════════════

def measure_interference_twopass(model, batch):
    """Measure per-row interference via two-pass backward at lm_head."""
    model.eval()
    V = model.config.vocab_size

    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
        logits, _ = model(batch)

        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        mtp2 = F.cross_entropy(
            logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
        mtp3 = F.cross_entropy(
            logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
        mtp_loss = 0.5 * mtp2 + 0.25 * mtp3

        target = model.lm_head.weight
        ce_grad = torch.autograd.grad(ce_loss, target, retain_graph=True)[0].float()
        mtp_grad = torch.autograd.grad(mtp_loss, target)[0].float()

    dots = (ce_grad * mtp_grad).sum(dim=1)
    ce_n = ce_grad.norm(dim=1).clamp(min=1e-8)
    mtp_n = mtp_grad.norm(dim=1).clamp(min=1e-8)
    cos = dots / (ce_n * mtp_n)

    global_cos = F.cosine_similarity(
        ce_grad.reshape(1, -1), mtp_grad.reshape(1, -1)).item()
    interference = (cos < 0).float().mean().item()
    per_row_03 = (cos.abs() > 0.3).float().mean().item()
    discrepancy = per_row_03 / max(abs(global_cos), 0.001)

    model.train()
    return {
        'global_cos': global_cos,
        'interference_rate': interference,
        'per_row_0.3': per_row_03,
        'discrepancy': discrepancy,
        'val_loss': ce_loss.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_350m():
    torch.manual_seed(args.seed)
    model = GPT2Medium().to(DEVICE)
    V = model.config.vocab_size

    optimizer = SingleDeviceMuonWithAuxAdam(model.get_muon_param_groups())

    results = {
        'variant': '350m_r4_b',
        'seed': args.seed,
        'total_steps': total_steps,
        'model_params': MODEL_PARAMS,
        'total_tokens_processed': total_tokens_processed,
        'unique_train_tokens': unique_train,
        'data_epochs': round(data_epochs, 2),
        'R_value': round(R_value, 2),
        'val_split': '480M-500M (Phase B matched)',
        'eval_curve': [],
        'measurements': [],
    }

    t0 = time.time()
    tokens_processed = 0
    model.train()

    for step in range(1, total_steps + 1):
        new_adam_lr = get_lr(step, 3e-4, warmup=200, total_steps=total_steps)
        for pg in optimizer.param_groups:
            if not pg.get('use_muon', False):
                pg['lr'] = new_adam_lr

        batch = get_batch(train_data)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, intermediates = model(batch)
            ce_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
            mtp2 = F.cross_entropy(
                logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
            mtp3 = F.cross_entropy(
                logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
            total_loss = ce_loss + 0.5 * mtp2 + 0.25 * mtp3

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_processed += tokens_per_step

        # Evaluation
        should_eval = (step == 1 or step % args.eval_every == 0
                       or step == total_steps)
        if should_eval:
            val_loss = eval_model(model)
            results['eval_curve'].append({
                'step': step,
                'val_loss': val_loss,
                'tokens': tokens_processed,
                'elapsed': round(time.time() - t0, 1),
                'ce_loss': ce_loss.item(),
                'mtp2': mtp2.item(),
                'mtp3': mtp3.item(),
                'total_loss': total_loss.item(),
            })
            print(f"  Step {step:>6}/{total_steps} | val={val_loss:.4f} | "
                  f"ce={ce_loss.item():.4f} | "
                  f"{tokens_processed / 1e6:.0f}M tok | {time.time() - t0:.0f}s")

        # Interference measurement
        if step == 1 or step % args.measure_every == 0 or step == total_steps:
            val_batch = get_batch(val_data)
            m = measure_interference_twopass(model, val_batch)
            results['measurements'].append({'step': step, **m})
            print(f"         per_row_0.3={m['per_row_0.3']:.1%}  "
                  f"neg_cos={m['interference_rate']:.1%}  "
                  f"global={m['global_cos']:+.4f}  "
                  f"PRDR={m['discrepancy']:.1f}×")

        # Checkpoint every 10000 steps
        if step % 10000 == 0:
            try:
                save_json(results, os.path.join(RESULTS_DIR, f'_partial_{outfile}'))
            except Exception:
                pass
        elif step % 5000 == 0 and not should_eval:
            print('.', end='', flush=True)

    results['total_time'] = round(time.time() - t0, 1)
    results['final_val_loss'] = eval_model(model, eval_iters=50)
    results['final_tok_per_sec'] = round(tokens_processed / (time.time() - t0))
    print(f"\n  Done. {results['total_time']:.0f}s | "
          f"final_val={results['final_val_loss']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

results = {'variant': '350m_r4_b', 'seed': args.seed, 'crashed': False}
try:
    results = train_350m()
except Exception as e:
    print(f"\n  *** CRASHED: {e}")
    import traceback; traceback.print_exc()
    results = {'variant': '350m_r4_b', 'seed': args.seed,
               'crashed': True, 'error': str(e)}
finally:
    save_json(results, os.path.join(RESULTS_DIR, outfile))

partial = os.path.join(RESULTS_DIR, f'_partial_{outfile}')
if os.path.exists(partial):
    os.remove(partial)

import subprocess
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    subprocess.run(['git', 'add', 'results/'], cwd=repo_root,
                   check=True, capture_output=True)
    subprocess.run(['git', 'commit', '-m', f'Phase C 350M R4: {outfile}'],
                   cwd=repo_root, check=True, capture_output=True)
    subprocess.run(['git', 'push'], cwd=repo_root,
                   check=True, capture_output=True)
    print("  Results pushed ✓")
except Exception:
    print(f"  Git push failed — save from {RESULTS_DIR}")

print(f"\n{'=' * 60}")
print(f"  350M R≈{R_value:.2f} variant B: seed={args.seed}  steps={total_steps}")
print(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
print(f"  Total time: {results.get('total_time', 0):.0f}s")
print(f"{'=' * 60}")
