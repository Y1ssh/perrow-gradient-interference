"""
Phase C: Pure Muon — lm_head.weight under Muon instead of Adam.

Standard config: Muon on blocks 2D weights, Adam on wte/wpe/lm_head.
This experiment: Muon on blocks 2D weights AND wte/lm_head (tied).
                 Adam on wpe only.

Tests whether per-row interference patterns change when lm_head uses
orthogonal (Muon) vs adaptive (Adam) optimization.

Variant B (shared MTP) — measures per-row interference every 1000 steps.

Usage:
    python3 experiments/phase_c_pure_muon.py --seed 42
"""

import os, sys, math, time, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from muon import SingleDeviceMuonWithAuxAdam

parser = argparse.ArgumentParser(description='Phase C: Pure Muon on lm_head')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--steps', type=int, default=30517)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=500)
args = parser.parse_args()

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_c'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.steps
outfile = f'pure_muon_b_seed{args.seed}_{args.steps}steps.json'

print("=" * 70)
print(f"  Phase C: Pure Muon (lm_head in Muon group)  seed={args.seed}")
print(f"  Variant B (shared MTP)  {total_steps} steps")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

import tiktoken
from datasets import load_dataset


def load_fineweb_tokens(max_tokens=500_000_000):
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
    return t


print("\n  Loading data...")
tokens = load_fineweb_tokens(max_tokens=500_000_000)
train_tokens = tokens[:480_000_000]
val_tokens = tokens[480_000_000:]
SEQ_LEN = args.seq_len
train_data = train_tokens[: len(train_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
val_data = val_tokens[: len(val_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
print(f"  Train: {train_data.shape[0]}  Val: {val_data.shape[0]}")


def get_batch(data):
    return data[torch.randint(0, data.shape[0], (args.batch_size,))]


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


def get_lr(step, base_lr, warmup=200, total_steps=30517):
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
# PURE MUON PARAM GROUPS
# ═══════════════════════════════════════════════════════════════════════════

def build_pure_muon_groups(model):
    """Put ALL 2D params (including wte/lm_head) in Muon. Only wpe in Adam."""
    muon_params = []
    adam_params = []
    seen = set()

    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() >= 2 and 'wpe' not in name:
            # blocks + wte/lm_head (tied) → Muon
            muon_params.append(p)
        else:
            # wpe (2D but embedding) → Adam
            adam_params.append(p)

    muon_n = sum(p.numel() for p in muon_params)
    adam_n = sum(p.numel() for p in adam_params)
    print(f"  Pure Muon groups:")
    print(f"    Muon: {muon_n:,} params ({len(muon_params)} tensors) "
          f"— blocks + wte/lm_head")
    print(f"    Adam: {adam_n:,} params ({len(adam_params)} tensors) — wpe only")

    return [
        dict(params=muon_params, use_muon=True,
             lr=0.02, momentum=0.95, weight_decay=0.01),
        dict(params=adam_params, use_muon=False,
             lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# INTERFERENCE MEASUREMENT
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

    model.train()
    return {
        'global_cos': global_cos,
        'interference_rate': interference,
        'cos_mean': cos.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_pure_muon():
    torch.manual_seed(args.seed)
    model = GPT2().to(DEVICE)
    V = model.config.vocab_size

    param_groups = build_pure_muon_groups(model)
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    results = {
        'variant': 'pure_muon_b',
        'seed': args.seed,
        'total_steps': total_steps,
        'total_tokens': total_steps * tokens_per_step,
        'eval_curve': [],
        'measurements': [],
    }

    t0 = time.time()
    tokens_processed = 0
    model.train()

    for step in range(1, total_steps + 1):
        # Only Adam groups get LR schedule
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

        # Interference measurement every 1000 steps
        if step == 1 or step % 1000 == 0 or step == total_steps:
            val_batch = get_batch(val_data)
            m = measure_interference_twopass(model, val_batch)
            results['measurements'].append({'step': step, **m})
            print(f"         interference={m['interference_rate']:.1%}  "
                  f"global_cos={m['global_cos']:+.4f}")

        if step % 5000 == 0:
            try:
                save_json(results, os.path.join(RESULTS_DIR, f'_partial_{outfile}'))
            except Exception:
                pass
        elif step % 1000 == 0 and not should_eval:
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

results = {'variant': 'pure_muon_b', 'seed': args.seed, 'crashed': False}
try:
    results = train_pure_muon()
except Exception as e:
    print(f"\n  *** CRASHED: {e}")
    import traceback; traceback.print_exc()
    results = {'variant': 'pure_muon_b', 'seed': args.seed,
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
    subprocess.run(['git', 'commit', '-m', f'Phase C Pure Muon: {outfile}'],
                   cwd=repo_root, check=True, capture_output=True)
    subprocess.run(['git', 'push'], cwd=repo_root,
                   check=True, capture_output=True)
    print("  Results pushed ✓")
except Exception:
    print(f"  Git push failed — save from {RESULTS_DIR}")

print(f"\n{'=' * 60}")
print(f"  Pure Muon B: seed={args.seed}")
print(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
print(f"  Total time: {results.get('total_time', 0):.0f}s")
print(f"{'=' * 60}")
