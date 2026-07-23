"""
Phase C: Gradient surgery baselines + AdamW comparison.

MODE 1 — Gradient surgery (C1-C3): Variant B (shared MTP) + Muon + surgery.
  --method pcgrad/gs/scatter  --optimizer muon  --variant b

MODE 2 — AdamW comparison (C4-C6): Standard training with AdamW.
  --method none  --optimizer adamw  --variant a/b/gnce

Run commands:
    # C1-C3: Surgery on variant B
    python3 experiments/phase_c_negatives.py --method gs      --optimizer muon --variant b --seed 42
    python3 experiments/phase_c_negatives.py --method pcgrad  --optimizer muon --variant b --seed 42
    python3 experiments/phase_c_negatives.py --method scatter --optimizer muon --variant b --seed 42

    # C4-C6: AdamW variants (2 seeds each)
    for v in a b gnce; do
      for s in 42 123; do
        python3 experiments/phase_c_negatives.py --method none --optimizer adamw --variant $v --seed $s
      done
    done
"""

import os, sys, math, time, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from model.auxiliary_losses import GNCELoss
from baselines.pcgrad_muon import pcgrad_surgery
from baselines.gs_muon import gs_per_row
from baselines.scatter_muon import scatter_redistribute

try:
    from muon import SingleDeviceMuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    HAS_MUON = False

# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Phase C: surgery + AdamW')
parser.add_argument('--method', required=True,
                    choices=['pcgrad', 'gs', 'scatter', 'none'])
parser.add_argument('--optimizer', required=True, choices=['muon', 'adamw'])
parser.add_argument('--variant', required=True, choices=['b', 'a', 'gnce'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--steps', type=int, default=30517)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=500)
args = parser.parse_args()

# Validation
if args.method != 'none':
    assert args.optimizer == 'muon', "Surgery requires --optimizer muon"
    assert args.variant == 'b', "Surgery requires --variant b"
    assert HAS_MUON, "Muon required for surgery mode"
if args.optimizer == 'muon':
    assert HAS_MUON, "Muon not installed"

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_c'
)
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.steps

outfile = (f"{args.method}_{args.optimizer}_{args.variant}"
           f"_seed{args.seed}_{args.steps}steps.json")

print("=" * 70)
print(f"  Phase C: method={args.method}  optimizer={args.optimizer}  "
      f"variant={args.variant}")
print(f"  seed={args.seed}  steps={total_steps}  → {outfile}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (same as Phase B)
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
    if batch_size is None:
        batch_size = args.batch_size
    return data[torch.randint(0, data.shape[0], (batch_size,))]


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
# FIND lm_head PARAM NAME (handles weight tying)
# ═══════════════════════════════════════════════════════════════════════════

def find_lm_head_name(model):
    """Find the name under which lm_head.weight appears in named_parameters().
    With weight tying (lm_head.weight = wte.weight), the first name wins."""
    target_id = id(model.lm_head.weight)
    for n, p in model.named_parameters():
        if id(p) == target_id:
            return n
    raise RuntimeError("lm_head.weight not found in named_parameters()")


# ═══════════════════════════════════════════════════════════════════════════
# COLLECT GRADIENTS (deduplicated for tied weights)
# ═══════════════════════════════════════════════════════════════════════════

def collect_grads(model):
    """Collect gradients from model, deduplicated by parameter id."""
    grads = {}
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in seen or p.grad is None:
            continue
        seen.add(id(p))
        grads[n] = p.grad.clone()
    return grads


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train():
    torch.manual_seed(args.seed)
    model = GPT2().to(DEVICE)
    V = model.config.vocab_size
    lm_head_name = find_lm_head_name(model)
    use_surgery = args.method != 'none'

    # --- Auxiliary module ---
    aux_module = None
    if args.variant == 'gnce':
        aux_module = GNCELoss(n_embd=768, layers=(9, 8)).to(DEVICE)

    # --- Optimizer ---
    if args.optimizer == 'muon':
        param_groups = model.get_muon_param_groups()
        if aux_module is not None:
            param_groups = param_groups + aux_module.get_aux_param_groups()
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        params = list(model.parameters())
        if aux_module is not None:
            params += list(aux_module.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4,
                                       betas=(0.9, 0.95), weight_decay=0.0)

    # --- Results ---
    results = {
        'variant': 'phase_c',
        'method': args.method,
        'optimizer': args.optimizer,
        'model_variant': args.variant,
        'seed': args.seed,
        'total_steps': total_steps,
        'total_tokens': total_steps * tokens_per_step,
        'eval_curve': [],
        'grad_diagnostics': [],
    }

    t0 = time.time()
    tokens_processed = 0
    model.train()

    for step in range(1, total_steps + 1):

        # ── LR schedule ──
        if args.optimizer == 'muon':
            new_adam_lr = get_lr(step, 3e-4, warmup=200, total_steps=total_steps)
            for pg in optimizer.param_groups:
                if not pg.get('use_muon', False):
                    pg['lr'] = new_adam_lr
        else:
            new_lr = get_lr(step, 3e-4, warmup=200, total_steps=total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        batch = get_batch(train_data)

        # ==============================================================
        # SURGERY MODE: two-pass backward
        # ==============================================================
        if use_surgery:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, intermediates = model(batch)
                ce_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
                mtp2 = F.cross_entropy(
                    logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
                mtp3 = F.cross_entropy(
                    logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
                mtp_loss = 0.5 * mtp2 + 0.25 * mtp3

            # Pass 1: CE gradients
            optimizer.zero_grad()
            ce_loss.backward(retain_graph=True)
            ce_grads = collect_grads(model)

            # Pass 2: MTP gradients
            optimizer.zero_grad()
            mtp_loss.backward()
            mtp_grads = collect_grads(model)

            # Gradient diagnostics (every 500 steps)
            if step % 500 == 0 or step == 1:
                lm_ce = ce_grads.get(lm_head_name)
                lm_mtp = mtp_grads.get(lm_head_name)
                if lm_ce is not None and lm_mtp is not None:
                    dots = (lm_ce * lm_mtp).sum(dim=1)
                    ce_n = lm_ce.norm(dim=1).clamp(min=1e-8)
                    mtp_n = lm_mtp.norm(dim=1).clamp(min=1e-8)
                    cos = dots / (ce_n * mtp_n)
                    results['grad_diagnostics'].append({
                        'step': step,
                        'interference_rate': (cos < 0).float().mean().item(),
                        'ce_grad_norm_mean': ce_n.mean().item(),
                        'mtp_grad_norm_mean': mtp_n.mean().item(),
                        'cos_mean': cos.mean().item(),
                    })

            # Apply surgery
            if args.method == 'pcgrad':
                final_grads = pcgrad_surgery(ce_grads, mtp_grads)
            elif args.method == 'gs':
                final_grads = gs_per_row(ce_grads, mtp_grads, lm_head_name)
            elif args.method == 'scatter':
                final_grads = scatter_redistribute(ce_grads, mtp_grads, lm_head_name)

            # Set final gradients
            optimizer.zero_grad()
            seen = set()
            for n, p in model.named_parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if n in final_grads:
                    p.grad = final_grads[n]

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            aux_info = {'ce_loss': ce_loss.item(),
                        'mtp_loss': mtp_loss.item()}

        # ==============================================================
        # STANDARD MODE: single-pass backward
        # ==============================================================
        else:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, intermediates = model(batch)
                ce_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))

                if args.variant == 'a':
                    total_loss = ce_loss
                    aux_info = {'ce_loss': ce_loss.item()}

                elif args.variant == 'b':
                    mtp2 = F.cross_entropy(
                        logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
                    mtp3 = F.cross_entropy(
                        logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
                    total_loss = ce_loss + 0.5 * mtp2 + 0.25 * mtp3
                    aux_info = {'ce_loss': ce_loss.item(),
                                'mtp2': mtp2.item(), 'mtp3': mtp3.item()}

                elif args.variant == 'gnce':
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
            results['eval_curve'].append({
                'step': step,
                'val_loss': val_loss,
                'tokens': tokens_processed,
                'elapsed': round(time.time() - t0, 1),
                **aux_info,
            })
            print(f"  Step {step:>6}/{total_steps} | val={val_loss:.4f} | "
                  f"ce={aux_info.get('ce_loss', 0):.4f} | "
                  f"{tokens_processed / 1e6:.0f}M tok | {time.time() - t0:.0f}s")

        # ── Checkpoint ──
        if step % 5000 == 0:
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
    print(f"\n  Done. {total_time:.0f}s | final_val={final_val:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

results = {'method': args.method, 'optimizer': args.optimizer,
           'model_variant': args.variant, 'seed': args.seed, 'crashed': False}
try:
    results = train()
except Exception as e:
    print(f"\n  *** CRASHED: {e}")
    import traceback; traceback.print_exc()
    results = {'method': args.method, 'optimizer': args.optimizer,
               'model_variant': args.variant, 'seed': args.seed,
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
    subprocess.run(['git', 'commit', '-m', f'Phase C: {outfile}'],
                   cwd=repo_root, check=True, capture_output=True)
    subprocess.run(['git', 'push'], cwd=repo_root,
                   check=True, capture_output=True)
    print("  Results pushed ✓")
except Exception:
    print(f"  Git push failed — save from {RESULTS_DIR}")

print(f"\n{'=' * 60}")
print(f"  Phase C: {args.method}/{args.optimizer}/{args.variant} seed={args.seed}")
print(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
print(f"  Total time: {results.get('total_time', 0):.0f}s")
print(f"{'=' * 60}")
