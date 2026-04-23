"""
Phase A: Per-row interference measurements at GPT-2 124M scale.

Experiments:
    A1: Muon interference trajectory (1000 steps, logged every 50)
    A2: AdamW interference trajectory (1000 steps, logged every 50)
    A3: Control — CE vs L1 (must show 0% interference)
    A4: Untied lm_head (checks weight-tying confound)
    A5: Matched val-loss comparison (Muon vs AdamW at same quality)

Paper deliverables:
    Table 1 — interference across scales/optimizers
    Table 2 — Muon vs AdamW at matched val loss
    Figure 1 — per-row cosine histograms
    Figure 2 — interference trajectory over training
    Figure 8 — per-layer breakdown (supplementary)

Usage:
    cd perrow-gradient-interference
    python experiments/phase_a_measurements.py
"""

import os, sys, math, time, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from measurement.measure_interference import (
    measure_interference, measure_per_layer_interference, quick_measure,
)

# Muon — graceful failure
try:
    from muon import SingleDeviceMuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    print("WARNING: Muon not installed. pip install git+https://github.com/KellerJordan/Muon")
    HAS_MUON = False

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_a'
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

import tiktoken
from datasets import load_dataset

SEQ_LEN = 1024
BATCH_SIZE = 16  # smaller batch for Phase A (measurement focus)


def load_fineweb_tokens(max_tokens=50_000_000):
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


def compute_token_frequencies(tokens, vocab_size=50304):
    """Count how often each token appears. Returns (vocab_size,) long tensor on CPU."""
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for i in range(0, len(tokens), 1_000_000):
        chunk = tokens[i:i + 1_000_000]
        counts.scatter_add_(0, chunk, torch.ones_like(chunk))
    return counts


# Load and prepare data
print("=" * 70)
print("  Loading data...")
print("=" * 70)
tokens = load_fineweb_tokens(max_tokens=50_000_000)
token_freqs = compute_token_frequencies(tokens)

# Split: first 48M train, last 2M val
train_tokens = tokens[:48_000_000]
val_tokens = tokens[48_000_000:]

train_data = train_tokens[: len(train_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)
val_data = val_tokens[: len(val_tokens) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN).to(DEVICE)

print(f"  Train: {train_data.shape[0]} sequences × {SEQ_LEN}")
print(f"  Val:   {val_data.shape[0]} sequences × {SEQ_LEN}")


def get_batch(data, batch_size=BATCH_SIZE):
    """Random sampling (not sequential)."""
    idx = torch.randint(0, data.shape[0], (batch_size,))
    return data[idx]


# ═══════════════════════════════════════════════════════════════════════════
# EVAL + LR SCHEDULE
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
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def get_lr(step, base_lr, warmup=100, total_steps=1000):
    """Linear warmup + cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_ratio)))


# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPER
# ═══════════════════════════════════════════════════════════════════════════

def save_json(obj, path):
    """Save to JSON with numpy/float-key handling."""
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, dict):
            # Convert float keys to strings (JSON requires string keys)
            return {str(k) if isinstance(k, float) else k: convert(v)
                    for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(x) for x in o]
        return o
    with open(path, 'w') as f:
        json.dump(convert(obj), f, indent=2)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: find measurement by step
# ═══════════════════════════════════════════════════════════════════════════

def find_measurement(measurements, step):
    """Find measurement entry closest to given step."""
    return min(measurements, key=lambda m: abs(m['step'] - step))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING + MEASUREMENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(name, model, optimizer, total_steps, measure_every=50,
                   loss_mode='mtp_shared', is_muon=True):
    """
    Train model with CE (+MTP), measure interference periodically.

    Args:
        name: experiment label
        model: GPT2 on cuda
        optimizer: Muon or AdamW
        total_steps: training steps
        measure_every: interference logging frequency
        loss_mode: 'mtp_shared' or 'ce_only'
        is_muon: affects LR schedule (Muon groups stay flat)

    Returns: results dict with measurements, training curve, per-layer data.
    """
    V = model.config.vocab_size
    results = {
        'name': name,
        'total_steps': total_steps,
        'loss_mode': loss_mode,
        'optimizer': 'muon' if is_muon else 'adamw',
        'measurements': [],
        'per_layer_snapshots': {},
        'token_freq_correlation': {},
    }

    t0 = time.time()
    tokens_processed = 0

    # Steps for extra data collection
    histogram_steps = {1, 100, 250, 500, 1000}
    per_layer_steps = {1, 500, 1000}
    freq_corr_steps = {500, 1000}

    model.train()

    for step in range(1, total_steps + 1):

        # ── LR schedule ──
        if is_muon:
            # Muon groups: flat LR.  Adam groups: cosine decay.
            new_adam_lr = get_lr(step, 3e-4, warmup=100, total_steps=total_steps)
            for pg in optimizer.param_groups:
                if not pg.get('use_muon', False):
                    pg['lr'] = new_adam_lr
        else:
            new_lr = get_lr(step, 1e-3, warmup=100, total_steps=total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        # ── Training step ──
        batch = get_batch(train_data)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch)
            ce_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))

            if loss_mode == 'mtp_shared':
                mtp2 = F.cross_entropy(
                    logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
                mtp3 = F.cross_entropy(
                    logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
                total_loss = ce_loss + 0.5 * mtp2 + 0.25 * mtp3
            else:
                total_loss = ce_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_processed += BATCH_SIZE * SEQ_LEN

        # ── Measurements ──
        should_measure = (step == 1 or step % measure_every == 0
                          or step == total_steps)

        if should_measure:
            val_loss = eval_model(model, eval_iters=10)

            # Core measurement — OUTSIDE autocast (function has its own defense)
            val_batch = get_batch(val_data)
            m = measure_interference(model, val_batch, loss_pair='ce_mtp')

            measurement = {
                'step': step,
                'val_loss': val_loss,
                'global_cos': m['global_cos'],
                'discrepancy': m['discrepancy'],
                'elapsed': round(time.time() - t0, 1),
                'tok_per_sec': round(tokens_processed / (time.time() - t0)),
            }
            for t_val, frac in m['per_row_fractions'].items():
                measurement[f'per_row_{t_val}'] = frac

            # Full histogram at key steps (for Figure 1)
            if step in histogram_steps:
                measurement['row_cosines'] = m['row_cosines'].tolist()

            results['measurements'].append(measurement)

            # Checkpoint partial results (crash recovery)
            if len(results['measurements']) % 5 == 0:
                try:
                    save_json(results, os.path.join(RESULTS_DIR, f'_partial_{name}.json'))
                except Exception:
                    pass

            elapsed = time.time() - t0
            print(f"  Step {step:>5} | val={val_loss:.4f} | "
                  f"global={m['global_cos']:+.4f} | "
                  f"per_row>0.3={m['per_row_fractions'][0.3]:.1%} | "
                  f"disc={m['discrepancy']:.1f}× | "
                  f"{tokens_processed / (elapsed + 1e-9) / 1000:.0f}K tok/s | "
                  f"{elapsed:.0f}s")

            # Per-layer breakdown at key steps (for Figure 8)
            if step in per_layer_steps:
                pl = measure_per_layer_interference(model, val_batch)
                results['per_layer_snapshots'][str(step)] = pl
                top3 = sorted(pl.items(),
                              key=lambda x: x[1]['per_row_0.3'], reverse=True)[:3]
                for lname, ld in top3:
                    print(f"         {lname}: "
                          f"global={ld['global_cos']:+.3f} "
                          f"per_row>0.3={ld['per_row_0.3']:.1%}")

            # Token frequency correlation at key steps
            if step in freq_corr_steps:
                row_cos = m['row_cosines']  # numpy (50304,)
                freqs = token_freqs.numpy().astype(float)
                freq_order = np.argsort(freqs)
                n = len(freq_order)
                quintiles = {}
                for q, label in enumerate(
                    ['bottom_20%', '20-40%', '40-60%', '60-80%', 'top_20%']
                ):
                    start = q * n // 5
                    end = (q + 1) * n // 5
                    idx = freq_order[start:end]
                    quintiles[label] = {
                        'mean_abs_cos': float(np.abs(row_cos[idx]).mean()),
                        'frac_above_0.3': float((np.abs(row_cos[idx]) > 0.3).mean()),
                    }
                results['token_freq_correlation'][str(step)] = quintiles
                print(f"         freq: rare={quintiles['bottom_20%']['frac_above_0.3']:.1%} "
                      f"common={quintiles['top_20%']['frac_above_0.3']:.1%}")

        elif step % 100 == 0:
            print('.', end='', flush=True)

    total_time = time.time() - t0
    results['total_time'] = round(total_time, 1)
    results['final_tok_per_sec'] = round(tokens_processed / total_time)
    print(f"\n  Done. {total_time:.0f}s | "
          f"{tokens_processed / total_time / 1000:.0f}K tok/s")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# A1: MUON INTERFERENCE AT 124M
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  EXPERIMENT A1: Muon interference at 124M")
print("=" * 70)

assert HAS_MUON, "Muon required for A1. pip install git+https://github.com/KellerJordan/Muon"

torch.manual_seed(42)
model_a1 = GPT2().to(DEVICE)
opt_a1 = SingleDeviceMuonWithAuxAdam(model_a1.get_muon_param_groups())
a1_results = {'measurements': []}  # default in case of crash
try:
    a1_results = run_experiment(
        "A1_muon_interference", model_a1, opt_a1,
        total_steps=1000, measure_every=50,
        loss_mode='mtp_shared', is_muon=True,
    )
except Exception as e:
    print(f"\n  *** A1 CRASHED: {e}")
    import traceback; traceback.print_exc()
finally:
    save_json(a1_results, os.path.join(RESULTS_DIR, 'a1_muon_interference.json'))

del model_a1, opt_a1
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# A2: ADAMW INTERFERENCE AT 124M
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  EXPERIMENT A2: AdamW interference at 124M")
print("=" * 70)

torch.manual_seed(42)
model_a2 = GPT2().to(DEVICE)

# AdamW — deduplicate tied weights
seen = set()
params_decay, params_nodecay = [], []
for name, p in model_a2.named_parameters():
    if id(p) in seen:
        continue
    seen.add(id(p))
    if p.dim() >= 2:
        params_decay.append(p)
    else:
        params_nodecay.append(p)

opt_a2 = torch.optim.AdamW([
    {'params': params_decay, 'lr': 1e-3, 'weight_decay': 0.01},
    {'params': params_nodecay, 'lr': 1e-3, 'weight_decay': 0.0},
], betas=(0.9, 0.95), eps=1e-8)

a2_results = {'measurements': []}  # default in case of crash
try:
    a2_results = run_experiment(
        "A2_adamw_interference", model_a2, opt_a2,
        total_steps=1000, measure_every=50,
        loss_mode='mtp_shared', is_muon=False,
    )
except Exception as e:
    print(f"\n  *** A2 CRASHED: {e}")
    import traceback; traceback.print_exc()
finally:
    save_json(a2_results, os.path.join(RESULTS_DIR, 'a2_adamw_interference.json'))

del model_a2, opt_a2
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# A3: CONTROL — CE vs L1 (must show 0% interference)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  EXPERIMENT A3: Control — CE vs L1 (must be 0%)")
print("=" * 70)

a3_results = {'name': 'A3_control', 'per_row_fractions': {}, 'crashed': False}
try:
    torch.manual_seed(42)
    model_a3 = GPT2().to(DEVICE)
    V = model_a3.config.vocab_size
    opt_a3 = torch.optim.AdamW(model_a3.parameters(), lr=1e-3)
    for s in range(50):
        batch = get_batch(train_data)
        opt_a3.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model_a3(batch)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        loss.backward()
        opt_a3.step()
    val_batch = get_batch(val_data)
    control = measure_interference(model_a3, val_batch, loss_pair='ce_l1')
    a3_results = {
        'name': 'A3_control',
        'global_cos': control['global_cos'],
        'per_row_fractions': control['per_row_fractions'],
        'discrepancy': control['discrepancy'],
        'val_loss': control['val_loss'],
    }
    if control['per_row_fractions'][0.3] < 0.01:
        print(f"  ✓ CONTROL PASSED — measurement calibrated")
    else:
        print(f"  ✗ CONTROL FAILED — something is wrong with measurement!")
except Exception as e:
    print(f"\n  *** A3 CRASHED: {e}")
    import traceback; traceback.print_exc()
finally:
    save_json(a3_results, os.path.join(RESULTS_DIR, 'a3_control.json'))

try:
    del model_a3, opt_a3
except NameError:
    pass
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# A4: UNTIED lm_head (weight-tying confound check)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  EXPERIMENT A4: Untied lm_head (confound check)")
print("=" * 70)

assert HAS_MUON, "Muon required for A4"

torch.manual_seed(42)
model_a4 = GPT2().to(DEVICE)

# UNTIE: create a separate lm_head.weight (no longer shares with wte)
model_a4.lm_head.weight = nn.Parameter(model_a4.lm_head.weight.clone())
print(f"  Weight tied: {model_a4.lm_head.weight is model_a4.wte.weight}")  # False

opt_a4 = SingleDeviceMuonWithAuxAdam(model_a4.get_muon_param_groups())
a4_results = {'measurements': []}  # default in case of crash
try:
    a4_results = run_experiment(
        "A4_untied_muon", model_a4, opt_a4,
        total_steps=500, measure_every=50,
        loss_mode='mtp_shared', is_muon=True,
    )
except Exception as e:
    print(f"\n  *** A4 CRASHED: {e}")
    import traceback; traceback.print_exc()
finally:
    save_json(a4_results, os.path.join(RESULTS_DIR, 'a4_untied.json'))

del model_a4, opt_a4
gc.collect(); torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# A5: MATCHED VAL-LOSS COMPARISON (analysis, no training)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ANALYSIS A5: Muon vs AdamW at matched val loss")
print("  (Addresses Logic Gap 3: convergence speed confound)")
print("=" * 70)

# For each A1 measurement, find closest A2 measurement by val_loss
matched = []
if a1_results.get('measurements') and a2_results.get('measurements'):
    a1_points = [(m['val_loss'], m) for m in a1_results['measurements']]
    a2_points = [(m['val_loss'], m) for m in a2_results['measurements']]

    for a1_loss, a1_m in a1_points:
        best_a2_loss, best_a2_m = min(a2_points, key=lambda x: abs(x[0] - a1_loss))
        if abs(a1_loss - best_a2_loss) < 0.5:
            matched.append({
                'val_loss': round((a1_loss + best_a2_loss) / 2, 4),
                'muon_step': a1_m['step'],
                'adamw_step': best_a2_m['step'],
                'muon_global_cos': a1_m['global_cos'],
                'adamw_global_cos': best_a2_m['global_cos'],
                'muon_per_row_0.3': a1_m['per_row_0.3'],
                'adamw_per_row_0.3': best_a2_m['per_row_0.3'],
                'muon_discrepancy': a1_m['discrepancy'],
                'adamw_discrepancy': best_a2_m['discrepancy'],
            })

    print(f"\n  {'Val Loss':>10} | {'Muon Step':>10} | {'AdamW Step':>10} | "
          f"{'Muon PRDR':>10} | {'AdamW PRDR':>10}")
    print(f"  {'-' * 62}")
    for mc in matched:
        print(f"  {mc['val_loss']:>10.4f} | {mc['muon_step']:>10} | {mc['adamw_step']:>10} | "
              f"{mc['muon_discrepancy']:>10.1f}× | {mc['adamw_discrepancy']:>10.1f}×")
else:
    print("  Skipped — A1 or A2 missing data")

a5_results = {'matched_comparisons': matched}
save_json(a5_results, os.path.join(RESULTS_DIR, 'a5_matched_loss.json'))


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PHASE A SUMMARY")
print("=" * 70)

def safe_final(results, label, opt_name):
    ms = results.get('measurements', [])
    if not ms:
        print(f"\n  {label} ({opt_name}): *** NO DATA (crashed) ***")
        return None
    f = ms[-1]
    print(f"\n  {label} ({opt_name}, step {f['step']}):")
    print(f"    Global cos:    {f['global_cos']:+.4f}")
    print(f"    Per-row>0.3:   {f['per_row_0.3']:.1%}")
    print(f"    Discrepancy:   {f['discrepancy']:.1f}×")
    print(f"    Val loss:      {f['val_loss']:.4f}")
    return f

a1_final = safe_final(a1_results, "A1", "Muon")
a2_final = safe_final(a2_results, "A2", "AdamW")

if a1_final and a2_final:
    ratio = a1_final['discrepancy'] / max(a2_final['discrepancy'], 0.01)
    print(f"\n  Muon vs AdamW: {ratio:.1f}× more with Muon")
else:
    print(f"\n  Muon vs AdamW: skipped (missing data)")

try:
    frac_03 = a3_results['per_row_fractions'][0.3]
    print(f"\n  A3 (Control): per_row>0.3 = {frac_03:.4%} "
          f"{'✓ PASS' if frac_03 < 0.01 else '✗ FAIL'}")
except (KeyError, TypeError):
    print(f"\n  A3 (Control): *** NO DATA ***")

a4_final = safe_final(a4_results, "A4", "Untied Muon")
if a4_final and a1_results.get('measurements'):
    a1_at_500 = find_measurement(a1_results['measurements'], step=500)
    print(f"\n  A4 vs A1 at step ~500:")
    print(f"    Tied:    {a1_at_500['per_row_0.3']:.1%}")
    print(f"    Untied:  {a4_final['per_row_0.3']:.1%}")

if a1_final:
    print(f"\n  DECISION:")
    if a1_final['discrepancy'] > 5:
        print(f"  ✓ Interference confirmed at 124M ({a1_final['discrepancy']:.1f}×) — PROCEED")
    else:
        print(f"  ⚠ Interference lower than expected ({a1_final['discrepancy']:.1f}×)")

total_phase_time = sum(
    r.get('total_time', 0) for r in [a1_results, a2_results, a4_results]
)
print(f"\n  Total Phase A time: {total_phase_time:.0f}s")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# GIT PUSH (best-effort)
# ═══════════════════════════════════════════════════════════════════════════

import subprocess

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    subprocess.run(['git', 'add', 'results/'], cwd=repo_root, check=True,
                   capture_output=True)
    subprocess.run(['git', 'commit', '-m', 'Phase A measurement results'],
                   cwd=repo_root, check=True, capture_output=True)
    subprocess.run(['git', 'push'], cwd=repo_root, check=True,
                   capture_output=True)
    print("  Results pushed to GitHub ✓")
except Exception as e:
    print(f"  Could not git push ({e}). Manually save from {RESULTS_DIR}")
