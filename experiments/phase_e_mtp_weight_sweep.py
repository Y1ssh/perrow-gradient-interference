#!/usr/bin/env python3
"""
Phase E: MTP-weight sweep + mixture-optimum diagnostics.

Purpose (addresses external-review Item 2, the "mixture-optimum" alternative
explanation): a single tied output head trained on t+1/t+2/t+3 CE with weights
1 / w2 / w3 will, to first order, converge to a weighted mixture of the three
conditionals and pay KL(P_{t+1} || mixture) as excess next-token loss. That
mechanism would explain the +0.39-nat gap WITHOUT any gradient-geometry story.

This script rules it in or out with two probes on the SAME phase_b recipe:
  1. Sweep the auxiliary weight:  --mtp_scale s  scales (0.5, 0.25) -> (0.5s, 0.25s).
     Under the mixture account the gap shrinks ~proportionally with s.
     Under an interference account it should not.
  2. Enriched final eval logs, for the trained model:
       - next-token (t+1) CE          [the headline metric]
       - t+2 CE and t+3 CE            [is shared-MTP BETTER at these? => learned the mixture]
       - mean per-token output entropy [a mixture is higher-entropy]
     Compared against the CE-only baseline (--mtp_scale 0), these settle
     the reviewer's first two cheap checks.

Recipe is byte-identical to experiments/phase_b_comparison.py variant 'b'
(GPT-2 124M, Muon+AuxAdam, FineWeb-Edu, batch 16x1024, 30,517 steps, LR 3e-4
warmup 200 cosine-decay) except for the auxiliary-weight multiplier.

Usage (one run at a time, spot-instance friendly):
    python3 experiments/phase_e_mtp_weight_sweep.py --mtp_scale 1.0  --seed 42   # = variant b (shared-MTP anchor)
    python3 experiments/phase_e_mtp_weight_sweep.py --mtp_scale 0.0  --seed 42   # = variant a (CE-only anchor)
    python3 experiments/phase_e_mtp_weight_sweep.py --mtp_scale 0.25 --seed 42   # interior sweep point
    python3 experiments/phase_e_mtp_weight_sweep.py --mtp_scale 0.5  --seed 42   # interior sweep point

The 4 runs (scale 0, 0.1, 0.01, 1.0) at seed 42 give the sweep + the two anchor
points; ~45-65 min each on one modern GPU.
"""
import os, sys, math, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt2 import GPT2, GPT2Config
from muon import SingleDeviceMuonWithAuxAdam

parser = argparse.ArgumentParser(description='Phase E: MTP-weight sweep + mixture diagnostics')
parser.add_argument('--mtp_scale', required=True, type=float,
                    help='Multiplier on the (0.5, 0.25) MTP weights. 0 = CE-only, 1 = standard b.')
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--tokens', type=int, default=500_000_000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--eval_every', type=int, default=500)
args = parser.parse_args()

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_e')
os.makedirs(RESULTS_DIR, exist_ok=True)

tokens_per_step = args.batch_size * args.seq_len
total_steps = args.tokens // tokens_per_step
W2, W3 = 0.5 * args.mtp_scale, 0.25 * args.mtp_scale

print("=" * 70)
print(f"  Phase E: mtp_scale={args.mtp_scale}  (w2={W2}, w3={W3})  seed={args.seed}")
print(f"  {args.tokens / 1e6:.0f}M tokens | {total_steps:,} steps | batch={args.batch_size}x{args.seq_len}")
print("=" * 70)

# ── DATA (identical to phase_b) ──
import tiktoken
from datasets import load_dataset

def load_fineweb_tokens(max_tokens=500_000_000):
    cache_path = f'fineweb_train_{max_tokens // 1_000_000}M.pt'
    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)
    print(f"  Downloading FineWeb-Edu and tokenizing ({max_tokens / 1e6:.0f}M tokens)...")
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    all_tokens = []
    for example in ds:
        toks = enc.encode_ordinary(example['text'])
        all_tokens.extend(toks)
        if len(all_tokens) >= max_tokens:
            break
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

def get_batch(data, batch_size=None):
    bs = batch_size or args.batch_size
    idx = torch.randint(0, data.shape[0], (bs,))
    return data[idx]

def get_lr(step, base_lr, warmup=200, total_steps=30517):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_ratio)))

@torch.no_grad()
def eval_enriched(model, eval_iters=40):
    """Next-token CE, t+2 CE, t+3 CE, and mean output entropy on held-out val.

    Memory-safe: CE terms use F.cross_entropy directly on bf16 logits (no full
    float32 materialization); entropy is accumulated row-by-row over the flattened
    (B*T) positions in chunks so the (B*T, V) softmax never lives all at once.
    """
    model.eval()
    V = model.config.vocab_size
    ce1, ce2, ce3, ent = [], [], [], []
    CHUNK = 4096  # rows of the (B*T, V) logit matrix per entropy chunk
    for _ in range(eval_iters):
        batch = get_batch(val_data)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch)
        ce1.append(F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1)).item())
        ce2.append(F.cross_entropy(logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1)).item())
        ce3.append(F.cross_entropy(logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1)).item())
        flat = logits.reshape(-1, V)
        tot, n = 0.0, 0
        for i in range(0, flat.shape[0], CHUNK):
            lp = F.log_softmax(flat[i:i+CHUNK].float(), dim=-1)
            tot += float((-(lp.exp() * lp).sum(-1)).sum().item())
            n += lp.shape[0]
        ent.append(tot / n)
    model.train()
    return (float(np.mean(ce1)), float(np.mean(ce2)),
            float(np.mean(ce3)), float(np.mean(ent)))

def eval_nexttoken(model, eval_iters=20):
    model.eval(); V = model.config.vocab_size; losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            batch = get_batch(val_data)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(batch)
            losses.append(F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1)).item())
    model.train(); return float(np.mean(losses))

# ── TRAIN ──
torch.manual_seed(args.seed)
model = GPT2().to(DEVICE)
V = model.config.vocab_size
optimizer = SingleDeviceMuonWithAuxAdam(model.get_muon_param_groups())

results = {'phase': 'e', 'mtp_scale': args.mtp_scale, 'w2': W2, 'w3': W3,
           'seed': args.seed, 'total_steps': total_steps,
           'total_tokens': total_steps * tokens_per_step, 'eval_curve': []}

t0 = time.time(); model.train()
for step in range(1, total_steps + 1):
    new_adam_lr = get_lr(step, 3e-4, warmup=200, total_steps=total_steps)
    for pg in optimizer.param_groups:
        if not pg.get('use_muon', False):
            pg['lr'] = new_adam_lr
    batch = get_batch(train_data)
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = model(batch)
        ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        if args.mtp_scale == 0.0:
            total_loss = ce_loss
        else:
            mtp2 = F.cross_entropy(logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
            mtp3 = F.cross_entropy(logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
            total_loss = ce_loss + W2 * mtp2 + W3 * mtp3
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == 1 or step % args.eval_every == 0 or step == total_steps:
        vl = eval_nexttoken(model)
        results['eval_curve'].append({'step': step, 'val_loss': vl})
        print(f"  step {step:6d}/{total_steps}  next-tok CE {vl:.4f}  ({(time.time()-t0)/60:.1f} min)")

    # STEP-1000 SNAPSHOT: matched to the phase_a diagnostic step so the eval pass can
    # re-measure the CE-vs-L1 control and CE-vs-CE ceiling on exactly this checkpoint,
    # this time committing per-row cosines + norms + the target-presence boolean.
    if step == 1000:
        CKPT_DIR = os.path.join(os.path.dirname(RESULTS_DIR), 'checkpoints')
        os.makedirs(CKPT_DIR, exist_ok=True)
        snap = os.path.join(CKPT_DIR, f"model_scale{args.mtp_scale}_seed{args.seed}_step1000.pt")
        torch.save({'model_state_dict': model.state_dict(), 'step': 1000,
                    'mtp_scale': args.mtp_scale, 'seed': args.seed,
                    'config': {'n_layer': model.config.n_layer, 'n_head': model.config.n_head,
                               'n_embd': model.config.n_embd, 'vocab_size': model.config.vocab_size,
                               'block_size': model.config.block_size}}, snap)
        print(f"  [step-1000 snapshot] {snap}")

# ── FINAL ENRICHED EVAL (the mixture diagnostics) ──
ce1, ce2, ce3, ent = eval_enriched(model, eval_iters=40)
results.update({
    'final_val_loss': ce1,
    'final_ce_t1': ce1, 'final_ce_t2': ce2, 'final_ce_t3': ce3,
    'final_mean_entropy': ent,
    'total_time': time.time() - t0,
})
print("\n  === FINAL ENRICHED EVAL ===")
print(f"  next-token (t+1) CE : {ce1:.4f}")
print(f"  t+2 CE              : {ce2:.4f}")
print(f"  t+3 CE              : {ce3:.4f}")
print(f"  mean output entropy : {ent:.4f} nats")

# ── SAVE MODEL CHECKPOINT (needed by estimate_kl.py / T1-T3; the CE-only s=0.0 and
#    shared-MTP s=1.0 checkpoints are the two the eval scripts consume) ──
CKPT_DIR = os.path.join(os.path.dirname(RESULTS_DIR), 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(CKPT_DIR, f"model_scale{args.mtp_scale}_seed{args.seed}.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'mtp_scale': args.mtp_scale, 'w2': W2, 'w3': W3, 'seed': args.seed,
    'total_steps': total_steps, 'final_val_loss': ce1,
    'config': {'n_layer': model.config.n_layer, 'n_head': model.config.n_head,
               'n_embd': model.config.n_embd, 'vocab_size': model.config.vocab_size,
               'block_size': model.config.block_size},
}, ckpt_path)
results['checkpoint_path'] = ckpt_path
print(f"  Saved checkpoint {ckpt_path}")

out = os.path.join(RESULTS_DIR, f"sweep_scale{args.mtp_scale}_seed{args.seed}_{total_steps}steps.json")
def convert(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=convert)
print(f"\n  Saved {out}")
