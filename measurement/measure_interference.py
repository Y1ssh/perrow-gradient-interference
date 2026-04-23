"""
Per-row gradient interference measurement.

THE DISCOVERY:
    When two losses (CE + MTP) share a weight matrix (lm_head), the global
    cosine similarity between their gradients reads ~0 ("no conflict").
    But examining individual ROWS — one per output neuron — reveals that
    60-98% of neurons receive genuinely conflicting gradient signals.

    PCGrad, CAGrad, MGDA, and all existing multi-loss methods operate on
    GLOBAL (flattened) gradient vectors. They ALL miss per-row conflict.

WHY IT WORKS:
    lm_head.weight has shape (vocab_size, n_embd).
    Row i = the weight vector for vocabulary token i.
    CE wants row i to push toward predicting t+1.
    MTP wants row i to push toward predicting t+2 or t+3.
    These are DIFFERENT targets → the per-row gradients conflict.
    But across all 50304 rows, conflicts in opposite directions cancel
    in the global average → global cosine ≈ 0 hides the per-row conflict.

    Analogy: A company survey says "employees are happy on average."
    But 60% of individual teams are miserable — the average hides it.

USAGE:
    from measurement.measure_interference import measure_interference

    result = measure_interference(model, val_batch, loss_pair='ce_mtp')
    print(f"Global cos: {result['global_cos']:.3f}")
    print(f"Per-row >0.3: {result['per_row_fractions'][0.3]:.1%}")
    print(f"Discrepancy: {result['discrepancy']:.1f}×")

    # Control (should show 0% — calibrates measurement)
    control = measure_interference(model, val_batch, loss_pair='ce_l1')
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _row_cosines(grad1, grad2):
    """
    Compute global and per-row cosine similarity between two gradient matrices.

    Args:
        grad1, grad2: tensors of shape (rows, cols), any dtype.
    Returns:
        global_cos: float — cosine between flattened vectors.
        row_cos: float32 tensor (rows,) — cosine per output neuron.

    Both inputs are cast to float32 for numerical precision.
    This matters on H100 where gradients arrive in bfloat16.
    """
    g1 = grad1.float()
    g2 = grad2.float()

    # Global: flatten to single vector, compute one cosine
    global_cos = F.cosine_similarity(
        g1.reshape(1, -1), g2.reshape(1, -1)
    ).item()

    # Per-row: each row independently
    row_cos = F.cosine_similarity(g1, g2, dim=1)  # (rows,)

    return global_cos, row_cos


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def measure_interference(model, batch, loss_pair='ce_mtp',
                         thresholds=(0.1, 0.2, 0.3, 0.5)):
    """
    Measure per-row gradient interference on lm_head between two losses.

    Args:
        model: GPT2 model (must have .lm_head.weight and .config.vocab_size)
        batch: (B, T) token indices, T >= 4
        loss_pair:
            'ce_mtp' — CE(t+1) vs MTP(t+2, t+3)  [main experiment]
            'ce_l1'  — CE(t+1) vs L1(|W|)          [control, expect 0%]
        thresholds: |cos| thresholds for per-row fraction reporting

    Returns: dict with:
        'global_cos':        float   — cosine between flattened CE and aux gradients
        'per_row_fractions': dict    — {threshold: fraction of rows with |cos| > threshold}
        'row_cosines':       ndarray — (vocab_size,) full per-row cosine distribution
        'val_loss':          float   — CE loss on this batch
        'discrepancy':       float   — per_row_0.3 / max(|global_cos|, 0.001)

    IMPORTANT:
        - Do NOT call this inside torch.cuda.amp.autocast().
          Gradients must be computed without autocast for accuracy.
          BF16 gradients from the model are cast to float32 internally.
        - Uses torch.enable_grad() so it works even if the caller has
          torch.no_grad() active.
    """
    assert batch.dim() == 2, f"Expected (B, T) batch, got shape {batch.shape}"
    assert batch.size(1) >= 4, f"Need T >= 4 for MTP-3, got T={batch.size(1)}"

    was_training = model.training
    model.eval()

    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
        logits, _ = model(batch)
        V = model.config.vocab_size

        # CE loss: position t predicts token t+1
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            batch[:, 1:].reshape(-1),
        )

        # Auxiliary loss
        if loss_pair == 'ce_mtp':
            # MTP: position t predicts token t+2 (weight 0.5) and t+3 (weight 0.25)
            mtp2_loss = F.cross_entropy(
                logits[:, :-2].reshape(-1, V),
                batch[:, 2:].reshape(-1),
            )
            mtp3_loss = F.cross_entropy(
                logits[:, :-3].reshape(-1, V),
                batch[:, 3:].reshape(-1),
            )
            aux_loss = 0.5 * mtp2_loss + 0.25 * mtp3_loss
        elif loss_pair == 'ce_l1':
            # Control: L1 norm of lm_head weights (independent of forward pass)
            aux_loss = model.lm_head.weight.abs().mean()
        else:
            raise ValueError(f"Unknown loss_pair: {loss_pair!r}")

        # Extract per-loss gradients on the shared weight matrix
        # retain_graph=True on first call: graph still needed for aux_loss
        target = model.lm_head.weight
        ce_grad = torch.autograd.grad(ce_loss, target, retain_graph=True)[0]
        aux_grad = torch.autograd.grad(aux_loss, target)[0]

    # Compute cosine similarities (in float32 for precision)
    global_cos, row_cos = _row_cosines(ce_grad, aux_grad)

    # Per-row fractions at each threshold
    per_row_fractions = {}
    for t in thresholds:
        per_row_fractions[t] = (row_cos.abs() > t).float().mean().item()

    # Discrepancy: how much per-row conflict is hidden by the global metric
    if 0.3 in per_row_fractions:
        frac_0_3 = per_row_fractions[0.3]
    else:
        frac_0_3 = (row_cos.abs() > 0.3).float().mean().item()
    discrepancy = frac_0_3 / max(abs(global_cos), 0.001)

    model.train(was_training)

    return {
        'global_cos': global_cos,
        'per_row_fractions': per_row_fractions,
        'row_cosines': row_cos.detach().cpu().numpy(),
        'val_loss': ce_loss.item(),
        'discrepancy': discrepancy,
    }


# ---------------------------------------------------------------------------
# Per-layer measurement (for Figure 8 / supplementary)
# ---------------------------------------------------------------------------

def measure_per_layer_interference(model, batch):
    """
    Measure per-row interference at EVERY 2D weight matrix in the model.

    Returns: dict {param_name: {'global_cos': float, 'per_row_0.3': float, 'shape': list}}

    Uses batch autograd.grad for efficiency — only 2 backward passes total
    regardless of how many parameters are measured.

    Tied weights (wte ↔ lm_head) are deduplicated.
    """
    assert batch.dim() == 2 and batch.size(1) >= 4

    was_training = model.training
    model.eval()

    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
        logits, _ = model(batch)
        V = model.config.vocab_size

        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1),
        )
        mtp2_loss = F.cross_entropy(
            logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1),
        )
        mtp3_loss = F.cross_entropy(
            logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1),
        )
        mtp_loss = 0.5 * mtp2_loss + 0.25 * mtp3_loss

        # Collect unique 2D parameters
        seen = set()
        names, params = [], []
        for name, p in model.named_parameters():
            if id(p) not in seen and p.dim() >= 2:
                seen.add(id(p))
                names.append(name)
                params.append(p)

        # 2 backward passes total (efficient)
        ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True)
        mtp_grads = torch.autograd.grad(mtp_loss, params)

    # Compute per-param cosine similarities
    results = {}
    for name, ce_g, mtp_g in zip(names, ce_grads, mtp_grads):
        global_cos, row_cos = _row_cosines(ce_g, mtp_g)
        results[name] = {
            'global_cos': global_cos,
            'per_row_0.3': (row_cos.abs() > 0.3).float().mean().item(),
            'shape': list(ce_g.shape),
        }

    model.train(was_training)
    return results


# ---------------------------------------------------------------------------
# Lightweight version for use during training loops
# ---------------------------------------------------------------------------

def quick_measure(model, batch):
    """
    Fast interference measurement: global_cos + per_row_0.3 on lm_head only.

    Called every N training steps for logging. Returns minimal dict.
    Same math as measure_interference but skips histogram and multi-threshold.
    """
    was_training = model.training
    model.eval()

    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
        logits, _ = model(batch)
        V = model.config.vocab_size

        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1),
        )
        mtp2_loss = F.cross_entropy(
            logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1),
        )
        mtp3_loss = F.cross_entropy(
            logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1),
        )
        mtp_loss = 0.5 * mtp2_loss + 0.25 * mtp3_loss

        target = model.lm_head.weight
        ce_grad = torch.autograd.grad(ce_loss, target, retain_graph=True)[0]
        mtp_grad = torch.autograd.grad(mtp_loss, target)[0]

    global_cos, row_cos = _row_cosines(ce_grad, mtp_grad)
    per_row = (row_cos.abs() > 0.3).float().mean().item()

    model.train(was_training)

    return {
        'global_cos': global_cos,
        'per_row_0.3': per_row,
        'discrepancy': per_row / max(abs(global_cos), 0.001),
        'val_loss': ce_loss.item(),
    }
