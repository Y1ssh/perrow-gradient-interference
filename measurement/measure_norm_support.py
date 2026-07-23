"""
measure_norm_support.py  --  direct per-row NORM measurement for the output projection.

WHY THIS EXISTS
---------------
The committed diagnostic (measure_interference.py) stores per-row *cosines* between the
CE and MTP gradients, but not their per-row *norms*. That is enough to show the per-row
directions are aligned (median cos = +1) while the aggregate (magnitude-weighted) cosine
is near zero. It is NOT enough to prove *why* the aggregate collapses:

    aggregate = sum_i w_i cos_i,   w_i = ||g_ce,i|| ||g_mtp,i|| / (||G_ce|| ||G_mtp||)  >= 0

Two mechanisms both produce a near-zero (or negative) aggregate with aligned per-row cos:
    (A) SUPPORT DIVERGENCE  -- CE puts its magnitude on a different set of rows than MTP,
        so the products ||g_ce,i||*||g_mtp,i|| are small exactly where each is large.
    (B) HIGH-NORM OPPOSED ROWS -- the small opposed minority (cos<0, ~0.33% of rows)
        carries disproportionate norm and drags the weighted sum down.

Distinguishing (A) from (B) requires the per-row norms, which is what this module logs.
It is a ~6-line extension of the existing instrument and runs on the SAME short diagnostic
run (~1000 steps), not the long training runs.

DROP-IN USAGE (mirrors measure_interference):
    from measurement.measure_norm_support import measure_norm_support
    r = measure_norm_support(model, val_batch)
    # r adds, on top of the usual fields:
    #   'ce_row_norms', 'mtp_row_norms'      : (V,) float32 per-row gradient norms
    #   'norm_profile_cos'                   : cos(a,b), cosine between the two per-row norm
    #                                          profiles in [0,1]. This is exactly the aggregate
    #                                          the head WOULD have if every per-row cosine were +1,
    #                                          so comparing it to the actual (possibly negative)
    #                                          aggregate isolates the opposed-mass contribution.
    #                                          (near 0 = disjoint support; near 1 = shared support)
    #   'opposed_norm_fraction'              : share of total ||g_ce||*||g_mtp|| mass on cos<0 rows
    #                                          (large => cancellation channel; small => support divergence)
"""
import numpy as np
import torch
import torch.nn.functional as F


def measure_norm_support(model, batch, thresholds=(0.1, 0.2, 0.3, 0.5)):
    assert batch.dim() == 2 and batch.size(1) >= 4
    was_training = model.training
    model.eval()

    with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
        logits, _ = model(batch)
        V = model.config.vocab_size
        ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        mtp2 = F.cross_entropy(logits[:, :-2].reshape(-1, V), batch[:, 2:].reshape(-1))
        mtp3 = F.cross_entropy(logits[:, :-3].reshape(-1, V), batch[:, 3:].reshape(-1))
        mtp_loss = 0.5 * mtp2 + 0.25 * mtp3
        target = model.lm_head.weight
        ce_g = torch.autograd.grad(ce_loss, target, retain_graph=True)[0].float()
        mtp_g = torch.autograd.grad(mtp_loss, target)[0].float()

    # per-row cosines (as before)
    row_cos = F.cosine_similarity(ce_g, mtp_g, dim=1)                      # (V,)
    global_cos = F.cosine_similarity(ce_g.reshape(1, -1), mtp_g.reshape(1, -1)).item()

    # --- the NEW quantities: per-row norms (the two summary stats are computed
    #     below, AFTER masking padding rows, so nothing is computed over padding). ---
    ce_n = ce_g.norm(dim=1)                                                # (V,)
    mtp_n = mtp_g.norm(dim=1)                                              # (V,)

    model.train(was_training)
    # Mask padding rows: real vocab is 50257 but the matrix is padded to 50304.
    # Apply the mask to EVERY per-row quantity and recompute the two summary
    # statistics on the real rows only, so nothing is computed over padding.
    V_real = 50257 if row_cos.numel() >= 50257 else row_cos.numel()
    row_cos_r = row_cos[:V_real]
    ce_n_r, mtp_n_r = ce_n[:V_real], mtp_n[:V_real]
    support_overlap = F.cosine_similarity(ce_n_r.reshape(1, -1), mtp_n_r.reshape(1, -1)).item()
    mass_r = ce_n_r * mtp_n_r
    opposed_norm_fraction = (mass_r[row_cos_r < 0].sum() / mass_r.sum().clamp(min=1e-12)).item()
    return {
        'global_cos': global_cos,   # aggregate is over the full flattened matrix (padding rows are ~0)
        'per_row_fractions': {t: (row_cos_r.abs() > t).float().mean().item() for t in thresholds},
        'row_cosines': row_cos_r.detach().cpu().numpy(),
        'ce_row_norms': ce_n_r.detach().cpu().numpy(),
        'mtp_row_norms': mtp_n_r.detach().cpu().numpy(),
        'norm_profile_cos': support_overlap,           # cos(a,b): aggregate IF every row were cos=+1; LOW => support divergence
        'opposed_norm_fraction': opposed_norm_fraction, # share of ||g_ce||*||g_mtp|| mass on cos<0 rows; HIGH => cancellation
        'val_loss': ce_loss.item(),
    }


# Interpretation guide (put the printed values in Table/Fig of the camera-ready):
#   norm_profile_cos near 0     + opposed_norm_fraction small  => SUPPORT DIVERGENCE confirmed.
#   norm_profile_cos >> aggregate (e.g. +0.3 vs -0.28) or opposed_norm_fraction large => cancellation does the work.
# Either outcome is publishable; the point is the measurement now DECIDES it instead of inferring.
