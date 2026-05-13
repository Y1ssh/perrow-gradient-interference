"""
Scatter (Version B): per-row gradient redistribution at lm_head.

For lm_head.weight rows where CE and MTP gradients conflict:
  1. Zero out the MTP gradient on those rows
  2. Scale up MTP gradient on non-conflicting rows to preserve
     total MTP gradient magnitude

All other model parameters: unmodified ce_grad + mtp_grad sum.
"""

import torch


def scatter_redistribute(ce_grads, mtp_grads, lm_head_name):
    """
    Scatter redistribution at lm_head.

    Args:
        ce_grads:      dict {name: tensor}
        mtp_grads:     dict {name: tensor}
        lm_head_name:  str — exact parameter name for lm_head weight

    Returns:
        dict {name: tensor} — surgered gradients
    """
    result = {}
    for name in ce_grads:
        g_ce = ce_grads[name]
        g_mtp = mtp_grads[name]

        if name == lm_head_name:
            V, D = g_ce.shape

            # Per-row cosine
            dots = (g_ce * g_mtp).sum(dim=1)
            ce_norm = g_ce.norm(dim=1).clamp(min=1e-8)
            mtp_norm = g_mtp.norm(dim=1).clamp(min=1e-8)
            cos = dots / (ce_norm * mtp_norm)

            conflict_mask = cos < 0
            non_conflict = ~conflict_mask

            # Save total MTP magnitude
            total_mag = g_mtp.norm()

            # Zero conflicting rows
            g_mtp_mod = g_mtp.clone()
            g_mtp_mod[conflict_mask] = 0

            # Scale up non-conflicting to preserve total magnitude
            remaining_mag = g_mtp_mod.norm()
            if remaining_mag > 1e-8 and non_conflict.any():
                scale = total_mag / remaining_mag
                g_mtp_mod[non_conflict] = g_mtp_mod[non_conflict] * scale

            result[name] = g_ce + g_mtp_mod
        else:
            result[name] = g_ce + g_mtp

    return result


if __name__ == '__main__':
    print("Scatter self-test")
    V, D = 1000, 768
    g_ce = torch.randn(V, D)
    g_mtp = torch.randn(V, D)

    # Force first 600 rows to conflict
    g_mtp[:600] = -g_ce[:600].abs()

    total_mag_before = g_mtp.norm().item()

    grads = {'lm_head': g_ce, 'other': torch.randn(D)}
    mtp_grads = {'lm_head': g_mtp, 'other': torch.randn(D)}

    out = scatter_redistribute(grads, mtp_grads, 'lm_head')
    g_mtp_result = out['lm_head'] - g_ce  # extract modified MTP component

    # Conflicting rows zeroed
    assert g_mtp_result[:600].abs().max() < 1e-6, "Conflicting rows should be zero"

    # Total magnitude preserved
    result_mag = g_mtp_result.norm().item()
    assert abs(result_mag - total_mag_before) / total_mag_before < 0.01, \
        f"Magnitude not preserved: {result_mag:.4f} vs {total_mag_before:.4f}"

    # Other params pass-through
    assert torch.allclose(out['other'], grads['other'] + mtp_grads['other'])
    print(f"  Magnitude preserved: {result_mag:.4f} ≈ {total_mag_before:.4f} ✓")
    print("  ✓ All checks passed")
