"""
Per-row Gram-Schmidt gradient surgery at lm_head.

For each row of lm_head.weight (V rows of D dimensions):
  If CE and MTP gradients conflict (negative dot product):
    Project MTP gradient onto orthogonal complement of CE gradient.
  Else:
    Keep MTP gradient unchanged.

All other model parameters: unmodified ce_grad + mtp_grad sum.
"""

import torch


def gs_per_row(ce_grads, mtp_grads, lm_head_name):
    """
    Per-row Gram-Schmidt projection at lm_head.

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
            # Shape: (V, D) — per-row projection
            dots = (g_ce * g_mtp).sum(dim=1)                       # (V,)
            ce_norm_sq = (g_ce * g_ce).sum(dim=1).clamp(min=1e-8)  # (V,)
            conflict_mask = dots < 0                                # (V,)

            proj_coeff = (dots / ce_norm_sq).unsqueeze(1)           # (V, 1)

            g_mtp_proj = g_mtp.clone()
            g_mtp_proj[conflict_mask] = (
                g_mtp[conflict_mask]
                - proj_coeff[conflict_mask] * g_ce[conflict_mask]
            )
            result[name] = g_ce + g_mtp_proj
        else:
            result[name] = g_ce + g_mtp

    return result


if __name__ == '__main__':
    print("GS per-row self-test")
    V, D = 1000, 768
    g_ce = torch.randn(V, D)
    g_mtp = torch.randn(V, D)

    # Force some rows to conflict
    g_mtp[:500] = -g_ce[:500] + 0.1 * torch.randn(500, D)

    grads = {'lm_head': g_ce, 'other': torch.randn(D, D)}
    mtp_grads = {'lm_head': g_mtp, 'other': torch.randn(D, D)}

    out = gs_per_row(grads, mtp_grads, 'lm_head')

    # Check conflicting rows got projected
    dots_before = (g_ce[:500] * g_mtp[:500]).sum(dim=1)
    dots_after = (g_ce[:500] * (out['lm_head'][:500] - g_ce[:500])).sum(dim=1)
    # After projection, mtp component should have non-negative dot with ce
    assert (dots_after >= -1e-4).all(), "Projected rows should not conflict"

    # Non-conflicting rows unchanged
    non_conflict = (g_ce[500:] * g_mtp[500:]).sum(dim=1) >= 0
    nc_idx = torch.where(non_conflict)[0] + 500
    if len(nc_idx) > 0:
        expected = g_ce[nc_idx] + g_mtp[nc_idx]
        assert torch.allclose(out['lm_head'][nc_idx], expected, atol=1e-5)

    # Other params pass-through
    assert torch.allclose(out['other'], grads['other'] + mtp_grads['other'])
    print("  ✓ All checks passed")
