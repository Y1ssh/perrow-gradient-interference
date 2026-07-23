"""
PCGrad gradient surgery for multi-loss training.
Reference: Yu et al. 2020, "Gradient Surgery for Multi-Task Learning"

Symmetric variant: when CE and MTP gradients conflict at a parameter,
BOTH are projected to remove the conflicting component.
Applied per-parameter (each weight matrix independently).
"""

import torch


def pcgrad_surgery(ce_grads, mtp_grads):
    """
    Apply PCGrad to per-task gradients.

    Args:
        ce_grads:  dict {param_name: tensor} — CE loss gradients
        mtp_grads: dict {param_name: tensor} — MTP loss gradients

    Returns:
        dict {param_name: tensor} — surgered combined gradients
    """
    result = {}
    for name in ce_grads:
        g_ce = ce_grads[name]
        g_mtp = mtp_grads[name]

        g_ce_flat = g_ce.reshape(-1).float()
        g_mtp_flat = g_mtp.reshape(-1).float()

        dot = torch.dot(g_ce_flat, g_mtp_flat)

        if dot < 0:
            # Conflict: project both
            g_ce_proj = g_ce - (dot / g_mtp_flat.norm().square().clamp(min=1e-8)) * g_mtp
            g_mtp_proj = g_mtp - (dot / g_ce_flat.norm().square().clamp(min=1e-8)) * g_ce
            result[name] = g_ce_proj + g_mtp_proj
        else:
            result[name] = g_ce + g_mtp

    return result


if __name__ == '__main__':
    print("PCGrad self-test")
    # Conflicting gradients
    ce = {'w': torch.tensor([1.0, 0.0]), 'b': torch.tensor([0.5])}
    mtp = {'w': torch.tensor([-1.0, 0.0]), 'b': torch.tensor([0.3])}
    out = pcgrad_surgery(ce, mtp)
    assert out['w'].shape == ce['w'].shape, "Shape mismatch"
    # w should differ from naive sum (conflict)
    naive_w = ce['w'] + mtp['w']
    assert not torch.allclose(out['w'], naive_w), "Conflicting should differ from sum"
    # b should equal naive sum (no conflict)
    assert torch.allclose(out['b'], ce['b'] + mtp['b']), "Non-conflicting should equal sum"
    print("  ✓ All checks passed")
