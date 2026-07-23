"""
GPT-2 Medium (350M) for scale validation experiments.

Same architecture as gpt2.py but with larger config:
    n_layer=24, n_head=16, n_embd=1024
    ~354M parameters

Usage:
    from model.gpt2_medium import GPT2Medium

    model = GPT2Medium().cuda()
    logits, intermediates = model(input_ids)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt2 import GPT2, GPT2Config


def GPT2Medium():
    """Create a GPT-2 Medium (350M) model."""
    config = GPT2Config(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        vocab_size=50304,
        block_size=1024,
        bias=False,
    )
    return GPT2(config=config)


if __name__ == '__main__':
    import torch
    print("=" * 60)
    print("  GPT2-Medium (350M) self-test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2Medium().to(device)

    # Param count
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    print(f"\n  Total params: {total:,} ({total / 1e6:.1f}M)")
    assert 350_000_000 < total < 360_000_000, f"Expected ~354M, got {total/1e6:.1f}M"

    # Forward pass
    idx = torch.randint(0, 50304, (2, 64), device=device)
    logits, intermediates = model(idx)
    print(f"  Logits:        {tuple(logits.shape)}")
    print(f"  Intermediates: {len(intermediates)} layers")
    assert logits.shape == (2, 64, 50304)
    assert len(intermediates) == 24

    # Muon param groups
    groups = model.get_muon_param_groups()
    muon_n = sum(p.numel() for p in groups[0]['params'])
    adam_n = sum(p.numel() for p in groups[1]['params'])
    print(f"  Muon params:   {muon_n:,} ({muon_n / 1e6:.1f}M)")
    print(f"  Adam params:   {adam_n:,} ({adam_n / 1e6:.1f}M)")

    print(f"\n  All checks passed. ✓")
    print("=" * 60)
