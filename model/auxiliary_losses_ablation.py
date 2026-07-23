"""
GNCELossAblation — GNCELoss with configurable hyperparameters for ablation.

Standalone copy of GNCELoss with extra constructor args:
    alpha_mult:  scales the auto-calibrated alpha (default 0.3)
    nce_weights: weights for nce2 and nce3 (default (0.5, 0.25))
    neg_type:    'roll' (deterministic) or 'random' (stochastic) negatives

Usage:
    from model.auxiliary_losses_ablation import GNCELossAblation

    gnce = GNCELossAblation(
        n_embd=768, layers=(9, 8), K=16,
        alpha_mult=1.0, nce_weights=(1.0, 0.5), neg_type='random',
    ).cuda()

    total_loss, info = gnce(ce_loss, intermediates, batch, model.wte)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNCELossAblation(nn.Module):
    """
    Layer-split NCE auxiliary loss with configurable hyperparameters.

    Same architecture as GNCELoss but with ablation-friendly constructor:
        alpha_mult:  multiplier for alpha auto-calibration (0.3 = default)
        nce_weights: (w2, w3) weights for horizon 2 and 3 NCE losses
        neg_type:    'roll' = deterministic roll, 'random' = random permutation

    Params: 2 × LayerNorm(n_embd) = 3,072 trainable parameters (at n_embd=768).
    """

    def __init__(self, n_embd=768, layers=(9, 8), K=16, tau=0.1,
                 alpha_mult=0.3, nce_weights=(0.5, 0.25), neg_type='roll'):
        super().__init__()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ln_3 = nn.LayerNorm(n_embd)
        self.n_embd = n_embd
        self.layers = layers
        self.K = K
        self.tau = tau
        self.alpha = None
        self.alpha_mult = alpha_mult
        self.nce_weights = nce_weights
        assert neg_type in ('roll', 'random'), f"neg_type must be 'roll' or 'random', got {neg_type!r}"
        self.neg_type = neg_type

    def nce_loss(self, hidden, target_emb):
        """
        Noise Contrastive Estimation loss.

        Args:
            hidden:     (B, T, D) — intermediate layer output
            target_emb: (B, T, D) — target token embeddings (DETACHED)

        Returns: scalar loss
        """
        B, T, D = hidden.shape

        K_eff = min(self.K, B - 1)
        if K_eff < 1:
            return hidden.new_tensor(0.0)

        h_norm = F.normalize(hidden, dim=-1)
        t_norm = F.normalize(target_emb, dim=-1)

        # Positive
        pos_cos = (h_norm * t_norm).sum(dim=-1) / self.tau  # (B, T)

        # Negatives
        neg_loss = torch.zeros_like(pos_cos)

        if self.neg_type == 'roll':
            for k in range(1, K_eff + 1):
                neg_emb = t_norm.roll(k, dims=0)
                neg_cos = (h_norm * neg_emb).sum(dim=-1) / self.tau
                neg_loss = neg_loss + F.logsigmoid(-neg_cos)
        else:  # 'random'
            arange_B = torch.arange(B, device=hidden.device)
            for k in range(K_eff):
                # Random shift in [1, B-1] per example — guarantees no self-negatives
                shifts = torch.randint(1, B, (B,), device=hidden.device)
                neg_indices = (arange_B + shifts) % B
                neg_emb = t_norm[neg_indices]
                neg_cos = (h_norm * neg_emb).sum(dim=-1) / self.tau
                neg_loss = neg_loss + F.logsigmoid(-neg_cos)

        neg_loss = neg_loss / K_eff

        loss = -(F.logsigmoid(pos_cos) + neg_loss).mean()
        return loss

    def forward(self, ce_loss, intermediates, targets, wte):
        """
        Compute total loss = CE + alpha * NCE(intermediate layers).

        Args:
            ce_loss:       scalar — already-computed CE loss
            intermediates: dict {int: tensor} — each block's output
            targets:       (B, T) token indices
            wte:           nn.Embedding — model's token embedding layer

        Returns:
            total_loss: scalar
            info:       dict with component losses and config for logging
        """
        h2 = self.ln_2(intermediates[self.layers[0]])[:, :-2]
        h3 = self.ln_3(intermediates[self.layers[1]])[:, :-3]

        with torch.no_grad():
            tgt2 = wte(targets[:, 2:])
            tgt3 = wte(targets[:, 3:])

        nce2 = self.nce_loss(h2, tgt2)
        nce3 = self.nce_loss(h3, tgt3)
        nce_combined = self.nce_weights[0] * nce2 + self.nce_weights[1] * nce3

        if self.alpha is None:
            with torch.no_grad():
                ratio = ce_loss.item() / max(nce_combined.item(), 1e-6)
                self.alpha = max(0.01, min(self.alpha_mult * ratio, 10.0))

        total_loss = ce_loss + self.alpha * nce_combined

        info = {
            'ce_loss': ce_loss.item(),
            'nce_loss': nce_combined.item(),
            'alpha': self.alpha,
            'nce2': nce2.item(),
            'nce3': nce3.item(),
        }
        return total_loss, info

    def get_aux_param_groups(self):
        """Returns Adam param groups for Muon optimizer."""
        return [dict(
            params=list(self.parameters()),
            use_muon=False,
            lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0,
        )]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.gpt2 import GPT2

    print("=" * 60)
    print("  GNCELossAblation self-test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2().to(device)
    V = model.config.vocab_size
    batch = torch.randint(0, V, (4, 64), device=device)

    # Test all ablation configurations
    configs = [
        dict(alpha_mult=0.3, nce_weights=(0.5, 0.25), neg_type='roll',
             label='default'),
        dict(alpha_mult=1.0, nce_weights=(0.5, 0.25), neg_type='roll',
             label='alpha=1.0'),
        dict(alpha_mult=0.3, nce_weights=(1.0, 0.5), neg_type='roll',
             label='weights=(1.0,0.5)'),
        dict(alpha_mult=0.3, nce_weights=(0.5, 0.25), neg_type='random',
             label='random negatives'),
        dict(alpha_mult=0.3, nce_weights=(0.5, 0.25), neg_type='roll',
             label='layers=(7,8)', layers=(7, 8)),
        dict(alpha_mult=0.3, nce_weights=(0.5, 0.25), neg_type='roll',
             label='K=4', K=4),
    ]

    for cfg in configs:
        label = cfg.pop('label')
        layers = cfg.pop('layers', (9, 8))
        K = cfg.pop('K', 16)

        gnce = GNCELossAblation(n_embd=768, layers=layers, K=K, **cfg).to(device)

        model.zero_grad()
        logits, intermediates = model(batch)
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
        total, info = gnce(ce, intermediates, batch, model.wte)
        total.backward()

        print(f"\n  {label}:")
        print(f"    CE={info['ce_loss']:.4f}  NCE={info['nce_loss']:.4f}  "
              f"alpha={info['alpha']:.4f}  total={total.item():.4f}")

        # Verify grad flows
        assert model.blocks[layers[0]].attn.q_proj.weight.grad is not None
        print(f"    Grad flow to block {layers[0]}: ✓")

    # Verify param count (always 3072 regardless of config)
    gnce_test = GNCELossAblation(n_embd=768)
    n = sum(p.numel() for p in gnce_test.parameters())
    assert n == 3072, f"Params: {n} != 3072"
    print(f"\n  Param count: {n:,} ✓")

    # Verify param group format
    groups = gnce_test.get_aux_param_groups()
    expected = {'params', 'lr', 'betas', 'eps', 'weight_decay', 'use_muon'}
    assert set(groups[0].keys()) == expected
    assert groups[0]['use_muon'] is False
    print(f"  Param groups: ✓")

    print(f"\n  All checks passed. ✓")
    print("=" * 60)
