"""
GPT-2 124M for per-row gradient interference experiments.

Architecture: 12 layers, 12 heads, 768 dim, 50304 vocab, 1024 context.
Separate Q/K/V/O projections (Muon official recommendation).
Pre-norm with F.rms_norm (no learnable norm params).
Weight-tied embedding ↔ lm_head.

Usage:
    from model.gpt2 import GPT2, GPT2Config
    from muon import SingleDeviceMuonWithAuxAdam

    model = GPT2().cuda()
    optimizer = SingleDeviceMuonWithAuxAdam(model.get_muon_param_groups())

    logits, intermediates = model(input_ids)
    # intermediates[i] = layer i output (before final norm)
    # Used by G_nce for layer-split MTP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304      # padded to multiple of 64
    block_size: int = 1024
    bias: bool = False


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Separate projections — Muon works better on individual Q/K/V
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj._SCALE_INIT = 1  # scale down at init (residual path)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj._SCALE_INIT = 1  # scale down at init (residual path)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.n_embd = config.n_embd

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (self.n_embd,)))
        x = x + self.mlp(F.rms_norm(x, (self.n_embd,)))
        return x


class GPT2(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = GPT2Config()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output head (tied with wte)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections: std = 0.02 / sqrt(2 * n_layer)
        scaled_std = 0.02 / (2 * config.n_layer) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=scaled_std)

        # Report param counts
        self._print_param_summary()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_param_summary(self):
        # Deduplicate tied weights
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()

        groups = self.get_muon_param_groups()
        muon_n = sum(p.numel() for p in groups[0]['params'])
        adam_n = sum(p.numel() for p in groups[1]['params'])

        print(f"  GPT2-124M initialized:")
        print(f"    Total params:  {total:>12,}  ({total/1e6:.1f}M)")
        print(f"    Muon params:   {muon_n:>12,}  ({muon_n/1e6:.1f}M)  "
              f"[{muon_n/total*100:.0f}%  Q/K/V/O + MLP per layer]")
        print(f"    Adam params:   {adam_n:>12,}  ({adam_n/1e6:.1f}M)  "
              f"[{adam_n/total*100:.0f}%  wte/wpe/lm_head]")

    def forward(self, idx):
        """
        Args:
            idx: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
            intermediates: dict {layer_idx: tensor} — each block's output
                           before final rms_norm. Used for G_nce layer-split.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        intermediates = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            intermediates[i] = x

        x = F.rms_norm(x, (self.config.n_embd,))
        logits = self.lm_head(x)
        return logits, intermediates

    def get_muon_param_groups(self):
        """
        Returns param_groups for SingleDeviceMuonWithAuxAdam.

        Muon: 2D weights inside transformer blocks (Q,K,V,O,fc,proj).
        Adam: everything else (wte, wpe, lm_head — with tied weights deduplicated).
        """
        hidden_matrix = []
        other_params = []
        seen = set()

        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))

            if name.startswith('blocks.') and p.dim() >= 2:
                hidden_matrix.append(p)
            else:
                other_params.append(p)

        return [
            dict(params=hidden_matrix, use_muon=True,
                 lr=0.02, momentum=0.95, weight_decay=0.01),
            dict(params=other_params, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1),
        ]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  GPT2-124M self-test")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2().to(device)
    # Verify forward pass
    idx = torch.randint(0, model.config.vocab_size, (2, 128), device=device)
    logits, intermediates = model(idx)
    print(f"\n  Forward pass:")
    print(f"    Input:          {tuple(idx.shape)}")
    print(f"    Logits:         {tuple(logits.shape)}")
    print(f"    Intermediates:  {len(intermediates)} layers, "
          f"each {tuple(intermediates[0].shape)}")
    # Verify weight tying
    assert model.lm_head.weight is model.wte.weight, "Weight tying broken!"
    print(f"    Weight tying:   ✓")
    # Verify gradient flows through intermediates (needed for G_nce)
    # Simulate: main CE loss + auxiliary loss on layer 8 intermediate
    model.zero_grad()
    idx2 = torch.randint(0, model.config.vocab_size, (2, 128), device=device)
    logits2, inter2 = model(idx2)
    main_loss = logits2.sum()
    aux_loss = inter2[8].sum()  # simulate NCE on layer 8 output
    total_loss = main_loss + aux_loss
    total_loss.backward()
    # Check: gradient reached a param BEFORE layer 8 (proves flow through intermediate)
    test_param = model.blocks[5].attn.q_proj.weight
    assert test_param.grad is not None, "No gradient through intermediate!"
    assert not torch.isnan(test_param.grad).any(), "NaN in gradient!"
    print(f"    Grad flow:      ✓  (aux on layer 8 → grad reaches layer 5)")
    # Verify param groups match Muon format
    groups = model.get_muon_param_groups()
    muon_keys = set(groups[0].keys())
    adam_keys = set(groups[1].keys())
    expected_muon = {'params', 'lr', 'momentum', 'weight_decay', 'use_muon'}
    expected_adam = {'params', 'lr', 'betas', 'eps', 'weight_decay', 'use_muon'}
    assert muon_keys == expected_muon, f"Muon keys mismatch: {muon_keys}"
    assert adam_keys == expected_adam, f"Adam keys mismatch: {adam_keys}"
    print(f"    Param groups:   ✓  (Muon: {len(groups[0]['params'])} tensors, "
          f"Adam: {len(groups[1]['params'])} tensors)")
    # Verify no duplicates across groups
    all_ids = [id(p) for g in groups for p in g['params']]
    assert len(all_ids) == len(set(all_ids)), "Duplicate params!"
    print(f"    No duplicates:  ✓")
    print(f"\n  All checks passed. ✓")
    print("=" * 60)
