---
type: claim-ledger
work_id: perrow-gradient-interference
paper: "Option 1 — Diagnostic / Measurement pivot"
created: 2026-07-07
rule: every asserted sentence is SUPPORTED (with exact evidence) or CUT
numbers_source: committed JSONs, final_val_loss aggregation, recomputed 2026-07-07
---

# Claim Ledger — Diagnostic Pivot

**Title (locked):** *Aligned but Apart: Why Cosine Conflict Diagnostics Misread
Auxiliary-Loss Interference at the Output Projection*

**Locus wording:** use **"output projection"** / **"shared output head"**, NOT "at the
tied output embedding" as the causal locus — A4 shows the alignment persists with the head
**untied** (99.3%), so the effect is a property of the output-projection gradient
(hᵀ·dL/dlogits), not of weight-tying. Tying is a setting, not the cause. (Mention tying as
the default config; report A4 as the control that rules it out.)

**One-sentence thesis (SUPPORTED, no new runs):** At a shared output projection, the per-row
gradients of a next-token loss and an auxiliary multi-token loss are near-perfectly *aligned
in direction* (median per-row cosine ≈ +1), while the *aggregate* cosine is ≈0 — so
cosine-based conflict diagnostics, standard in multi-task optimization, misread this
interference; the operative axis is gradient **support** (which rows carry magnitude), not
direction.

---

## A. SUPPORTED claims (each with exact evidence)

| # | Claim (as it will appear) | Evidence (committed data) | Numbers |
|---|---|---|---|
| A1 | Per-row CE vs MTP gradients are near-perfectly aligned in direction | A1 `row_cosines`, step 1000, 50,304 rows | median **+1.000**, cos<0 in **0.33%**, \|cos\|>0.3 in **98.38%** |
| A2 | A calibrated control confirms the instrument (an unrelated loss reads ~0) | A3 CE-vs-L1 | per_row_0.3 = **0.42%** (vs 98.38%) |
| A3 | The aggregate cosine is ≈0 despite per-row alignment (the paradox) | A1 trajectory | global_cos ≈ **−0.08** at step 1000 while per_row_0.3 = 0.98 |
| A4 | The decoupling **emerges during training** (both start aligned) | A1 21-point trajectory | step 1: global **+0.99**, per_row 0.998 → step 200: global **−0.17**, per_row 0.97 |
| A5 | Aggregate≈0 forces disjoint support (math + measurement) | logic: per-row cos≈+1 ⇒ cancellation impossible ⇒ magnitude must sit on different rows | (argued; per-row-norm figure is the only *upgrade* that needs a run — mark as future work) |
| A6 | Adding shared-head MTP degrades next-token CE in this testbed | Phase B A vs B | A=**3.9851±0.014 (n=5)**, B=**4.3767±0.021 (n=3)**, Δ=**+0.392** |
| A7 | Detaching the aux from the head (stop-grad) makes it **worse** | Phase B B_sg | B_sg=**4.4927±0.010 (n=3)**, Δ=**+0.508** |
| A8 | Neither standard nor tuned capacity-separated aux recovers A within noise | Phase B gnce + phase_d (10,9) | std G_nce=**4.0241±0.021 (n=5)**, Δ+0.039; tuned (10,9) n=3 mean=**4.0026±0.016**, Δ+0.018 (>1σ); NextLat=**4.0365**, Δ+0.051 |
| A9 | Alignment persists with the head **untied** → not a tying artifact | A4 untied | per_row_0.3 = **99.3%** (vs 98.7% tied at ~step 500) |
| A10 | The pattern is not unique to the head — it is general across matrices | A1 per_layer_snapshots, 74 matrices | head is the max, but attn v_proj/o_proj also 91–93% with **negative** global cos |
| A11 | The head is the *consequential* locus because it directly produces the logits | framing tied to A10 | (interpretation, stated as such) |
| A12 | Rare tokens show higher per-row alignment than common ones (both optimizers) | A1 + A2 token_freq_correlation | Muon rare **0.987** vs common **0.952**; AdamW **1.00** vs **0.949** |
| A13 | Conflict-removal surgery is a no-op here (little to remove) — *scoped, caveated* | Phase C | with ~0.3% conflicting rows, PCGrad/GS/Scatter ≈ B±noise; **state n=1 + two-pass drift + PCGrad-all-params caveats inline** |
| A14 | The pattern qualitatively holds at 350M (single illustrative seed) | 350M r4 | A=3.7516, B=4.1291, Δ+0.378 (**n=1**); per_row_0.3 ≈ 98.9% |
| A15 | Effect appears under both Muon and AdamW | A1 vs A2 (disclose LR/wd mismatch) | both show per_row ≈0.95–0.99, global ≈0 |
| A16 | Complements Godey & Artzi (2026): single-loss norm-space → we do multi-loss direction-space | related work | (positioning, no number) |

## B. CUT claims (drop entirely — would need runs this pivot forgoes)

| # | Cut claim | Why cut | Would need |
|---|---|---|---|
| B1 | "MTP **causes** the next-token degradation" (causal) | causation not established | separate-head MTP |
| B2 | "The effect is about **multi-token prediction** as practiced" | this is shared-head multi-offset, not separate-head MTP | separate-head MTP |
| B3 | "**Amplifies with scale** / scale-invariant / super-linear" | 350M is n=1 | 350M n=3 + 124M CE-only baseline |
| B4 | "**Surgery proves** the interference is per-row" | n=1, two-pass drift, PCGrad scope confound | matched-seed single-pass surgery, n=3 |
| B5 | "G_nce **recovers** the baseline" | neither std nor tuned recovers within noise when aggregated | (nothing — just don't claim it) |
| B6 | The non-replicating **T4 numbers** ("118% damage reduction", "beats CE-only") | never replicated on H100 | (delete from code + docs) |
| B7 | "PRDR / discrepancy ratio" as a reported metric | ill-conditioned, dropped | (report raw per-row distribution instead) |

## C. Framing rules for the whole paper

1. Language is **descriptive/associational**, never causal: "the degradation coincides with
   aligned-but-support-mismatched gradients," never "support mismatch causes."
2. Scope stated up front and in the conclusion: **shared single output head, ≤350M,
   undertrained (R≈3.9), small batch (16K tok/step)**.
3. Separate-head MTP, 350M n=3, matched-seed surgery, per-row-norm measurement are named
   explicitly as **future work that would extend the claim** — turning the gaps into a
   roadmap instead of holes.
4. Every number traces to a committed JSON (this ledger is the source of truth).

## D. Numbers appendix (authoritative, recomputed 2026-07-07)

- A=3.9851±0.0138 (n=5) · B=4.3767±0.0209 (n=3, Δ+0.392) · B_sg=4.4927±0.0101 (n=3, Δ+0.508)
- G_nce std=4.0241±0.0207 (n=5, Δ+0.039) · NextLat=4.0365±0.0179 (n=3, Δ+0.051)
- G_nce tuned (10,9) n=3=4.0026±0.0156 (Δ+0.018; best seed 3.9891)
- 350M r4 (n=1): A=3.7516, B=4.1291, Δ+0.378
- Per-row (A1 step 1000, 50,304 rows): median +1.000, cos<0 0.33%, |cos|>0.3 98.38%
- Control (A3): per_row_0.3 0.42%
- Committed result files: 66 JSONs
