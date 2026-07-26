# Part B item B2 — measure_norm_support: the deciding run (n=3 per optimizer, CONFIRMED)

## Measurement (user ran on H100; variant B, 124M, step 1000; seeds 42/123/456 each optimizer)

| Optimizer | n | norm_profile_cos | opposed_norm_fraction | global_cos (aggregate) | per-row \|cos\|>0.3 |
|---|---|---|---|---|---|
| Muon  | 3 | 0.979 ± 0.003 | 0.604 ± 0.026 | -0.163 ± 0.030 | 98.3% |
| AdamW | 3 | 0.977 ± 0.007 | 0.692 ± 0.011 | -0.317 ± 0.020 | 98.6% |

Per-run (seed 42/123/456):
  Muon  npc = 0.978 / 0.976 / 0.982 ; onf = 0.607 / 0.628 / 0.577 ; agg = -0.193 / -0.163 / -0.133
  AdamW npc = 0.985 / 0.971 / 0.974 ; onf = 0.691 / 0.703 / 0.681 ; agg = -0.319 / -0.335 / -0.296

## Verdict: SUPPORT DIVERGENCE REFUTED; CANCELLATION CONFIRMED (both optimizers, n=3)
Instrument guide: norm_profile_cos near 0 + opposed small => support divergence;
                  norm_profile_cos >> aggregate OR opposed large => cancellation.
Measured: norm_profile_cos ~ 0.98 (NOT ~0) => CE and MTP load the SAME rows (support OVERLAPS).
          opposed_norm_fraction 0.60 (Muon) / 0.69 (AdamW) => MAJORITY of the
          ||g_ce||*||g_mtp|| mass sits on rows where the two gradients OPPOSE.
=> The near-zero/negative aggregate is driven by a HIGH-NORM OPPOSED MINORITY of rows
   (cancellation), NOT by disjoint gradient support. Tight error bars; robust.
   AdamW cancels HARDER than Muon (onf 0.69 vs 0.60; aggregate -0.32 vs -0.16):
   optimizer-dependent in MAGNITUDE, identical in MECHANISM. Matches the external
   reviewer's branch-2 prediction, now measured across seeds AND both optimizers.

## Consequence: the paper's mechanism flips. "Support divergence" is retired.
DIES (must be rewritten): abstract "not the cancellation"; intro "support diverges";
  method sec:dirsupport "support divergence"; results sec on support divergence;
  discussion "points to support divergence rather than cancellation"; Fig 1 schematic
  (drew DISJOINT support — the opposite of measured); the "different output rows" claim.
SURVIVES (honest, still strong):
  - Per-row alignment across the lexical majority (median +1, ~98% of rows |cos|>0.3).
  - Per-row vs aggregate DECOUPLING is real; the flattened cosine STILL misreads the
    interaction — but because a high-norm opposed MINORITY (frequent tokens) dominates
    the aggregate, NOT because support diverges.
  - Fig 6 (norm decomposition) is now BETTER supported: the collapse is set by per-row
    NORM structure, and the norm-profile measurement pins the specific channel.
  - Surgery recovers nothing => the detected opposition is not the fixable lever.
REFRAMED THESIS:
  "CE and MTP per-row gradients are aligned in direction across the lexical majority, yet
   the aggregate cosine reads near zero / negative because a high-norm opposed minority,
   concentrated in frequent tokens, dominates the flattened sum. The aggregate cosine is
   a lossy summary that a conflict diagnostic misreads as no-interference; the real
   interference is a norm-weighted opposition it cannot localize, and the gradient surgery
   it triggers does not address the actual next-token degradation."

## Evidence artifact
  fig7_norm_support.pdf — n=3 both optimizers: norm_profile_cos ~1 vs negative aggregate (Panel A),
  opposed_norm_fraction > 0.5 (Panel B). analysis/norm_support_table.md, norm_support_summary.json.
