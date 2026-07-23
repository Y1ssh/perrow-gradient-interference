# Part B item B2 — measure_norm_support result (the deciding run)

## Measurement (user ran on H100; seed 42, Muon, 124M, variant B, step 1000)
    global_cos (aggregate):     -0.1634
    per_row |cos|>0.3:           98.6%
    norm_profile_cos:            0.9790
    opposed_norm_fraction:       0.5744
    val_loss (pure CE):          5.498
    (committed a1 step-1000 global_cos was -0.082; this fresh run -0.163 — same sign,
     ~2x magnitude, normal single-run variation)

## Verdict: SUPPORT DIVERGENCE REFUTED; CANCELLATION CONFIRMED (Muon, n=1)
Instrument's own guide:
  norm_profile_cos near 0 + opposed_norm_fraction small => SUPPORT DIVERGENCE
  norm_profile_cos >> aggregate OR opposed_norm_fraction large => CANCELLATION
Measured: norm_profile_cos = 0.979 (NOT near 0 => support is ~fully OVERLAPPING, not disjoint),
          norm_profile_cos - aggregate = 1.142 cosine units of collapse from opposition,
          opposed_norm_fraction = 0.574 (majority of ||g_ce||*||g_mtp|| mass on cos<0 rows).
=> The near-zero/negative aggregate is driven by a HIGH-NORM OPPOSED MINORITY (cancellation),
   NOT by disjoint gradient support. This is the external reviewer's "branch 2", decisively.

## Consequences for the manuscript
DIES:
  - "Support divergence" as THE mechanism (refuted: 0.98, not ~0).
  - Abstract/conclusion "not the cancellation of opposing per-row gradients" (now measured false).
  - Fig 1 schematic (draws CE/MTP magnitude on DISJOINT rows — the opposite of measured).
SURVIVES (honest reframe):
  - Per-row alignment across the lexical majority (median +1, ~99.7% of rows by COUNT).
  - Per-row vs aggregate DECOUPLING is real; flattened cosine STILL misreads — but because a
    high-norm opposed MINORITY (frequent tokens) dominates the aggregate, not support divergence.
  - MTP degrades CE; surgery recovers nothing => the detected opposition is a red herring.
  - Fig 6 (norm decomposition) + Fig 2 (emergence) survive; Fig 6 now BETTER supported.
REFRAMED THESIS (honest):
  "CE/MTP per-row gradients are aligned across the lexical majority, but a high-norm opposed
   minority concentrated in frequent tokens drags the aggregate negative. The flattened cosine
   cannot resolve this structure, and the gradient surgery it triggers does not address the
   actual next-token degradation."

## Open robustness question (why n=1 is not yet enough to publish the NEW claim)
  - Support divergence is refuted even at n=1 (0.98 is nowhere near 0 — robust to noise).
  - But opposed_norm_fraction (0.574) and global_cos (-0.163 vs committed -0.082) show run-to-run
    variation. To publish the CANCELLATION mechanism confidently, confirm across seeds 42/123/456
    (Muon) and run the AdamW branch (aggregate ~-0.28; reviewer expects even stronger cancellation).
  - Each run is ~2.5 min on the H100. 3 Muon seeds + 1-3 AdamW seeds = ~15 min total.

## Title note
  "Aligned but Apart": "Aligned" (per-row +1) still holds. "Apart" originally meant "apart in
  support" — now false. "Apart" can be reinterpreted as "a high-norm minority pulls apart the
  aggregate", or the title may need a tweak. Flag for the rewrite.
