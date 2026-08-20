# Triage of the external technical review (2026-07-07)

Verdict on the review: **high quality and largely correct.** Its central mathematical point is
airtight and I verified it on our own data. Splitting into what I fixed now, what is Part B
(the norm/cancellation reframe + the one run), and what I judge does NOT need action.

## VERIFIED — the review's core math is right (checked on committed data)
Identity: `agg = Σ_i a_i b_i c_i / (||G_ce|| ||G_mtp||)` with `a_i,b_i≥0`.
- If all per-row `c_i ≥ 0` then `agg ≥ 0`. A **negative** aggregate REQUIRES opposed rows (`c_i<0`)
  carrying weight. Pure support divergence can only push `agg → 0⁺`, never below 0.
- Our measured aggregates: **Muon −0.082, AdamW −0.285** (both negative; frac_neg 0.33% / 0.28%).
- Therefore a *strict* "not cancellation" claim is **false**: some opposed high-norm mass must be
  present. Its magnitude is UNMEASURED (needs per-row norms → the Part-B run).
- The review is also right that `cos(a,b)` (norm-profile cosine) is the *sharper* test than our
  current flat-norm mean (equal-weights counterfactual). Our shipped instrument now returns exactly
  `norm_profile_cos` + `opposed_norm_fraction`.

## FIXED THIS TURN (the 3 auditor findings; all presentation/code, not conclusions)
1. `measure_norm_support.py` now actually masks the 47 padding rows (was a dead `real` variable);
   renamed the returned key to `norm_profile_cos` with the cleaner interpretation.
2. discussion.tex future-work ref corrected `sec:aligned → sec:emergent` (where the identity + Fig 6 live).
3. TEN_OUT_OF_TEN.md softened: Fig 6 *partially* addresses the interp ask (proves norm-weighting),
   does NOT separate support-divergence from opposed-minority; that needs the run.

## PART B STATUS (2026-07-23) — B1/B2/B3 DONE; B2 RESULT INVERTED THE REFRAME
B2 was run (n=3 seeds per optimizer, both Muon and AdamW). Result: **norm-profile cosine ≈ 0.98**
(NOT ≈0), **opposed-norm fraction 0.60 (Muon) / 0.69 (AdamW)**. This REFUTES support divergence
(the losses load the SAME rows) and CONFIRMS cancellation (a high-norm opposed minority carries the
mass). So B1's anticipated "both operate, proportion unmeasured" reframe was replaced by the stronger,
measured statement: **norm-weighted cancellation by a high-norm opposed minority; support does not
diverge.** The whole manuscript mechanism was rewritten accordingly (16 spots across 5 files). See
docs/RESULT_B2_norm_support.md and Figure 7 (fig:normsupport).

B1. ✅ DONE — every strict "not cancellation" / "rather than cancellation" removed, but reframed to
    the MEASURED cancellation (not the hedged "both operate"), because B2 settled it.
B2. ✅ DONE — ran on H100, n=3 per optimizer. norm_profile_cos 0.979/0.977, opposed_norm_fraction
    0.604/0.692. Converted the headline from inference to a direct result.
B3. ✅ DONE — Sec-3.3 toy replaced with a 3-aligned-low-norm + 1-opposed-high-norm construction where
    every per-row cosine is DEFINED (three read +1, one reads −1) and the aggregate goes negative from
    the one high-norm opposed row. Matches the measured mechanism (b), no support divergence needed.
B4. **Add the gradient-structure explanation** for alignment (both row-grads live in span{h_t}; rare
    tokens = few contexts = low rank = near-parallel). Turns 3 scattered findings (=+1 spike, rare-token
    ordering, CE-only 94.5%) into one mechanism AND appropriately deflates "alignment is surprising".
B5. **Report the CE-vs-L1 control's AGGREGATE** (not just per-row 0.42%) to show aggregate≈0 is
    uninformative — the per-row metric does the work.
B6. **Fig 1 relabel:** "conflict (artifact)" → the honest failure is "≈0 misread as *no interference*
    when interference is real" (surgery correctly declines at ≈0 and still fails). Also flag the middle
    panel as *hypothesized* norm structure (schematic), not measured.
B7. **Paired test** on shared seeds (42,123,456) as primary for the A-vs-B gap; keep Welch as secondary.
B8. **GradNorm (Chen 2018)** citation — the direct prior "magnitude not direction" MTL work; plus
    position vs CAGrad / Nash-MTL / IMTL. Preempt "per-block≠global cosine is a known flattened-cosine
    caveat".
B9. **Temporal gap:** log per-row/aggregate on the SAME 30k-step run that produces the loss gap
    (currently mechanism@1000 steps, effect@30517 steps, different runs). Needs a run.
B10. Demote stop-grad to a footnote (confounded); soften "capacity is the lever" to suggestive;
     tag every 350M number "illustrative (n=1)"; drop/bury Cohen's d=−23.7.

## JUDGED NOT ACTIONABLE / already handled
- "Didn't run the script" — correct, that's B2; it's a compute decision the user owns.
- Abstract hedge density — partly already consolidated; will revisit in B1.
- Muon post-transform gradient bug worry — our measurement takes autograd.grad on lm_head BEFORE any
  optimizer transform, so the logged gradient is the raw loss gradient (no Muon orthogonalization). Not a bug.

## SECOND EXTERNAL REVIEW (2026-07-23) — pre-submission, full response
A second detailed reviewer read the post-flip paper. Verdict: measurement real, scope-honesty
above average, core point publishable, but two "blocking" items. Each was checked against the
committed arrays/code before acting (no guessing).

R1. **Active-row denominator** — CORRECT and verified on arrays. Because CE and shared MTP read the
    SAME logits, a row with no target in the batch has g_mtp = 0.75·g_ce EXACTLY (measured norm ratio
    among near-parallel rows = 1.3329 ≈ 1/0.75). 75% of rows are parallel by construction (cos>1-1e-6),
    only ~12% are active. Restricted to active rows the median cosine is 0.54 (not +1.00), 2.8% opposed,
    86% >0.3. ✅ ADDED to §4.1 + fig1 caption as the honest denominator. CRUCIAL: the mass-weighted
    mechanism is UNCHANGED (parallel rows carry only ~0.6% of the mass) → norm-profile cos 0.98 and
    opposed-fraction 0.60/0.69 identical over all rows vs active rows. Sharper, not weaker.
R2. **Mixture-optimum alternative** — LEGITIMATE, the one true blocker; needs compute. A single head on
    1/0.5/0.25-weighted targets converges to a weighted mixture and pays KL on next-token; predicts a gap
    in the right ballpark WITHOUT gradient geometry. ⏳ OPEN → experiments/phase_e_mtp_weight_sweep.py
    (weight sweep + t+1/t+2/t+3 CE + output entropy). Verdict: gap ∝ weight & entropy↑ → mixture (reframe);
    gap persists at tiny weight → interference. Either way the cosine-diagnostic point stands.
R3. **Three stale "support appears to compete" sentences** — ✅ already fixed in the prior line-by-line pass.
R4. **"Nothing to project away" logic backwards** — ✅ FIXED (5 spots: main/results/appendix/related×2).
    The opposition is negligible by count (0.3%) but DOMINANT by mass (60-69%); per-row Gram-Schmidt
    removes exactly it and nothing moves → "mass-dominant yet causally inert" (stronger claim, matches data).
R5. **Soften "measure the wrong quantity"** — ✅ FIXED → "not wrong about the update, but uninformative
    about WHERE the interference lives" (abstract + fig0 caption).
R6. **Exact factorization** — ✅ ADDED + verified: aggregate = ρ_norm × ⟨cos⟩_mass = 0.98×(-0.082) [Muon].
    Weights sum to exactly the norm-profile cosine. Equation added to §4.2.
R7. **Decode the ~166 opposed rows** — ⏳ OPTIONAL. Prior check: frequency link only MODERATE
    (Spearman token_id vs mass = -0.59); table would be honest but must not overclaim "frequent tokens".
R9. **Statistics** — ✅ FIXED: p≈5.5e-9 → p<1e-4; Welch sign clarified (CE-only minus MTP); MDE added
    (~0.05 nats, headline is 8× floor, tuned-aux below it); Holm note (4/5 survive, only tuned fails);
    Muon-vs-AdamW opposed-fraction test added (Welch t=5.4, p=0.015, genuinely optimizer-dependent).
R10. **Citations** — ✅ ADDED (all verified on arXiv): Kurin 2201.04122, Xin 2209.11379, Gao 1907.12009;
     Godey "different tensor" precision note added.
R11. **Define G_nce / NextLat** — ✅ inline definitions added at first mention.
R13. **Presentation** — ✅ title line-break fixed (no more "Mis-/read"); fig0 caption flagged
     "schematic; illustrative proportions, not plotted data"; abstract leads with active-row honesty.

Compiles clean: 15 pages, 0 undefined refs, 0 em-dashes, all citations resolve.
(Page count as of this triage; the paper is 28 pp after subsequent revisions.)
