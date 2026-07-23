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

## PART B (do after this, per user) — the reframe + the one run
B1. **Soften/kill every strict "not cancellation" / "rather than cancellation".** Live spots:
    main.tex:48 (abstract), discussion.tex:7 and :47-48, results.tex:114 ("no cancellation" reading).
    Reframe to: "support divergence suppresses the aligned contribution; a small high-norm opposed
    mass tips the sign negative — both operate, proportion unmeasured."
B2. **Run `measure_norm_support.py`** on the short diagnostic pass → report `norm_profile_cos` vs the
    actual aggregate and `opposed_norm_fraction`. This converts the headline from inference to result.
    (Needs FineWeb download + local RTX 3050; ~1-3 h; or a GPU host.)
B3. **Replace the Sec-3.3 toy** with a 3-row example where BOTH gradients are nonzero on every row
    (so per-row cos is defined and =+1) with disjoint magnitude → agg→0⁺, plus a 4th opposed high-norm
    row to show the sign going negative. (The review is right the current 2-row toy has all-undefined
    per-row cosines — it shows agg=0 but not "per-row +1 coexisting with agg 0".)
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
