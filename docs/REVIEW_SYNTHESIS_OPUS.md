# Second-Pass Review (Opus 4.8) — "Aligned but Apart"

Five reviewer profiles, run one at a time on `claude-opus-4-8` with a 16K-token budget, against the
**post-first-revision** manuscript. This pass is deeper than the Sonnet pass and the ratings rose.

## Ratings

| Reviewer | Rating | Score |
|---|---|---|
| TMLR Action Editor | Accept-with-minor-revisions | 7/10 |
| Optimization / MTL | Major-revisions | 6/10 |
| LLM-pretraining | Accept-with-minor-revisions | 7/10 |
| Statistician | Accept-with-minor-revisions | 7/10 |
| Interpretability / measurement | Major-revisions | 6/10 |

**All five: shippable to TMLR once the FIX-BY-WRITING items are applied. None requires new runs.**
Every reviewer independently praised: the like-for-like calibrated control, the now-exemplary scope
discipline (measurement-windows paragraph, single-run tags), and the honest G_nce/surgery reporting.

## What I verified in the data before acting (so fixes are grounded, not guessed)

1. **350M comparator (all 5 flagged as undefined) — RESOLVED, and it strengthens the paper.**
   `phase_c_350m_measure.py:149-161` always computes CE-grad vs MTP-grad per-row cosine, *regardless of
   how the model was trained*. So 94.5% (CE-only model) vs 98.9% (MTP-trained model) IS apples-to-apples
   — same pairing, different training. The CE-only 94.5% shows the alignment is intrinsic to the
   output-projection gradient geometry, present even without an MTP objective in training. Clarify, don't drop.

2. **Token-frequency claim (3 reviewers: "metric is saturated") — a non-saturating statistic EXISTS.**
   The JSONs store `mean_abs_cos` per quintile alongside the saturated `frac_above_0.3`. It has real
   headroom and tells the cleaner story:
   | quintile | Muon mean_abs_cos | AdamW mean_abs_cos |
   |---|---|---|
   | rarest 20% | 0.922 | 0.998 |
   | most-common 20% | 0.847 | 0.848 |
   Switch Fig 5 + text to `mean_abs_cos`; the rare>common ordering holds and is no longer read off a ceiling.

3. **"Emerges within ~200 steps" — now precise.** Muon aggregate crosses zero between steps 100–150;
   AdamW between 150–200. Replace the hand-waved "~200 steps" with the per-optimizer crossing.

4. **Norm-weighted opposed-row fraction (interp's #1 "most valuable") — NOT computable from committed data.**
   Only per-row cosines are stored, not per-row norms. The literal fix needs a re-run. Instead I state the
   magnitude-weighting caveat explicitly and honestly bound what the data can/cannot rule out (a writing fix
   that concedes the exact point without fabricating a number).

5. **n=5 (A) vs n=3 (B):** B's seeds {42,123,456} are a subset of A's {42,123,456,789,1337}. State factually.

## FIX-BY-WRITING items applied (all on existing data)
- **F1. 350M comparator defined** — state the pairing is identical CE-vs-MTP; frame 94.5% as intrinsic-alignment evidence, not an undefined foil.
- **F2. Token-frequency → `mean_abs_cos`** — Fig 5 + abstract/text use the non-saturating statistic; keep single-run/both-optimizer tag; demote to "suggestive."
- **F3. Precise crossing steps** in §4.2 (Muon 100–150, AdamW 150–200).
- **F4. Magnitude-weighting caveat** — §Method/dirsupport: aggregate is norm-weighted, per-row median is not; enumerate what the data can and cannot exclude; per-row-norm measurement stays future work.
- **F5. Head-vs-network reconciliation** — §4.3: the high-per-row + non-positive-aggregate signature is generic across 74 matrices; state the head's near-zero aggregate (vs others' strongly negative) is what the "not cancellation" reading leans on, and that strongly-negative aggregates elsewhere are not adjudicated here.
- **F6. Soften stop-gradient** — "inconsistent with passive victim of conflict" → "consistent with a support/capacity account" (intro + §4.4), matching the hedge used elsewhere.
- **F7. Symmetrize tuned-G_nce** — "point estimate still above A" phrasing → "not separable at n=3; neither recovery nor non-recovery"; anchor the capacity claim on the standard variant (4.024, p≈0.010), which IS separable.
- **F8. Hedge "support competes" in Related(iii) + Conclusion** — never state the inferred mechanism unqualified.
- **F9. n=5/n=3 sentence** — B ran on a subset of three of A's five seeds.
- **F10. Tag Fig 2 single-seed-per-optimizer** and the ≈+0.99 init value as a single-run observation.

## Verdict
The core diagnostic result was already well-supported; this pass hardens the *secondary* claims
(persistence, token frequency, emergence timing, the support-divergence inference) so none is worded a
notch stronger than single-run data licenses. With F1–F10, all five reviewers' stated bar is met on
existing data. The one thing genuinely uncloseable without new compute — the norm-weighted opposed
fraction — is converted from a silent gap into an explicit, honest caveat.
