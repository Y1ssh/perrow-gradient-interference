# Ship Checklist — "Aligned but Apart" (mechanism: norm-weighted cancellation)

**Status: ONE COMPUTE ITEM OPEN** — second pre-submission review addressed in full except the
mixture-optimum test (R2), which needs 4 GPU runs (~3.5 h). Mechanism itself is confirmed and survived
the review's hardest challenge (active-row denominator). Paper compiles clean at 15 pp.
Last updated 2026-07-23 (second review response).

## The result (locked, n=3 per optimizer)
| Optimizer | norm_profile_cos | opposed_norm_fraction | aggregate | per-row |cos|>0.3 |
|---|---|---|---|---|
| Muon  | 0.979 ± 0.003 | 0.604 ± 0.026 | -0.163 ± 0.030 | 98.3% |
| AdamW | 0.977 ± 0.007 | 0.692 ± 0.011 | -0.317 ± 0.020 | 98.6% |

**Verdict:** support divergence REFUTED (norm-profile cos ~0.98, not ~0); norm-weighted
cancellation CONFIRMED (opposed minority carries 60-69% of mass). Both optimizers, tight bars.
AdamW cancels harder than Muon: optimizer-dependent magnitude, identical mechanism.

## DONE
- [x] 6 confirmation runs (Muon+AdamW x seeds 42/123/456), all 50,257 rows, no NaN, harvested + saved as artifacts.
- [x] Aggregate + verdict locked; fig7_norm_support drawn (n=3 evidence, both panels).
- [x] Fig 1 schematic redrawn from measured proportions (was disjoint-support, now opposed-minority).
- [x] Mechanism rewritten in all 16 spots across abstract/intro/method/results/discussion/conclusion.
- [x] Method toy replaced (3-aligned + 1-opposed-high-norm; every per-row cos defined) [review B3].
- [x] Paired test on shared seeds 42/123/456 added as primary (t=50.0, p=4.0e-4); Welch secondary [B7].
- [x] Cohen's d=-23.7 dropped [B10]; stop-grad demoted to confounded/suggestive [B10]; 350M tagged n=1.
- [x] GradNorm (Chen 2018) + ForkMerge (Jiang 2023) cited, verified from arXiv abstracts [B8].
- [x] Title kept ("Aligned but Apart... Misread..."); "Apart" = the aggregate's misread, never glossed as support.
- [x] Downstream docs reconciled: README, vault draft.md + 01-abstract-intro.md, claim_ledger A5, CODE_FIXES,
      EXTERNAL_REVIEW_TRIAGE (B1/B2/B3 marked done).
- [x] Fresh repo compile: 15 pages, 0 undefined refs, all 8 figures present + referenced.
- [x] Paper (.tex): 0 em-dashes.
- [x] Numbers consistent across paper + README + claim_ledger + RESULT_B2.

## SECOND REVIEW RESPONSE (2026-07-23) — see docs/EXTERNAL_REVIEW_TRIAGE.md for detail
- [x] R1 active-row denominator: added honest 12%-active / median-0.54 framing to §4.1 + fig1 caption;
      verified mechanism unchanged (parallel rows carry 0.6% of mass). Survived the review's hardest hit.
- [x] R4 surgery logic corrected in 5 spots ("mass-dominant yet causally inert", not "nothing to remove").
- [x] R5 "wrong quantity" softened to "uninformative about where interference lives" (abstract + fig0).
- [x] R6 exact factorization aggregate = rho_norm x <cos>_mass added + verified (0.98 x -0.082).
- [x] R9 stats: p<1e-4 truncation, Welch sign, MDE ~0.05, Holm note, Muon-vs-AdamW test (p=0.015).
- [x] R10 citations added + arXiv-verified: Kurin 2201.04122, Xin 2209.11379, Gao 1907.12009.
- [x] R11 G_nce / NextLat defined inline; R13 title line-break fixed, fig0 flagged schematic.
- [ ] **R2 mixture-optimum test (THE open blocker):** `bash experiments/run_mtp_sweep.sh` (4 runs, ~3.5 h)
      then `python analysis/analyze_mtp_sweep.py`. Reframe or confirm per the verdict.
- [ ] R7 (optional): decode the ~166 opposed rows into a table (frequency link only moderate; don't overclaim).

## USER'S REMAINING ACTIONS (before push)
- [ ] Optional: run `bash clean_repo.sh --apply` to sweep 371 Zone.Identifier files (gitignored; will NOT
      ship even if left, but cosmetic). (Agent's Trash-tool cleanup was declined; this is the local sweep.)
- [ ] `git add -A && git commit && git push` from your clone (set git identity first).
- [ ] Sanity: open paper/main.pdf, eyeball Fig 1 (schematic) + Fig 7 (evidence).

## CAMERA-READY (after acceptance)
- [ ] `\usepackage[accepted]{tmlr}` + real author block (currently anonymous placeholder).
- [ ] Swap arXiv preprint bib entries for published-venue versions where available.

## OPTIONAL POLISH (not blockers)
- [ ] B4: gradient-structure explanation for the alignment (both row-grads in span{h_t}).
- [ ] B5: report CE-vs-L1 control's AGGREGATE (currently only per-row 0.42%).
- [ ] README/CODE_FIXES markdown em-dashes (structural, pre-existing; paper itself is clean).
