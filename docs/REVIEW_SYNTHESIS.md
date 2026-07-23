# Five-Persona Review — "Aligned but Apart"

Five reviewer profiles read the full compiled manuscript against ground-truth committed data.
Each was told TMLR's bar is *claims-supported-by-evidence + some-audience-interest*, not novelty/impact.

## Ratings (unanimous direction)

| Reviewer profile | Rating | Score |
|---|---|---|
| TMLR Action Editor (senior) | Major revisions | 5/10 |
| Optimization / MTL expert | Major revisions | 5/10 |
| LLM-pretraining practitioner | Major revisions | 4/10 |
| Statistician | Major revisions | 4/10 |
| Interpretability / measurement | Major revisions | 5/10 |

**Consensus:** the science is honest and the scope-writing is unusually disciplined — but the
manuscript in its current form **overstates the temporal reach of its central claim**, and has a
handful of smaller claim/evidence gaps. **Every top concern is fixable with writing/analysis only —
no new experiments.** All five independently converged on the same #1 issue.

## What all five praised
1. The **CE-vs-L1 calibration control** (0.42% vs 98.4%) — the right sanity check; makes "aligned, not conflicting" credible rather than a metric artifact.
2. **Candor**: explicit "what we do / do not establish", surgery disclaimed in an appendix, tuned G_nce reported as non-significant rather than dressed up, support-divergence admitted as inference.
3. **Reproducibility posture**: committed stats script, pinned optimizer, figures regenerated from JSONs.

## The consensus blocker (all 5) — run-duration mismatch  🔴
The abstract says per-row cosine is "≈+1 **throughout training**" and §4.2 says it "holds at ≈+1 for
the **rest of training**." But the 124M per-row instrument (A1) is a **dedicated 1000-step run**,
while the **+0.39-nat loss gap** comes from a separate **30,517-step run** (500M tokens). At the
primary scale, alignment is shown for only the first ~3% of the horizon whose degradation is being
explained. Genuine late-training persistence is measured **only at 350M, n=1** (per-row 0.989 at step
87,000). The text never discloses that these are different runs.
→ **Fix by writing**: name each run's step count, rescope "throughout training" to the measured
window, and lean explicitly on the 350M n=1 data (both A=0.945 and B=0.989) for late-training
persistence. *(Fully closing it at 124M would need a new run — that's future work, not a ship blocker.)*

## Other agreed fixes (no new experiments)
2. **Cohen's d = −23.7 uncaveated** (4 reviewers): inflated by tiny within-seed SD (~0.01–0.02) at small n; add one sentence + a CI on Δ so it isn't read as independent evidence.
3. **Abstract statistic mismatch** (interp; verified true): abstract compares "0.3% cos<0 (MTP)" against "0.4% |cos|>0.3 (control)" — different statistics. Control JSON has no cos<0 fraction. Fix to like-for-like (98.4% vs 0.42% at |cos|>0.3).
4. **Untagged single-run results** (AE): Fig 3 per-layer, A4 untied, Fig 5 token-freq are one-off point estimates but lack the "single seed/run" tag applied to 350M and surgery. Tag them.
5. **Support-divergence hedge arrives late** (3–4 reviewers): declarative in Abstract/Results, admitted-as-inference only in Discussion. Add "(inferred, not directly measured; see §5)" at first use.
6. **"Undertrained, small-batch" unquantified** (3 reviewers): add batch = 16,384 tokens/step and ≈4.0 tokens/param (500M/124M) to Setup.
7. **"No auxiliary recovers within noise" vs p≈0.19 tension** (MTL): the tuned variant is statistically indistinguishable from A; state the power argument explicitly (non-significance at n=3 ≠ recovery) rather than a flat categorical claim.
8. **Report both A and B at 350M/87k** (AE): the differential (0.945 vs 0.989) is the informative quantity for the persistence argument, not B alone.

## Verdict
With fixes 1–8 applied, the reviewers' stated concerns are addressed on **existing data**. The paper
moves from "Major revisions (overstated temporal claim)" to a defensible TMLR submission whose claims
match its evidence. The only thing that fundamentally *cannot* be closed without new runs — full
124M late-training per-row logging, separate-head MTP — is already correctly booked as future work.
