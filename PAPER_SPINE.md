# PAPER_SPINE.md

Reference architecture for *Aligned but Apart: Why Cosine Conflict Diagnostics
Misread Auxiliary-Loss Interference at the Output Projection*.

**Purpose.** Every edit to the manuscript is checked against this document, not
against the previous wording. Prior review rounds repeatedly found their only
substantive defects in freshly written prose while the committed data had not
changed; the cause was editing against strings with no statement of what each
section is *for*. This is that statement.

**How to use it.** Before writing a sentence, find its layer. A sentence may
assert only what its layer's *Permitted* list allows, must not assert anything on
its *Forbidden* list, and may print a number only if that number is *Owned* by
that layer (elsewhere: cross-reference, do not restate). Any new explanatory
clause must be recomputed from a committed JSON **before** it is written.

**Numbering.** This document references LaTeX **labels**, never section or figure
numbers. Numbers move whenever a subsection is split or a float relocates; labels
do not. `tests/check_paper_claims.py` asserts that every label named here exists
in the manuscript, so a rename breaks the gate rather than silently rotting this
map.

---

## Story in one paragraph, no numbers

An auxiliary multi-token objective is added to a language model and shares the
model's tied output projection. Main-task loss gets worse. The standard
diagnostic -- a cosine between the two losses' gradients -- reads near zero, which
practitioners read as gradient conflict, so they reach for gradient surgery. It
does not help. Measuring the same gradients *per vocabulary row* inverts the
picture: on rows that carry supervision the two losses point nearly the same way.
The near-zero aggregate is not disagreement but a weighted average in which a
tiny, very high-norm minority of rows -- the most frequent tokens and punctuation
-- outvotes an agreeing majority. And the reason the loss is worse is not
geometry at all: with one shared head, the sum of the two losses is algebraically
one cross-entropy against a blended next-and-future target, so the optimum has
moved. Surgery adjusts the direction of travel; it cannot move the optimum back.
**Moral:** a diagnostic that aggregates over structured parameters can report
conflict where there is agreement and send you to fix a problem you do not have.

---

## Layer dependency graph

    L0 setting
     └─ L1 instrument (per-row cosine)
         └─ L2 decoupling finding
             └─ L3 identity: aggregate = rho_norm x <cos>_mass  (poses a two-way fork)
                 └─ L4 norm measurement (picks the branch: cancellation, not disjoint support)
                     ├─ L5 calibration & generality
                     └─ L6 falsification of the conflict account (negative result)
                         └─ L7 replacement account (soft-label identity, moved optimum)
                             └─ L8 zero-parameter test of the replacement
                                 └─ L9 moral & scope

Each layer is *strictly* dependent on the one above. L3 poses the fork that L4
resolves; L6 creates the explanatory vacancy that L7 fills. Reordering breaks the
argument, not just the prose.

---

## L0 -- Setting

* **Section:** `sec:intro`, `sec:related`
* **Job:** Establish that the situation exists and that practitioners diagnose it
  with an aggregate cosine.
* **Permitted:** auxiliary losses sharing a tied output projection are common;
  aggregate cosine is the standard conflict diagnostic; surgery is the standard
  response.
* **Forbidden:** any claim about what *our* measurement shows. No numbers from
  L2+ may appear here except as a forward-referenced summary in the abstract and
  contributions list.
* **Owns:** nothing numeric.
* **Note:** the Related Work statement about separate-head MTP architectures must
  say they share *one unembedding* (Gloeckle et al., DeepSeek-V3 both do). The
  earlier claim that they use separate unembeddings was factually wrong and is
  Forbidden.

## L1 -- Instrument

* **Section:** `sec:instrument` (the per-row instrument), `sec:dirsupport` (direction versus support)
* **Job:** Define the measurement. Stop flattening the matrix; take one cosine per
  vocabulary row. Define the active-row classifier.
* **Permitted:** the definition; that the classifier is `|mtp_norm/ce_norm - 0.75| > 0.01`,
  i.e. a **proxy** for target presence; that gradients are measured by
  `torch.autograd.grad` on the raw loss, before any optimizer transform.
* **Forbidden:** "opposed rows are active by definition"; "a criterion equivalent
  to target presence". The classifier tests magnitude only. A row that receives a
  target moves the ratio away from 0.75; the converse does not hold exactly,
  because a row whose ratio coincidentally sits inside the tolerance is counted
  parallel regardless. The estimand (true target presence) and the estimator (the
  ratio classifier) are distinct, and `active = full/(1-f)` is exact only under
  the estimand.
* **Owns:** tolerance 0.01; ratio 0.75; the sd convention (sample sd, ddof=1,
  never standard errors -- with the mixed-n disclosure).

## L2 -- Finding: decoupling

* **Section:** `sec:aligned` (aligned on rows that carry supervision), `sec:emergent` (decouples during training)
* **Job:** Show per-row alignment coexisting with a near-zero aggregate, and that
  the two *start together* and separate during training.
* **Permitted:** per-row aligned on active rows; aggregate near zero or negative;
  the decoupling is emergent, not initial; the full-vocabulary median is +1
  because most rows are scalar multiples by construction.
* **Forbidden:** attributing the aggregate to disjoint support (that is L4's
  branch, and it is refuted); calling the full-vocabulary statistic informative
  about pair-specific structure.
* **Owns:** active-row median cosine **0.529 +/- 0.008** (Muon, n=3) and
  **0.547 +/- 0.016** (AdamW); percent opposed on active rows **3.7 +/- 0.2%**
  (Muon) / **3.2 +/- 0.2%** (AdamW), quoted as "3-4%"; active fraction
  **7.7-8.4%** across all six runs; full-vocabulary median **+1.00** with
  **0.33%** opposed and **98.4%** above |cos|>0.3 (phase-A diagnostic batch);
  the **140 rows / 0.28%** figure belongs to L4's batch, not here; aggregate
  **-0.163 +/- 0.030** (Muon) / **-0.317 +/- 0.020** (AdamW); crossing below zero
  between steps **100-150** (Muon) and **150-200** (AdamW); f pair
  **3.1%** (five conforming runs) / **4.5%** (including the Muon seed-123 outlier
  at 11.52%).
* **Source:** `results/norm_support/*.json` (6 runs); `results/phase_a/a{1,2}_*.json`.

## L3 -- Identity: why an aggregate can lie

* **Section:** `sec:dirsupport` + Eq. (1); `fig:normdecomp`
* **Job:** Prove algebraically that the aggregate is a *norm-weighted* object, and
  thereby narrow the cause to exactly two candidates: **disjoint support** or
  **norm-weighted cancellation**.
* **Permitted:** the exact factorization `agg = rho_norm x <cos>_mass`; that a
  flat-norm world would report the mean per-row cosine; that a negative aggregate
  mathematically *requires* opposed rows to carry weight.
* **Forbidden:** "the collapse is not opposition" -- false, since a negative
  aggregate requires opposed mass. The defensible form is "per-row norm
  structure, not the typical per-row cosine, sets the aggregate."
* **Owns:** the identity itself; mean-per-row-cosine-if-flat **+0.94** / **+0.96**
  against actual **-0.08** / **-0.28** (single diagnostic run per optimizer);
  the worked bridge **-0.193 = 0.978 x (-0.197)** (Muon, seed 42) and
  **-0.319 = 0.985 x (-0.324)** (AdamW, seed 42), valid within one measurement.

## L4 -- Measurement: which branch

* **Section:** `sec:mechanism` (norm support); `fig:normsupport`, `fig:tokenlorenz`, `fig:normdecomp`, `tab:perrow`
* **Job:** Instrument the per-row *norms* and settle L3's fork.
* **Permitted:** the two losses load the **same** rows, so disjoint support is
  refuted; a high-norm opposed minority carries the majority of the mass, so
  cancellation is confirmed; the mechanism is optimizer-dependent in magnitude
  but identical in kind; decoding the top rows gives frequent function words and
  punctuation.
* **Forbidden:** "support diverges" / "disjoint support" as the mechanism (refuted
  by the 0.98 norm-profile cosine); claiming the opposed minority *causes* the
  loss increase (L6/L7 handle causation, and causation is not established).
* **Owns:** norm-profile cosine **0.979 +/- 0.003** (Muon) / **0.977 +/- 0.007**
  (AdamW) full-vocabulary, **0.978 +/- 0.004** / **0.976 +/- 0.007** active;
  opposed-norm fraction **0.604 +/- 0.026** / **0.692 +/- 0.011** full,
  **0.608 +/- 0.020** / **0.708 +/- 0.010** active (note: 0.709 is what averaging the
  four-decimal appendix values gives; from the arrays it is 0.708486); Welch between optimizers
  **t = 5.42**, df 2.77, p 0.015 (full) and **t = 7.82** (active);
  **140 opposed rows = 0.28%** of vocabulary carrying **60.7%** of mass;
  **47 rows** = two-thirds of all mass; top four **41%**; per-seed appendix
  values (all 24).
* **Source:** `results/norm_support/*.json`; `figures/token_table_opposed.csv`;
  `figures/lorenz_curve.csv`.

## L5 -- Calibration & generality

* **Section:** `sec:calib` (calibration), `sec:general` (general, survives untying), `sec:rare` (rare-token artifact, 350M)
* **Job:** Rule out that the finding is an artifact of the instrument, of the
  head, of weight tying, or of scale.
* **Permitted:** the CE-vs-L1 control on the matched active-row denominator; the
  permutation nulls; the CE-vs-CE disjoint-half comparison as a **floor**, not a
  ceiling; generality across all network matrices; survival of untying; the
  apparent rare-token effect is a **construction artifact** of the denominator.
* **Forbidden:** calling the disjoint-half number a "ceiling"; comparing a
  full-vocabulary figure against an active-row control and calling it a common
  denominator; claiming the token-id block control holds the frequency profile
  fixed -- the blocks are contiguous id ranges spanning wide frequency ranges, so
  it does not.
* **Owns:** control **1.0%** above |cos|>0.3 on active rows against **83%** for
  CE-vs-MTP on the same denominator; permutation null **0.085-0.111** decile,
  **0.031** global; **74** matrices with median **-0.36** (Muon) / **-0.34**
  (AdamW) and **89%** / **86%** negative; untied **99.3%** against **98.7%** tied
  at the matched step 500; 350M gap **+0.377** (CE-only 3.752, shared-MTP 4.129,
  87,000 steps, 355M params) against 124M **+0.3915**; active fraction collapsing
  **25% -> 1.5%** common to rare.
* **Source:** `results/phase_a/*.json`, `analysis/ceilings_results.json`,
  `results/**/scale_350m_r4_*.json`, `figures/fig5_token_frequency.csv`.

## L6 -- Falsification of the conflict account

* **Section:** `sec:interventions`; `fig:interventions`; `tab:conditions`; `app:surgery`
* **Job:** Test the conflict account on its own terms -- remove the conflict and
  see whether the damage goes. It does not. This is the negative result that
  creates the vacancy L7 fills.
* **Permitted:** shared MTP degrades next-token loss; stop-gradient is worse;
  conflict-targeted surgery does not recover the baseline; the standard
  capacity-separated auxiliary demonstrably does not recover it, and the tuned
  variant is **inconclusive at our detection floor** (not "does not recover").
* **Forbidden:** blanket "no auxiliary recovers the baseline" (the tuned variant
  is inconclusive, not refuted); attributing the tuned-vs-standard difference to
  either the layer choice or the multiplier -- **neither term is resolvable at
  these seed counts**, and both confidence intervals are wider than the
  difference they would decompose; "reprojecting gradients *cannot* move an
  optimum" (the exact statement holds for symmetric projection; the head-only
  variants are empirically inert).
* **Owns:** Table 1 in full -- CE-only **3.9851 +/- 0.0138** (n=5), G_nce
  **4.0241 +/- 0.0207** (n=5), G_nce tuned **4.0026 +/- 0.0156** (n=3), NextLat
  **4.0365 +/- 0.0179** (n=3), shared MTP **4.3767 +/- 0.0209** (n=3),
  stop-gradient **4.4927 +/- 0.0101** (n=3); deltas **+0.039** (p 0.010),
  **+0.018** (p 0.19), **+0.051** (p 0.018), **+0.3915** (p 8e-5), **+0.5076**
  (p 6e-9); Welch **t = -28.8**, df 3.08, CI **[0.35, 0.43]**; paired
  **t = 49.9**, p 4.0e-4; Holm ordering; power: paired 80% MDE **0.045**, and at
  SE 0.008/df 2, 0.05 -> **86%** and 0.04 -> **72%**; Welch at SE 0.014/df 3.1,
  0.05 -> **68%**, 80% needs **0.06**.
* **Source:** `results/phase_b/*.json`, `results/**/ablation_alpha0.1_*.json`.

## L7 -- Replacement account

* **Section:** `sec:softlabel` (what the shared-head objective optimizes) + Eq. (2)
* **Job:** Supply the mechanism the negative result leaves vacant. Exact rewrite:
  the shared-head sum *is* one cross-entropy against a blended target, so the
  optimum moved.
* **Permitted:** the identity (verified numerically to 1e-9); the KL bound; that
  the optimum moves and the main-task cost is bounded; that this explains why
  surgery is inert.
* **Forbidden:** claiming the mixture account is *confirmed* (it is the leading
  candidate); claiming the mixture cost is the whole of the degradation.
* **Owns:** ceiling **log 1.75 = 0.560** nats; observed gap is **70%** of the
  ceiling; blended target = **57%** next / **43%** future.

## L8 -- Zero-parameter test

* **Section:** `sec:methodkl` (the KL estimator), `sec:mixturetest` (sweep + prediction); `fig:mixturesweep`
* **Job:** Test L7 with no fitted parameters, against an anchored sweep.
* **Permitted:** the anchor gate passing; the band bracketing the low weights and
  undershooting at full weight; the residual **named, not assigned**; the two
  independent side-checks (predictive entropy, future-offset performance);
  truncation biases the estimate **upward**, so fuller marginalization lowers the
  prediction and widens the residual.
* **Forbidden:** quoting only the most favourable estimator variant or half (this
  error class recurred four times: drop-variant shape, k=64 interval top,
  drop-variant pointwise, half-0-only slopes) -- always quote the range across
  variants and halves; calling the sweep-shape test strong (it is weak; T1's
  s=1 prediction is the hypothesis test).
* **Owns:** band **[0.2316, 0.3363]** at k=2048 against observed **0.3715** =
  **62-91%** of the matched gap; residual raw **+0.035 = 1.4 sigma**, extrapolated
  **+0.041 = 1.6 sigma** (half 0) and **+0.046 = 1.8 sigma** (half 1), quoted as
  **1.4-1.8 sigma**; sigma **0.0255 = sqrt(2) x 0.018**, a round stand-in fixed
  in `analysis/estimate_kl.py:205-207`, with the empirical quadrature alternative
  **0.0251**; within-sweep gaps **0.091 / 0.183 / 0.372**; predictive entropy
  **1.26** observed against **0.95** predicted (75.6%); future offsets better by
  **2.91** (t+2) and **2.57** (t+3); coverage **0.90-0.92**; 2,048 positions per
  row.
* **Source:** `analysis/kl_scan_results.json`, `results/phase_e/sweep_scale*.json`.

## L9 -- Moral & scope

* **Section:** `sec:discussion`, `sec:limitations`, `sec:conclusion`
* **Job:** State what the measurements establish, the two distinguishable failure
  modes, and -- explicitly -- what is *not* established.
* **Permitted:** the diagnostic misreads this regime; two failure modes (Muon's
  near-zero aggregate is a false negative; AdamW's clearly negative aggregate is
  a misleading positive, since surgery is inert either way); the regime is
  undertrained, small-batch, <=350M.
* **Forbidden:** any claim of causation between norm-weighted opposition and the
  loss increase; any claim about MTP *as practiced* with separate heads.
* **Owns:** nothing numeric of its own; every number here is cross-referenced.

---

## Known structural gaps (state, do not paper over)

1. **Co-location.** L2/L4 measure at step 1000; L6 measures loss at step 30,517.
   The aligned-gradient state and the degraded-loss state are never exhibited in
   one measurement. Currently bridged by assumption. **Largest gap.**
2. **No CE-only per-row baseline at 124M.** The cleanest control -- same
   instrument on a model that never saw the auxiliary -- does not exist at the
   main scale.
3. **Full-vocabulary control discrepancy -- PARTIALLY EXPLAINED (this pass).**
   The apparent 90x gap is between `results/phase_a/a3_control.json` (0.42%) and
   `analysis/ceilings_results.json` (36.2%). What the committed data *do* show:
   A3's threshold curve is 0.1 -> 74.9%, 0.2 -> 35.2%, 0.3 -> 0.42%, 0.5 -> 0%, so
   **34.8 percentage points of |cos| mass fall between the 0.2 and 0.3 bins** and
   the above-0.3 statistic is demonstrably threshold-sensitive on this run. What
   the data do **not** show: a3 commits no `row_cosines` array, only these four
   bins, so the shape of the distribution inside [0.2, 0.3) is unobservable and we
   **cannot establish** that the 0.3 cut sits on a near-vertical segment rather
   than that the two runs' distributions genuinely differ near this quantile. The
   two snapshots are also at different training states (a3 val_loss 7.656 against
   5.572 at step 1000), so comparing A3's above-0.2 fraction with the sweep's
   above-0.3 fraction is not evidence of a small shift. What survives
   unambiguously, and is what the control exists to establish: the aggregate is
   near zero on **both** runs (+0.0022 and -0.0105). Record as a threshold-choice
   caveat with the curve given, not as a resolved discrepancy; resolving it needs
   the per-row array from a re-run.

4. **Sweep is single-seed** (seed 42). L8, the strongest quantitative claim,
   rests on one seed.
5. **Untying is thin** -- one run, 500 steps, against 30,517 elsewhere.
6. **Loss-only outcome** -- no downstream task, so "degrades" means validation
   loss rises.

## Standing rules

1. No sentence claims more than its layer's job permits.
2. Every number has exactly one owning layer; elsewhere, cross-reference.
3. Any new explanatory clause is recomputed from JSON *before* being written.
4. Symmetric evidential standards: if one term of a comparison gets a confidence
   interval, so does the term it is compared against.
5. The gate asserts both presence of current claims and **absence** of retracted
   framings, whitespace-normalized, over `.tex` *and* `analysis/`, `measurement/`,
   `figures/` source.
6. Spreads are sample sd (ddof=1), never standard errors; the mixed-n convention
   is disclosed where it applies.
7. Em-dashes: zero in prose, in both unicode and LaTeX `---` form.
