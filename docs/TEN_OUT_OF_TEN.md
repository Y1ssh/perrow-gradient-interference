# How to make "Aligned but Apart" 10/10 — the honest list

You asked for 10/10 "without making false claims and using everything we have at max."
Here is the straight answer, split into what I just did, what you can still do on existing data,
and the one honest ceiling.

## A. Done this pass (writing/style — no claims touched)
- **Em-dashes: 58 → 0.** Every prose em-dash was a real "AI-generated" tell at that density
  (one per ~74 words). Converted to commas, colons, parentheses, or sentence breaks. Recompiled clean.
  I checked I didn't just swap in a colon-cluster (colons are ~1 per 140 words, not a tell).
- **Hedge-phrase drumbeat reduced:** "consistent with" 10 → 7, varied the routine ones, kept the
  ones that carry the deliberate *associational-not-causal* meaning.
- **Formulaic transitions:** none of consequence (no Moreover/Furthermore/Notably pile-up; the one
  "importantly" is "more importantly" and earns its place; "as such" is idiomatic, not a transition).
- Numeric + consistency fixes from the audit (3.743 not 3.744; checklist Section D/G reconciled).

## A2. Done this pass (NEW substantive result — the "other way", zero runs)
- **Norm-decomposition identity + Figure 6.** The aggregate cosine equals the plain *mean* of
  per-row cosines exactly when gradient magnitude is flat across rows. That mean is **+0.94 (Muon) /
  +0.96 (AdamW)**; the actual aggregate is **-0.08 / -0.28**. So the ~1.0 collapse that the standard
  diagnostic reports is **demonstrably** a per-row *norm* effect, not a directional one, computed
  entirely from committed data. This is the biggest substantive lift available without compute. It
  *partially* addresses the interpretability reviewer's ask: it proves the aggregate collapse is a
  norm-weighting effect, but it does NOT yet separate support divergence from a high-norm opposed
  minority. That separation needs the norm-profile cosine cos(a,b) and the opposed-norm fraction,
  which require the per-row norms `measure_norm_support.py` logs, i.e. the one short run (item 4).
  Note also: Fig 6 uses the *flat-norm mean* counterfactual (equal weights), which is a related but
  less specific statistic than cos(a,b); the run computes the sharper number.
- **Honesty guard:** a *negative* aggregate mathematically requires the opposed minority to carry
  some weight, so the figure claims "norm structure, not the typical cosine" — NOT "not opposition."
- **Shipped the instrument** (`measure_norm_support.py`) that logs per-row norms and reports the two
  statistics (norm-profile overlap; opposed-norm fraction) that separate support divergence from a
  high-norm opposed minority. It runs on the short diagnostic pass, so item 4 below is now *one run
  with the code already written*.

## B. What still separates this from a 10 — and which are fixable on data in hand
**Fixable by writing (I can do these now, zero new runs):**
1. **A one-figure "cartoon" of support divergence.** The paper's central idea (aligned directions,
   disjoint magnitude) is currently only prose + the per-layer scatter. A small schematic (two vocab-row
   vectors pointing the same way but with magnitude on different rows) would make the mechanism legible
   in one glance. This is the single highest-value addition and needs no data.
2. **A worked toy example / identity box** showing algebraically how per-row cos≈+1 with aggregate≈0
   arises from disjoint support (a 2-row illustration). Turns the inference into something the reader
   can verify by hand.
3. **Tighten the contributions list** so each bullet maps 1:1 to a figure/number (reviewers like a
   contribution they can point at).

**Genuinely NOT fixable without new compute (the honest ceiling):**
4. **The norm-weighted opposed-row fraction.** The one measurement that would turn "support divergence"
   from a strong inference into a *direct* result. Committed JSONs store per-row cosines but not per-row
   norms. Every reviewer noted this; it is now an explicit caveat, not a hidden gap. One short re-run
   (log per-row norms at the existing measurement steps) closes it.
5. **124M late-training persistence** (per-row logging past step 1000 at 124M) and **separate-head MTP**
   (the regime used at scale). Both are correctly booked as future work.

## C. The honest ceiling
A **diagnostic/measurement paper on committed data** realistically tops out at a confident
**Accept (8/10)** — strong, clean, well-scoped, but its central mechanism is *inferred*, not directly
measured, and its scale is small. That is not a flaw in the writing; it is the nature of the evidence.
Two reviewers explicitly said the norm-weighted measurement (item 4) is what would move it to a clear 9–10.
Claiming a literal 10 without that measurement would require overstating the inference as a result — which
is exactly the kind of false claim you told me not to make.

**My recommendation:** let me add items 1–3 now (they make the paper as good as the data allows, and
genuinely lift the reading experience), ship it, and note item 4 as the natural follow-up. If you want the
true 9–10 version, item 4 is one cheap re-run (log per-row gradient norms at the measurement steps you
already use) and I can write the exact measurement patch for it.
