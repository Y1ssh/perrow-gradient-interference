---
type: audit
work_id: perrow-gradient-interference
created: 2026-07-07
scope: full repo + paper + workflow, phase-by-phase ship-readiness
verdict: NOT READY TO SHIP — 1 pivotal experiment missing, ~6 code dents, reproducibility broken
---

# Paper 1 — Ship-Readiness Audit (phase by phase)

All numbers below were **recomputed from the committed JSONs**, and all code claims
were **read out of the actual source files** in `perrow-gradient-interference-main/`.
Every headline number in the manuscript reproduces exactly. The problems are (a) missing
experiments, (b) code-label dents, (c) reproducibility, and (d) claim/framing hygiene —
not fabricated data.

Severity key: 🔴 blocker (paper not sound to ship) · 🟠 should-fix before submission ·
🟡 polish / reviewer-will-notice · 🟢 verified-correct (do not relitigate).

---

## 0. One-line verdict

**Not ready.** The data is real and the numbers check out, but: the single pivotal
experiment (separate-head MTP) does not exist; 350M is n=1; the repo does not reproduce as
shipped (empty requirements, unpinned Muon, no stats code); and there are ~6 code-level
mislabels — including the "prints GPT2-124M while running the 350M model" bug you flagged,
and the dropped `discrepancy`/"PRDR" metric still living in the instrument and gating a
"PROCEED" decision.

---

## 1. 🔴 Missing experiments (these gate soundness)

1. **Separate-head MTP does not exist anywhere in the repo.** Grep for
   `separate.head / per_horizon / independent head / k-1 head` → zero hits. The
   corrected-mechanism-spec v2 §9.1 calls this "the single most important run" and says
   "without this, the paper is about a formulation nobody uses, mislabeled." Real MTP
   (Gloeckle 2024, DeepSeek-V3) uses *separate* heads; this repo's "MTP" is one tied head
   supervised for t+1/t+2/t+3 at once (`phase_b_comparison.py:251-255`). Until this run
   lands, every "MTP damages next-token" claim is really "shared-head multi-offset damages
   next-token." **Blocker.**

2. **350M is n=1 (seed 42 only)** across all three 350M dirs. The "amplification with
   scale" claim (per_row_0.3 94.87%→98.88%) rests on a single seed. Spec §7 wants n=3 for A
   and B at 350M (~$38). Until then, delete every "scale-invariant / amplifies-with-scale"
   sentence or mark n=1. **Blocker for the scale claim.**

3. **No 124M CE-only per-row baseline.** Every Phase-A run is `mtp_shared`; there is no
   `ce_only` per-row measurement at 124M, so "amplification" can't be shown at both scales.

4. **Surgery is single-seed (42) and compared to the wrong baseline.** Recompute vs the
   *matched* B@seed42=4.3903 (not the n=3 mean 4.3767): GS Δ+0.027, Scatter Δ+0.067,
   PCGrad Δ+0.154. GS/Scatter are within two-pass bf16 noise (see §2.6). Either rerun
   matched-seed / single-pass, or drop "surgery proves per-row." **Blocker for the surgery claim.**

5. **No committed statistics code.** No t-test / Cohen's d / p-value / scipy.stats anywhere.
   The "within noise", "p>0.97", "n=3 sufficient" claims are not reproducible.

---

## 2. 🟠 Code-level dents & mislabels (the "denting" you asked about)

1. **`gpt2.py:137` hardcodes `print("  GPT2-124M initialized:")`.** The 350M model is built
   from the *same* `GPT2` class via `GPT2Medium()` (`gpt2_medium.py`), so **every 350M run
   prints "GPT2-124M initialized"** while actually instantiating the 354M model. Same bug at
   `gpt2.py:200` (`"GPT2-124M self-test"`). Fix: derive the label from param count or config,
   e.g. `f"GPT2 initialized ({total/1e6:.0f}M params):"`.

2. **The dropped "PRDR"/`discrepancy` metric is still live in the shipping instrument.**
   `measure_interference.py` computes `discrepancy = per_row_0.3 / max(|global_cos|,0.001)`
   in `measure_interference`, `quick_measure`, and stores it in every result JSON. The spec
   says PRDR is "ill-conditioned and dropped." Worse: `phase_a_measurements.py:598` gates the
   go/no-go **decision** on it (`if discrepancy > 5: PROCEED`), and A5 prints it under a
   `Muon PRDR / AdamW PRDR` column header (`:305-309`). If PRDR is dropped from the paper it
   should be dropped from the code, or explicitly marked "diagnostic only, not reported."

3. **`measure_interference.py` docstring hard-codes the *pre-correction* (wrong) story.**
   The module header still says *"60-98% of neurons receive genuinely conflicting gradient
   signals"* and *"the per-row gradients conflict"* with the "company survey / 60% of teams
   miserable" analogy. This is the exact inversion the 2026-06-05 correction fixed
   (per-row cosine is **+1**, aligned; only ~0.3% conflict). This is the most dangerous dent
   because it's in the instrument a reviewer will read first, and it directly contradicts the
   manuscript. Rewrite the docstring to the aligned/support-mismatch framing.

4. **`phase_b_comparison.py:5` docstring: "Tests whether architectural separation (G_nce)
   eliminates per-row gradient interference"** and the variant comment `b  CE + shared MTP
   (causes 98% interference)`. Same inverted framing — 98% is the *alignment* fraction, not
   "interference." `auxiliary_losses.py` header still advertises the **T4 numbers that did
   not replicate** ("118% damage reduction — MTP becomes NET POSITIVE, 4/5 seeds beat CE-only")
   on a 16M model — spec §2 T1.3 says these "must never reach the paper." Delete from docstring.

5. **PCGrad is scope-confounded vs GS/Scatter.** `pcgrad_muon.py` operates **per-parameter on
   the whole flattened matrix** (`dot = torch.dot(g_ce_flat, g_mtp_flat)`) across **all** params
   including the Muon trunk; GS/Scatter (`gs_muon.py`, `scatter_muon.py`) act **per-row on
   lm_head only**. So PCGrad's larger damage (Δ+0.154) partly reflects it perturbing the whole
   network, not a per-row effect. Must be disclosed in the surgery table caption.

6. **Two-pass bf16 drift in surgery.** `phase_c_negatives.py` runs surgery with **two separate
   `backward()` passes** under bf16 autocast (`:308-325`: `ce_loss.backward(retain_graph=True)`
   then `mtp_loss.backward()`), whereas variant B uses a single fused pass. That alone is
   ~0.02-0.04 nats of numerical drift — larger than GS's "+0.027" effect. Either single-pass
   or document the drift as an error bar.

7. **Minor dents:**
   - `gpt2.py:49` sets `o_proj._SCALE_INIT = 1` but `_SCALE_INIT` is never read anywhere (dead attribute).
   - **47 padding vocab rows** (50304 vs 50257 real GPT-2 tokens) are counted in the per-row
     stats (~0.09% of rows) — harmless but should be masked or noted.
   - `auxiliary_losses.py` is titled "contrastive NCE" but the docstring for NextLat says
     "Uses cosine distance, not contrastive NCE" — and G_nce's own `nce_loss` is a
     logsigmoid over cosine similarities, i.e. closer to cosine-contrastive than true NCE.
     The spec calls it "contrastive embedding-matching, not prediction"; keep terminology
     consistent between code and paper.
   - `A2` (AdamW) uses lr 1e-3/wd 0.01 while `A1` (Muon) Adam group uses 3e-4/wd 0.1 — the
     Phase-A Muon-vs-AdamW comparison is LR- and wd-mismatched (see §3).

---

## 3. 🟠 Confounds to disclose (verified in code)

- **Weight-decay confound (real).** Muon head group `weight_decay=0.1` (`gpt2.py:191`) vs
  AdamW comparison `weight_decay=0.0` (`phase_c_negatives.py:232`). Same head LR, different
  decay → the "optimizer-robust (Δ≈+0.4 under both)" claim is confounded. Spec v2 §3 reinstates
  this (v1 wrongly retracted it). Disclose or rerun matched.
- **Per-layer non-specificity.** Per-row alignment is high across many matrices, not unique to
  lm_head (attention v_proj/o_proj show 91-93% with *negative* global cosine). The paper must
  argue the head matters because it *directly produces the logits*, not because the metric
  singles it out. Don't let the per-layer figure read as "the head isn't special."
- **Tiny batch, no grad-accum:** 16×1024 = 16K tokens/step at *all* scales incl. 350M — noisy.
- **Undertrained:** R≈3.86-4.02 tok/param (~5× below Chinchilla). "MTP damages" may be
  regime-specific; MTP benefits often need more training/scale.
- **Damage is aux-weight-dependent** (headline 0.5/0.25; Phase-D shows lower weight → less damage).
- **Pure-Muon-B** moves wte/lm_head into the Muon group → its worst loss (4.5563) mainly shows
  Muon is bad for the output matrix, a *separate* point from interference. Careful using it as
  "the matched surgery baseline."

---

## 4. 🟠 Claim / framing hygiene

- **The "tuned (10,9) G_nce ≈ A within noise" claim is a best-seed pick.** The three full
  30517-step (10,9) runs are 4.0197 / 3.9991 / 3.9891 → **mean 4.0026, Δ+0.018 vs A=3.9851**
  (>1σ above A, sd_A=0.0138). Only the single best seed (3.9891) is near A. Standard G_nce is
  4.0241 (Δ+0.039, no recovery). So *neither* standard nor tuned G_nce recovers A when
  aggregated — the draft's abstract-intro §"Numbers in hand" note #7 already flags this, but
  make sure no figure/table shows the 3.9848/3.9891 single seed as "recovery."
- **Title still says "Conflict" in `paper/draft.md` H1** ("Per-Row Gradient Conflict in
  Multi-Token Prediction"), contradicting the corrected finding (aligned, not conflict). The
  newer sections already moved to "Aligned but Apart." Reconcile the stub title before anyone
  reads the repo→paper together.
- **Bibliography unverified.** `02-related-work.md` marks 9+ citations ⚠ (author/venue/year/arXiv
  ID unconfirmed), incl. Godey & Artzi 2026 arXiv ID, Recon authors, Aynetdinov/Godey-2024 IDs.
  Do the bibliographic pass before submission.
- **`draft.md` body is still a stub** — Method/Experiments/Results/Discussion unwritten. Only
  Abstract+Intro+Contributions (`01-`) and Related Work (`02-`) are drafted. Phase 5 migration
  is the bulk of remaining writing (spec estimates 20-40h).

---

## 5. 🟠 Reproducibility (verified — repo does not reproduce as shipped)

- **`requirements.txt` is 0 bytes.** No pinned deps at all.
- **Muon is unpinned and not vendored.** `from muon import SingleDeviceMuonWithAuxAdam` in
  4 files; `setup.sh:263` installs `git+https://github.com/KellerJordan/Muon` at HEAD (no
  commit pin). No `muon.py` in the repo. A future `pip install` may get a different optimizer.
- **`README.md` is 2 lines.** No install / data / run / results-layout / license-of-data notes.
- **`GNCELossAblation` duplicates `GNCELoss`** (`auxiliary_losses_ablation.py` vs
  `auxiliary_losses.py`) — the `nce_loss` bodies must stay byte-identical for `neg_type='roll'`
  or the Phase-B vs Phase-D comparison isn't valid. Currently they match; add a test that asserts it.
- **`setup.sh:292` hardcodes `git config user.email "yash@research.local"`** and makes dirs
  `phase_e/phase_f` that don't exist in results — harmless but sloppy; the fake email will end
  up in commit metadata.
- **`.instance_log`** committed (contains a setup timestamp only — no secrets, but shouldn't be
  in the repo).
- **`*:Zone.Identifier` sidecar on every file** — Windows "downloaded from the internet"
  metadata. Strip before pushing (`find . -name '*:Zone.Identifier' -delete`).

---

## 6. 🟢 Verified correct — the solid foundation (do NOT relitigate)

- Causal masking (`is_causal=True`), weight tying, and no train/val leakage (124M train[:480M]
  val[480M:]; 350M train=[0:480M]+[500M:1389M] val[480M:500M]).
- **Eval is pure next-token CE for every variant** (`eval_model`) → the +0.39 nat gap is
  apples-to-apples.
- Per-row measurement is on **held-out val** batches; seeding is applied (`torch.manual_seed`).
- **Control calibrates:** CE-vs-L1 → per_row_0.3 = 0.42% vs 98.38% for CE-vs-MTP.
- Cosine / PCGrad / GS / Scatter projection math is correct (self-tests present and pass logic).
- Alignment is present from step 1 → supports "structural, not learned."
- **Every committed headline number reproduces exactly** (see §7).

---

## 7. Numbers — recomputed from committed JSONs (all match)

| Quantity | Manuscript | Recomputed | ✓ |
|---|---|---|---|
| 124M A (n=5) | 3.9851±0.014 | 3.9851, sd 0.0138 | ✓ |
| 124M B (n=3) Δ | 4.3767, +0.392 | 4.3767, +0.3915 | ✓ |
| B_sg (n=3) Δ | 4.4927, +0.508 | 4.4927, +0.5076 | ✓ |
| G_nce std (n=5) Δ | 4.0241, +0.039 | 4.0241, +0.0390 | ✓ |
| NextLat (n=3) Δ | 4.0365, +0.051 | 4.0365, +0.0513 | ✓ |
| G_nce tuned (10,9) n=3 | "≈A (~3.99)" | mean 4.0026 (Δ+0.018); min 3.9891 | ⚠ best-seed |
| Surgery vs B@42=4.3903 | PCGrad/GS/Scatter | 4.5444/4.4175/4.4575 | ✓ |
| Pure-Muon-B | 4.5563 | 4.5563 | ✓ |
| 350M r4 (n=1) A/B Δ | 3.7516/4.1291,+0.378 | 3.7516/4.1291,+0.3775 | ✓ |
| Per-row a1 (50,304 rows) | med +1.0; cos<0 0.33%; \|cos\|>0.3 98.38% | identical | ✓ |
| Control (a3) | ~0.42% | 0.415% | ✓ |
| 350M-B per_row_0.3 | 98.88% | 0.9888 | ✓ |
| phase_b_50M A/B (robustness) | A≈3.94/B=4.31 | A=3.9264(n=5)/B=4.3050(n=1) | ✓ |

Committed result files: **66 JSONs** (phase_a=5, phase_b=19, phase_b_50M_repeated=9,
phase_c=10, three 350M dirs=1 each, phase_d=20).

---

## 8. Prioritized ship checklist

**Must-do before the paper is sound (🔴):**
1. Run separate-head MTP (the pivotal experiment) — decides the scope of every claim.
2. 350M n=3 for A and B (+ per-row-norm logging) — kills the n=1 objection, makes
   "support mismatch" a *measured* figure not an inference.
3. Matched-seed / single-pass surgery rerun — or drop "surgery proves per-row."
4. Commit statistics code (Welch's t vs A, Cohen's d, n-sufficiency).

**Should-fix before submission (🟠):**
5. Fix the `GPT2-124M` hardcoded print (label from config/param-count).
6. Rewrite `measure_interference.py` + `phase_b_comparison.py` + `auxiliary_losses.py`
   docstrings to the corrected aligned/support-mismatch framing; delete the non-replicating
   T4 numbers.
7. Decide PRDR: remove `discrepancy` from code + the PROCEED gate, or mark diagnostic-only.
8. Fill `requirements.txt`, pin Muon to a commit, expand README.
9. Reconcile `draft.md` title ("Conflict" → "Aligned but Apart"); finish the bibliographic pass.
10. Disclose confounds (weight-decay; PCGrad all-params vs GS/Scatter head-only; two-pass drift;
    per-layer non-specificity) in the relevant captions.

**Polish (🟡):**
11. Strip `*:Zone.Identifier`, `.instance_log`, fake git email, dead `_SCALE_INIT`,
    the phase_e/phase_f mkdir; mask the 47 padding rows; add a byte-identity test for the two
    GNCELoss copies.
12. Write the `draft.md` body (Method/Experiments/Results/Discussion) — the 20-40h Phase-5 bulk.
