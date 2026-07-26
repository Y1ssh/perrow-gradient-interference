# Code Fixes — diagnostic-pivot cleanup

The repo is read-only in this workspace, so corrected copies live in
`code_fixes/fixed/` (filenames flatten `/` → `_`, e.g. `model_gpt2.py` ==
`model/gpt2.py`). Originals are in `code_fixes/orig/`. A single unified diff for
all five files is `code_fixes/all_fixes.diff` — apply from the repo root with:

```bash
cd perrow-gradient-interference-main
git apply /path/to/all_fixes.diff        # or: patch -p1 < all_fixes.diff
```

All five patched files pass `python -m py_compile`. No behavioural change to any
number in the paper — these are label/metric/docstring corrections plus removal
of two dead attributes.

---

## 1. `model/gpt2.py` — the "GPT2-124M" mislabel (the bug you flagged)

- **`print("  GPT2-124M initialized:")`** → `print(f"  GPT2 initialized ({total/1e6:.0f}M params):")`.
  The 350M model (`GPT2Medium()`) reuses this same `GPT2` class, so every 350M
  run previously printed "GPT2-124M initialized" while building the 354M model.
  The label now derives from the actual parameter count.
- **`"GPT2-124M self-test"`** → `"GPT2 self-test"`.
- **Removed two dead `_SCALE_INIT = 1` assignments** (`o_proj` at :49 and
  `c_proj` at :67). `_SCALE_INIT` is set but **never read anywhere** in the repo
  (`grep` confirms zero reads). ⚠ **Latent issue to decide:** the comment says
  "scale down at init (residual path)" — a GPT-2-style `1/sqrt(2*n_layer)`
  residual-projection init that was evidently intended but **never
  implemented**. Removing the dead lines is correct hygiene; if you actually want
  that init, it must be added explicitly (it would change training dynamics, so
  it is NOT applied here — flagged only).

## 2. `measurement/measure_interference.py` — pre-correction docstring (most important)

- Rewrote the module docstring. The old text said the aggregate ~0 hides that
  **"60-98% of neurons receive genuinely conflicting gradient signals"** with a
  "company survey / 60% of teams miserable" analogy — the exact inversion the
  2026-06-05 correction fixed. New text states the corrected finding: per-row
  gradients are **aligned** (median cosine ~+1, ~0.3% cos<0); the aggregate ~0/
  negative reflects a **norm-weighted cancellation** by a high-norm opposed
  minority (measured: norm-profile cosine ~0.98, opposed-norm fraction 0.60/0.69),
  and cosine-based conflict diagnostics measure the wrong quantity here.
- Marked the returned **`discrepancy`** metric **DIAGNOSTIC-ONLY** in both the
  usage example and the return-dict docstring (it is ill-conditioned as
  `|global_cos| → 0` and is not a reported quantity). The field is retained so
  existing result JSONs still load; it is simply no longer presented as a result.

## 3. `experiments/phase_a_measurements.py` — PRDR gate + header

- **The go/no-go DECISION previously gated on `discrepancy > 5`** (the dropped
  PRDR ratio). Rewrote it to key on the **reported** quantity — per-row
  directional alignment (`per_row_0.3 > 0.5`) — with a comment explaining why.
- Changed the A5 matched-loss table header from `Muon PRDR / AdamW PRDR` to
  `Muon disc* / AdamW disc*` with a footnote that `disc` is a diagnostic ratio,
  not reported.

## 4. `experiments/phase_b_comparison.py` — inverted framing

- Module docstring said the experiment tests whether G_nce **"eliminates per-row
  gradient interference"** and that variant b **"causes 98% interference."**
  Rewrote to the corrected framing: the per-row gradients are **aligned**, the
  98% is the *alignment* fraction, the shared-head damage tracks
  support/capacity competition, and no capacity-separated auxiliary recovers A.
  Also relabelled variant `b` as "shared multi-offset MTP (single tied head)" to
  stop implying separate-head MTP.

## 5. `model/auxiliary_losses.py` — non-replicating T4 numbers

- Deleted the advertised **T4 numbers** ("118% damage reduction — MTP becomes
  NET POSITIVE", "4/5 seeds beat CE-only", "3.9× lower variance") — these were
  16M exploratory results that did not replicate at scale (spec §2 T1.3: must
  never reach the paper). Replaced with the actual 124M Phase-B outcome (std
  G_nce Δ+0.039; tuned (10,9) Δ+0.018 — not statistically separable from A at n=3, p≈0.19, but its point estimate is ~1 sd above A, i.e. no recovery) and corrected the
  "eliminating per-row interference" claim to "capacity-separates the auxiliary."

---

## Not changed here (require a decision or a run — flagged, not patched)

- **Two-pass bf16 drift** in `phase_c_negatives.py` (two `backward()` passes vs
  variant B's single fused pass) — a numerical-methods change that would alter
  surgery results; must be decided, not silently patched.
- **PCGrad all-params vs GS/Scatter head-only** scope confound — a design choice
  to disclose in the surgery caption, not a code bug.
- **A1/A2 LR & weight-decay mismatch** (Muon head wd=0.1/lr=3e-4 vs AdamW
  wd=0.0–0.01/lr=1e-3) — disclose or rerun matched; not a label fix.
- **47 padding vocab rows** — handled in `stats.py` (masked to 50,257); the
  measurement code itself still stores all 50,304, which is fine as raw data.
