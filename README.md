# Aligned but Apart: Why Cosine Conflict Diagnostics Misread Auxiliary-Loss Interference at the Output Projection

*Per-row gradient geometry at a shared output projection.*

Code and committed results for the paper studying how a next-token
cross-entropy (CE) loss and an auxiliary multi-token-prediction (MTP) loss
interact at a language model's **shared output projection** (the tied
`lm_head`).

**Headline finding.** At the shared output head, the per-row gradients of CE and
MTP are near-perfectly **aligned in direction** (median per-row cosine ≈ +1;
only ~0.3% of rows have cosine < 0), even though the **aggregate** (flattened)
cosine reads ≈ 0 to negative. Instrumenting the per-row gradient **norms**
directly (n=3 seeds per optimizer) shows why: the two losses load their
magnitude on the **same** rows (norm-profile cosine ≈ 0.98, so support does
*not* diverge), yet the majority of the gradient-magnitude mass (≈60% under
Muon, ≈69% under AdamW) sits on the tiny high-norm **opposed** minority. The
near-zero aggregate is therefore a **norm-weighted cancellation** by a high-norm
opposed minority, not a broad directional conflict. Standard cosine-based
conflict diagnostics *misread* this: the aggregate is set by per-row gradient
**norms**, not by the directions it averages. Adding shared-head MTP degrades
next-token CE (+0.39 nats at 124M), and neutralizing the measured opposition
(gradient surgery) does not repair it.

> **Scope.** All results are on a single **shared tied output head**, GPT-2 at
> **124M** (primary, n=5 seeds) and **350M** (single illustrative seed),
> undertrained (≈3.9 tokens/param), batch = 16×1024 = 16K tokens/step. See
> "Scope & caveats" below and the paper's Discussion.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# install torch for YOUR cuda first (see requirements.txt header), then:
pip install -r requirements.txt
```

Muon has no PyPI version — **pin it to a commit** in `requirements.txt` for
reproducibility (the bare git URL installs HEAD, which may drift).

## Data

FineWeb, streamed via HuggingFace `datasets` and tokenized with the GPT-2 BPE
(`tiktoken`). Token budgets differ by phase and are recorded in each result
JSON (`total_tokens`, `unique_train_tokens`):

| Phase | Scale | Tokens (train/val) |
|---|---|---|
| A (measurement) | 124M | 48M / 2M |
| B (comparison)  | 124M | 480M / 20M |
| 350M (r4)       | 354M | 1369M / 20M |

Train/val splits are disjoint (Phase B: `train[:480M]`, `val[480M:500M]`).

## Reproduce

```bash
# 1. Per-row measurements (Figures 1–3, 5)
python experiments/phase_a_measurements.py            # A1–A5: Muon, AdamW, control, untied, matched

# 2. Core intervention comparison (Figure 4)  — one run at a time, crash-safe
for v in a b b_sg gnce nextlat; do
  for s in 42 123 456 789 1337; do
    python experiments/phase_b_comparison.py --variant $v --seed $s
  done
done

# 3. G_nce hyperparameter ablations (Phase D)
python experiments/phase_d_ablations.py --layers 10 9 --seed 42 --steps 30517

# 4. Statistics + figures (from committed JSONs — no GPU needed)
python analysis/stats.py    --results results --out analysis   # Welch t, Cohen's d, per-row stats
python figures/make_figures.py --results results --out figures # regenerates fig1–fig6 (data figures)
python figures/make_schematic.py --out figures                 # regenerates fig0 (conceptual schematic)

# 5. Surgery / AdamW negatives (Phase C)
python experiments/phase_c_negatives.py --method gs --optimizer muon --variant b --seed 42

# 6. MTP-weight sweep + mixture-optimum diagnostics (Phase E) — one GPU, ~3.5 h
#    Rules the "mixture-optimum" alternative in or out; logs t+1/t+2/t+3 CE + entropy.
bash experiments/run_mtp_sweep.sh          # runs scale 0.0, 1.0, 0.1, 0.01 at seed 42
python analysis/analyze_mtp_sweep.py       # prints verdict table (no GPU)
```

## Code layout

```
model/        gpt2.py, gpt2_medium.py, auxiliary_losses.py (+ _ablation)
measurement/  measure_interference.py       — per-row CE-vs-MTP cosine on the head
              measure_norm_support.py       — NEW: also logs per-row NORMS and returns
                                              norm_profile_cos + opposed_norm_fraction,
                                              the two statistics that separate support
                                              divergence from a high-norm opposed minority.
                                              Runs on the short diagnostic pass.
experiments/  phase_a..d drivers
              phase_e_mtp_weight_sweep.py — MTP-weight sweep + t+1/t+2/t+3 CE and
                                            output-entropy logging (mixture-optimum test);
                                            run_mtp_sweep.sh drives the 4 runs
baselines/    gs_muon, pcgrad_muon, scatter_muon (gradient surgery)
analysis/     stats.py       — Welch t-test, Cohen's d, per-row distribution (masks 47 padding rows)
              stats_table.md — regenerated summary table
figures/      make_figures.py   — regenerates fig1–fig6 (png+pdf+csv) from results/
              make_schematic.py — regenerates fig0, the conceptual mechanism
                                  schematic (not data-derived; used as paper Fig. 1)
              analyze_mtp_sweep.py — reads results/phase_e/, prints the sweep verdict
tests/        test_gnce_equivalence.py — torch-free AST guard: the GNCE 'roll' path is
                                         identical across auxiliary_losses{,_ablation}.py
paper/        main.tex + sections/ + references.bib + TMLR style; main.pdf (15 pp);
              figures/ holds the fig0–fig6 PDFs the manuscript includes. See paper/README.md.
docs/         ship checklist, readiness audit, code-fix log, bibliography verification,
              peer-review syntheses, external-review triage, claim ledger.
```

## Results layout

```
results/
  phase_a/                 5 JSONs  — per-row measurement (A1–A5)
  phase_b/                19 JSONs  — 124M A/B/B_sg/G_nce/NextLat × seeds
  phase_b_50M_repeated/    9 JSONs  — 50M-token robustness repeat
  phase_c/                10 JSONs  — gradient surgery + AdamW
  phase_c_350m_r4_a/       1 JSON   — 350M variant A (87k steps)
  phase_c_350m_r4/         1 JSON   — 350M variant B (87k steps)
  phase_c_350m/            1 JSON   — SUPERSEDED short 350M run (10.7k steps)
  phase_e/                 (created by run_mtp_sweep.sh) — MTP-weight sweep results
  phase_d/                20 JSONs  — G_nce ablation grid
```
66 committed result JSONs total. Every headline number in the paper is
recomputed from these by `analysis/stats.py`.

## Scope & caveats (read before citing)

- **Shared tied output head only.** This is *not* separate-head MTP as in
  Gloeckle et al. 2024 / DeepSeek-V3 — those use independent per-horizon heads.
  Our "MTP" supervises one tied head for t+1/t+2/t+3 jointly.
- **350M is a single seed** — treated as illustrative, not a scale law.
- **Surgery baselines** (Phase C) are single-seed and use two backward passes
  under bf16 (vs the fused single pass in Phase B); the ~0.02–0.04 nat numerical
  drift is comparable to the surgery effect. Reported with that caveat.
- **Muon vs AdamW** in Phase A differ in LR/weight-decay; the optimizer
  comparison is qualitative.
- The `discrepancy` field in result JSONs is a **diagnostic-only** ratio (not
  reported); see `measurement/measure_interference.py`.

## Repository hygiene

Before pushing a clean snapshot, run `clean_repo.sh` (strips
`*:Zone.Identifier` sidecars, `.instance_log`, and the placeholder git identity
in `setup.sh`). See `REPRO_NOTES.md`.
