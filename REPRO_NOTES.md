# Reproducibility notes & repo hygiene

Files in this `repro/` pack (drop into the repo root):

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned deps (torch/numpy/scipy/tiktoken/datasets + Muon-by-commit). Replaces the current **0-byte** file. |
| `README.md` | Full install/data/run/results-layout/scope. Replaces the current **2-line** file. |
| `clean_repo.sh` | Dry-run cruft report; `--apply` to delete. |
| `test_gnce_equivalence.py` | CI guard that Phase-B and Phase-D share the same `nce_loss` on the 'roll' path (no torch needed). |

## What still needs a human decision (not auto-fixable)

1. **Verify the Muon pin against your box.** `requirements.txt` pins
   `git+https://github.com/KellerJordan/Muon@056a3c5869cf#egg=muon`. That SHA is
   the last upstream commit at or before the training instance's setup date
   (2026-05-18, from `results/.instance_log`); it was inferred, not read off the
   training environment. If the box is still reachable, confirm with
   `pip show muon`. Without a pin the optimizer can silently change under a
   future `pip install`.

2. **Torch/CUDA version.** No result JSON recorded the torch or CUDA version, so
   `requirements.txt` uses a conservative range (`torch>=2.4,<2.10`). If you know
   the exact training version, pin it.

## Cruft to strip (measured in the current repo)

- **90 `*:Zone.Identifier` sidecars** (Windows "downloaded-from-internet"
  metadata) — one per file. `clean_repo.sh --apply` removes them.
- **`.instance_log`** was committed (a setup timestamp, no secrets). Removed from the
  tree, and `.gitignore` now excludes it.
- **`setup.sh` hardcoded a placeholder git identity** (`user.email "yash@research.local"`).
  Removed; the file now carries only a commented example, so the cloning user sets their own.
- **`setup.sh` created `results/phase_e` and `results/phase_f`** before either existed. It now
  creates the directories the shipped scripts actually write to: `phase_a`--`phase_e`,
  `norm_support` and `checkpoints`. There is no `phase_f`.
- **`.gitignore`** was the stock GitHub Python template and ignored neither
  `.instance_log` nor `*Zone.Identifier`. Both are now ignored, along with the LaTeX
  build intermediates under `paper/venues/*/`.

## Correction to an earlier audit note

An earlier pass said the two `GNCELoss` copies "match." They are **not**
byte-identical: `auxiliary_losses_ablation.py` adds an
`if self.neg_type == 'roll' / else` branch (to support random negatives) that
the main `auxiliary_losses.py` lacks. What matters for validity is that the
**'roll' branch is operation-identical** to the main loop — this is now
enforced by `test_gnce_equivalence.py` (verified passing against the current
repo: the roll-specialized AST equals the main `nce_loss` AST exactly).

## Verified-clean (no action needed)

- `*.pt` weight caches and `_partial_*.json` are already git-ignored.
- No credentials, tokens, or API keys found in any tracked file.
- All 5 patched source files (see `../code_fixes/`) pass `python -m py_compile`.
