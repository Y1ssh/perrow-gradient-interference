# Step 9 fold-pack audit against repo data (2026-07-26)

Every fold-pack number checked against `analysis/kl_scan_results.json`,
`analysis/ceilings_results.json`, `results/phase_e/sweep_scale*.json`, and
`results/norm_support/norm_support_muon_seed42.json`. Read-only audit.

## VERIFIED CORRECT (fold as written)

| Claim | Fold pack | Repo data |
|---|---|---|
| Anchor s=0 | 3.9941 (D0.009) | 3.994140625 |
| Anchor s=1 | 4.3656 (D0.024) | 4.365625 |
| Sweep gaps | 0.0914 / 0.1832 / 0.3715 | 0.09141 / 0.18320 / 0.37148 |
| Band @k=2048 s=1 | [0.232, 0.336] | [0.23158, 0.33634] |
| Coverage extrapolation | 0.330, slope ~0.072 | 0.3306, slopes 0.0729 / 0.0676 |
| Escalation not triggered | D(1024->2048)=0.0016<0.01 | 0.0015 |
| Residual raw | +0.035 | +0.0351 |
| Residual extrapolated | +0.042 | +0.0409 |
| vs headline 0.392 | 0.056-0.062 | +0.0557 / +0.0614 |
| Continuity k=256 p3eqp2 | 0.342 in [0.32,0.36] | 0.3420 PASS |
| Continuity k=256 drop | ~0.23 in [0.21,0.26] | 0.2416 / 0.2359 PASS |
| T2-proper | +0.950 pred vs +1.256 obs (~76%) | 0.94966 / 1.25631 = 75.6% |
| T3 | t+2 by 2.913, t+3 by 2.566 | 2.91328 / 2.56641 |
| Control CE-vs-L1 active | median -0.029, 1.0% >0.3 | -0.02911, 1.0276% |
| Control full-vocab | 36.2% | 36.198% |
| CE-vs-CE active median | -0.236 / -0.231 | -0.23648 / -0.23125 |
| CE-vs-CE full median | +0.71 | 0.70506 / 0.72498 |
| CE-vs-CE full >0.3 | 91-93% | 93.33% / 91.20% |
| 3-batch robustness | 8.9%/0.541, 14.5%/0.539 | exact match |
| Robustness opposed | 3.0% / 3.1% | 3.011% / 3.121% |
| Active >0.3 (CE-vs-MTP) | 83% | 83.1% (cos>0.3); 84.0% (abs) |
| Gate item 6 (old file) | all 12 variants present | CONFIRMED by opening `analysis/kl_estimate_results.json`: 12 rows = k{64,128,256} x {drop,p3eqp2} x halves{0,1}, no missing combination; its T1 interval [0.2309, 0.3528] matches the paper's cited [0.231, 0.353] |
| k-scan file completeness | (not a gate item) | `analysis/kl_scan_results.json`: 12 rows = k{256,1024,2048} x 2 modes x 2 halves |

## CORRECTIONS REQUIRED BEFORE FOLDING (4 substantive + 1 presentation)

1. **n_positions is 2,048 per estimate, NOT 4,096.** Block 2 prose says
   "4,096 held-out positions". `--n_positions` defaults to 4096 but the code
   passes `args.n_positions // 2` per split-half, and every row records
   `n_positions: 2048`. Correct prose: "2,048 held-out positions per
   validation half (4,096 total across the split)".

2. **Coverage at k=2048 is 0.90-0.92, not 0.91-0.92.** Actual range
   0.9045-0.9155. The ledger's "0.905-0.915" is right; block-2 prose
   "coverage 0.91-0.92" understates the spread. Use 0.90-0.92.

3. **Gate item 9 cites STALE coverage (0.81-0.83).** That was the first run
   (k<=256). With the k-scan the figure to report wherever T1 is cited is
   0.90-0.92. Internal contradiction inside the fold pack.

4. **Abstract's "62-95%" is not supported; use 62-91%.** Band at k=2048 is
   62.3%-90.5% of the matched gap; across all k the max is 92.1%. Nothing
   reaches 95%.

5. **Sigma depends on which sigma you quote; state it explicitly (presentation,
   not an error).** The fold pack quotes sigma ~= 0.025-0.03, and "~1.4 sigma"
   is CORRECT at the upper end: extrapolated residual 0.0409 / 0.03 = 1.36.
   With the script's own `gap_noise_sd = 0.02546` the same residual is 1.61
   sigma, and the raw residual 0.0351 is 1.38 sigma. All readings sit in
   (1 sigma, 2 sigma], so **branch (b) is unchanged either way**. The paper
   should name the sigma it uses rather than a bare multiple, e.g. "residual
   0.035-0.041 nats, i.e. 1.4-1.6 sigma for run noise sigma = 0.025-0.03".

## STILL MISSING / OPEN IN THE REPO

- **Soft-label material is NOT in the paper.** Blocks 1 and 2 are unwritten.
  The only trace is a DANGLING forward reference at
  `paper/sections/results.tex:217` pointing at `sec:interventions`, which is
  not the soft-label section. Must resolve when block 1 lands.
- **Sweep/KL figure does not exist.** Block 2 asks for measured-gaps-vs-s with
  the T1 band overlaid. No such file in `figures/`, and `make_figures.py` has
  no sweep function (grep count 0), so a fresh clone cannot build it.
- **Stale 0.42% control citations remain** at `results.tex:24`, `results.tex:44`,
  `main.tex:87` (block 4 replaces these with the 1.0% matched control).
- **Fig 2 caption ghost "Right:" clause** at `results.tex:44` (block 4).
- **"approx 25 percentage points"** at `results.tex:339` needs correcting to
  ~10pp (gate item 2).
- **`measure_ceilings.py:128` prints a wrong gloss.** Since the measured
  -0.236 is not > 0.7 the script prints "near ceiling", but the fold pack
  establishes this quantity is a FLOOR. The printed word is misleading even
  though the committed number is correct.

## REPO HYGIENE

- **CRLF line endings**: 24 of the first 200 tracked files are CRLF (Windows
  zip round-trip). All 49 modified files show real diffs; the `.tex`/`.md`/
  `.py` edits are genuine, and the JSON/CSV diffs are float re-serialization
  (numerically verified identical: all 6 norm_support JSONs, max array
  difference 0.0).
- **348 `*Zone.Identifier` files** present (not in `.git/`). Covered by
  `.gitignore` (2 patterns) so they will not ship, but they clutter the tree.
- **git**: real checkout, HEAD `7580180` (merge commit), 49 modified, 0
  untracked. The k-scan outputs ARE present locally.
- **Gate item 7 is now CLOSED**: `analysis/ceilings_results.json` exists
  (1551 bytes) with control, both ceilings, and the 2-point robustness loop.
