# Statistics — recomputed from committed JSONs

## 124M final_val_loss (Phase B)

| Variant | n | mean | sd |
|---|---|---|---|
| a | 5 | 3.9851 | 0.0138 |
| b | 3 | 4.3767 | 0.0209 |
| b_sg | 3 | 4.4927 | 0.0101 |
| gnce | 5 | 4.0241 | 0.0207 |
| gnce_tuned_10-9 | 3 | 4.0026 | 0.0156 |
| nextlat | 3 | 4.0365 | 0.0179 |

## Baseline A vs each variant (Welch's t, Cohen's d)

| Comparison | Δmean | t | p | df | Cohen's d | n(A) | n(other) |
|---|---|---|---|---|---|---|---|
| A_vs_b | +0.3915 | -28.838 | 7.58e-05 | 3.08 | -23.68 | 5 | 3 |
| A_vs_b_sg | +0.5076 | -59.725 | 5.54e-09 | 5.53 | -39.96 | 5 | 3 |
| A_vs_gnce | +0.0390 | -3.509 | 0.0099 | 6.98 | -2.22 | 5 | 5 |
| A_vs_nextlat | +0.0513 | -4.267 | 0.0176 | 3.47 | -3.36 | 5 | 3 |
| A_vs_gnce_tuned_10-9 | +0.0175 | -1.599 | 0.1870 | 3.89 | -1.21 | 5 | 3 |

## Per-row cosine (A1, padding masked)

- rows: 50,257 (removed 47 padding)
- median cos: **1.0000**
- frac cos<0: **0.33%**
- frac |cos|>0.3: **98.38%**
- global cos: -0.0820
- control (CE-vs-L1) per_row_0.3: **0.42%**

## 350M (single seed — no test)
- A=3.7516, B=4.1291, Δ=+0.3775 (n=1)
