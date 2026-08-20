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

## Inline prose statistics (Sections 4.1 / 4.5)

Section 4.1 restatement, sample sd (ddof=1):

| optimizer | opposed-norm (full) | opposed-norm (active) | norm-profile (full) | norm-profile (active) | parallel mass f |
|---|---|---|---|---|---|
| adamw | 0.6918 ± 0.0115 | 0.7085 ± 0.0095 | 0.9766 | 0.9762 | 2.82–3.53% |
| muon | 0.6041 ± 0.0256 | 0.6079 ± 0.0201 | 0.9788 | 0.9775 | 2.34–11.52% |
- adamw seed 42: shift +0.0174, f = 3.53%, identity residual -0.0079, opposed-mass-inside-parallel 0.765%
- adamw seed 123: shift +0.0144, f = 2.82%, identity residual -0.0060, opposed-mass-inside-parallel 0.583%
- adamw seed 456: shift +0.0183, f = 2.82%, identity residual -0.0015, opposed-mass-inside-parallel 0.142%
- muon seed 42: shift +0.0232, f = 3.89%, identity residual -0.0014, opposed-mass-inside-parallel 0.131%
- muon seed 123: shift -0.0253, f = 11.52%, identity residual -0.1070, opposed-mass-inside-parallel 9.472%
- muon seed 456: shift +0.0136, f = 2.34%, identity residual -0.0002, opposed-mass-inside-parallel 0.015%

- parallel-mass fraction f: 3.08% mean over the 5 conforming runs, 4.49% over all 6
- Welch t on opposed-norm fraction: 5.42 (full vocabulary), 7.82 (active rows)

Section 4.5 shape deviation from exact linearity:

- s=0.25: normalized [0.2433, 0.2674] vs linear 0.25; max deviation 0.0065 nats = 0.25σ (band width, NOT what 4.5 quotes: 0.0090 nats = 0.35σ)
- s=0.5: normalized [0.5105, 0.5382] vs linear 0.5; max deviation 0.0142 nats = 0.56σ (band width, NOT what 4.5 quotes: 0.0103 nats = 0.40σ)
