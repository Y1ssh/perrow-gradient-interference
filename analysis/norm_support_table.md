# Norm / Support measurement — pooled results

Deciding statistics per optimizer (mean +/- sd across seeds).

- `norm_profile_cos` near 0 => support divergence; near 1 => support overlaps.
- `opposed_norm_fraction` low => cancellation small; high => cancellation drives the sign.

| Optimizer | n | norm_profile_cos | opposed_norm_fraction | global_cos | per-row \|cos\|>0.3 | Verdict |
|---|---|---|---|---|---|---|
| adamw | 3 | 0.977 ± 0.007 | 0.692 ± 0.011 | -0.317 ± 0.020 | 98.6% | CANCELLATION (high-norm opposed minority sets the aggregate sign) |
| muon | 3 | 0.979 ± 0.003 | 0.604 ± 0.026 | -0.163 ± 0.030 | 98.3% | CANCELLATION (high-norm opposed minority sets the aggregate sign) |
