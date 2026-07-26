# Active-row diagnosis (round-2 review, Parts C/D/E/F)

## The construction floor
Because CE and shared-MTP read the SAME logits, a vocabulary row `i` that receives
**no target** in the measurement batch has, exactly,
`g_mtp_i = (0.5+0.25)*g_ce_i = 0.75*g_ce_i` — a scalar multiple, so its per-row
cosine is `+1` by construction, independent of anything the model learned. This is
verified: among the bottom 88% of rows by gradient norm, the ratio `||g_mtp||/||g_ce||`
departs from 0.75 in only **0.1%** of cases.

## Classification by construction (not by cosine threshold)
A row is **active** iff `|(||g_mtp||/||g_ce||) - 0.75| > 0.01`. This closes the
categories (previously 75% parallel + 12% active = 87%, missing 13%):
- The "missing 13%" were essentially-parallel rows whose *computed* cosine fell in
  (0.99, 1-1e-6] due to bf16/fp32 rounding on tiny softmax-tail norms — they are
  parallel-by-construction and now classify as parallel by the ratio test.
- **parallel + active = 100%.** (Muon seed42: 92.2% parallel + 7.8% active.)

## Active-row statistics (n=3 per optimizer, classifier above)
| Optimizer | active frac | active-row median cos | active %opposed | active %\|cos\|>0.3 |
|---|---|---|---|---|
| Muon  | 8.1 ± 0.2% | 0.529 ± 0.006 | 3.71 ± 0.17% | 83.1 ± 0.7% |
| AdamW | 8.0 ± 0.2% | 0.547 ± 0.013 | 3.17 ± 0.18% | 84.0 ± 1.4% |

Full-vocab (for contrast): median +1.00, %opposed 0.3%, %|cos|>0.3 98.4%.

**The active fraction is batch-relative** (8% here; an earlier single-batch phase_a
measurement read ~12%/6,063 rows) — the exact value depends on how many distinct
tokens appear as targets in the measurement batch (Heaps' law). This is why we add a
multi-batch robustness line (§3.2). What is *stable* across classifier tolerance and
seeds is the **collapse of the median from +1.00 to ~0.53** and the rise in %opposed
from 0.3% to ~3%.

## Mechanism is INVARIANT to the correction (the key point)
Active rows carry **96%** of all gradient-magnitude mass; parallel rows carry ~4%.
So the mass-weighted statistics are unchanged whether computed full-vocab or
active-only:
- `norm_profile_cos`: full 0.979/0.977 → active-only 0.978/0.984
- `opposed_norm_fraction`: full 0.604/0.692 → active-only 0.630/0.709

The active-row correction sharpens the honest denominator without touching the
paper's actual claim. The +1 median was inflated by construction; the mechanism
(a high-norm opposed minority sets the aggregate) stands.

## "the rest" fix (Part C internal inconsistency)
Abstract said "the rest are scalar multiples" where "the rest" read as 88% but §4.1
said 75%. Resolved: parallel-by-construction = ~88-92% (ratio test, folding in the
fp-noisy rows); active = ~8-12% (batch-relative). Both the abstract and §4.1 now use
the ratio-based split and agree.
