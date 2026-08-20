# GPU run brief — co-location and CE-only baseline

Two passes. Together they close structural gaps 1 and 2 in `PAPER_SPINE.md`, the
two largest holes left in the argument. Everything else in the paper is already
supported by committed data; **these runs are optional upgrades, not blockers.**

Both use `measurement/run_norm_support.py`, which now takes `--steps`,
`--loss-mode`, `--measure-at` and `--tag`. Defaults reproduce the committed
step-1000 runs byte-for-byte, so nothing already in the repo changes.

---

## Why these two

**Gap 1 — co-location (largest).** Every per-row measurement is at step 1,000.
Every loss comparison is at step 30,517. So the paper never exhibits the
aligned-gradient state and the degraded-loss state *in the same measurement* at
124M. A reviewer can reasonably ask whether the alignment simply decays by the
time the loss gap opens. Currently answered only by a single-seed 350M
observation.

**Gap 2 — no CE-only per-row baseline at 124M.** The cleanest control is the same
instrument on a model that never saw the auxiliary. Note what this does and does
not test: the measurement always compares a CE gradient against an MTP gradient,
so the question is whether the per-row alignment is a property of the *loss pair*
or an artifact of *having trained on* the pair. Without it, "the two losses agree
per-row" cannot be separated from "training on both made them agree."

---

## Pass A — co-location (the important one)

```bash
cd ~/perrow-gradient-interference
python measurement/run_norm_support.py \
  --optimizer muon --seed 42 \
  --steps 30517 \
  --measure-at 1000,5000,10000,20000 \
  --tag _colocated
```

Trains the shared-MTP variant to the same 30,517 steps as the loss experiments,
taking the full per-row measurement at five points. Output:
`results/norm_support/norm_support_muon_seed42_colocated.json`, containing a
`snapshots` list plus the final measurement in the usual top-level fields.

**Hardware.** A single NVIDIA H100, rented through the Prime Intellect compute
exchange on hardware operated by Verda (DataCrunch). Prime Intellect is an
exchange rather than a datacenter operator, so the provider shown for an
instance is the party that actually runs it.

**Runtime.** The committed 1,000-step run took **121.5 s** wall on the H100
(`total_time` in `norm_support_muon_seed42.json`). 30,517 steps is ~30x that, so
budget **≈1 hour** of training plus a few seconds per snapshot — call it 1.5 h
with data loading. Cheaper than it looks; run it first and in the background.

**What each outcome means — pre-register this before looking:**

| result at step 30,517 | reading |
|---|---|
| active-row median stays ≈0.5, npc stays ≈0.98 | gap 1 closes; the assumption was right and the paper can state it as measured |
| median decays toward 0 but npc stays high | alignment is a *transient*; §4.1's claim must be scoped to early training. This would be a substantive correction, not a tweak |
| npc itself falls | the mechanism changes over training; L4's branch selection is step-dependent and must say so |

The third outcome would be the most interesting and the most work. Do not
pre-judge which one shows up.

## Pass B — CE-only per-row baseline

```bash
python measurement/run_norm_support.py \
  --optimizer muon --seed 42 \
  --loss-mode ce_only \
  --tag _ceonly
```

1,000 steps, ~2 min (same budget as the committed runs). Trains with cross-entropy alone, then runs the identical
CE-vs-MTP per-row measurement.

**Pre-registered reading:** if the active-row median and `norm_profile_cos` land
near the shared-MTP values (≈0.53 and ≈0.98), the alignment is a property of the
loss pair and the instrument, not of having trained on both — which strengthens
§4.1 considerably. If they differ materially, the alignment is partly *induced by
training on the auxiliary*, and §4.1 must say so. Either result is publishable;
they are not equally convenient, which is the point of writing this down first.

## Optional Pass C — control per-row array (cheap)

Gap 3 in the spine is a threshold-sensitivity caveat that cannot be resolved
because `results/phase_a/a3_control.json` committed only four binned fractions,
not the per-row cosines. Any future control run should commit `row_cosines`. If
Pass B is run anyway, its output already contains the full array, so this may
come free.

---

## After the runs

1. `python analysis/stats.py --results results --out analysis` — regenerates
   `stats_results.json`, which the manuscript's table reads from.
2. `python figures/make_figures.py --results results --analysis analysis --out figures`
3. `python tests/check_paper_claims.py` — must print `N/N checks passed`. The gate
   recomputes every printed number from the JSONs, so if a fold changed a value
   without updating the text, this fails.
4. Recompile: `cd paper && tectonic -X compile main.tex`.

Send back the two JSONs (or push them); the fold into the manuscript is text-only
and needs no GPU.

## Do not

- Change `max_tokens=50_000_000` in the data loader. It is a *pool*, not a budget;
  ~16.4M tokens are consumed. Changing it stops the run reproducing A1.
- Re-run the committed step-1000 configurations without `--tag`. The default
  filename is the one already in the repo and will be overwritten.
- Treat a favourable Pass A result as confirming causation. It closes a *timing*
  gap. Causation between norm-weighted opposition and the loss increase is
  Forbidden at every layer of the spine and stays that way.
