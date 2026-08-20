#!/usr/bin/env python3
"""Phase E verdict: measured sweep gaps vs the mixture prediction (from estimate_kl.py)
and the linear/interference alternative. No hard-coded curve."""
import json, glob, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'results', 'phase_e')
runs = {}
for p in sorted(glob.glob(f"{D}/sweep_scale*_seed42_*.json")):
    d = json.load(open(p)); runs[float(d['mtp_scale'])] = d
if 0.0 not in runs:
    print("Need scale=0.0 (CE-only) anchor. Run run_mtp_sweep.sh first."); raise SystemExit
base = runs[0.0]['final_ce_t1']

# measured gaps
gaps = {s: runs[s]['final_ce_t1'] - base for s in sorted(runs)}
# predicted curve from estimate_kl.py (if present)
klf = os.path.join(ROOT, 'analysis', 'kl_estimate_results.json')
pred = lin = None
if os.path.exists(klf):
    kl = json.load(open(klf))
    pred = {float(k): v for k, v in kl.get('predicted_curve_mean', {}).items()}
    lin = {float(k): v for k, v in kl.get('linear_alternative', {}).items()}

print(f"{'scale':>6} {'measured gap':>13} {'mixture pred':>13} {'linear alt':>11}")
for s in sorted(runs):
    mp = f"{pred[s]:.4f}" if pred and s in pred else "  n/a"
    la = f"{lin[s]:.4f}" if lin and s in lin else "  n/a"
    print(f"{s:>6.2f} {gaps[s]:>13.4f} {mp:>13} {la:>11}")

# anchor gate
print("\n=== ANCHOR GATE (±0.04 nats) ===")
if 1.0 in runs:
    b42 = runs[1.0]['final_ce_t1']
    print(f"  scale=1.0 final CE = {b42:.4f}  vs committed B@42 = 4.399  "
          f"-> {'PASS' if abs(b42-4.399)<=0.04 else 'CHECK'}")
print(f"  scale=0.0 final CE = {base:.4f}  vs committed A (mean 3.985 ±2sd≈0.028)  "
      f"-> {'PASS' if abs(base-3.985)<=0.04 else 'CHECK (state seed-42 A if recoverable)'}")

# pre-interpretation power check (is the sweep a shape test or just a consistency check?)
if pred and lin:
    print("\n=== IS THE SHAPE TEST POWERED? (pre-interpretation) ===")
    noise = 0.018  # within-condition run sd
    for s in [0.25, 0.5]:
        if s in pred and s in lin:
            sep = abs(pred[s] - lin[s])
            print(f"  s={s}: |mixture-linear| = {sep:.4f} = {sep/noise:.1f}sigma "
                  f"-> {'POWERED shape test' if sep>2*noise else 'underpowered; treat as consistency check only'}")
    print("\n=== VERDICT LOGIC ===")
    print("  CONFIRM mixture if: T1 KL(s=1) reproduces observed 0.392 within its interval (see kl_estimate_results T1_KL_s1)")
    print("                      AND measured gaps match mixture pred pointwise within ~2*noise at all scales.")
    print("  FAVOR interference if: gaps match 0.392*s (linear) while mixture pred sits significantly elsewhere.")
    print("  Linearity ALONE favors neither (heavy-tail mixture is also near-linear).")
else:
    print("\n(run estimate_kl.py first to get the predicted curve for comparison)")
