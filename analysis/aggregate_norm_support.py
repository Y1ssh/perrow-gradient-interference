#!/usr/bin/env python3
"""
aggregate_norm_support.py — pool the norm/support runs into a headline table.

Reads every results/norm_support/norm_support_*.json, groups by optimizer, and
reports mean +/- sd of the four deciding statistics:

    norm_profile_cos       (near 0 => support divergence; near 1 => support overlaps)
    opposed_norm_fraction  (low => cancellation channel small; high => cancellation)
    global_cos             (the actual aggregate cosine)
    per_row |cos|>0.3      (the per-row alignment fraction, majority metric)

and prints the verdict per optimizer. Writes:
    analysis/norm_support_table.md
    analysis/norm_support_summary.json

Usage:
    python analysis/aggregate_norm_support.py --results results/norm_support --out analysis
"""
import os, sys, json, glob, argparse, statistics


def load_runs(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, 'norm_support_*.json'))):
        with open(p) as f:
            d = json.load(f)
        # keep only the summary stats (arrays are large; not needed here)
        runs.append({
            'file': os.path.basename(p),
            'optimizer': d.get('optimizer', 'unknown'),
            'seed': d.get('seed'),
            'global_cos': d['global_cos'],
            'norm_profile_cos': d['norm_profile_cos'],
            'opposed_norm_fraction': d['opposed_norm_fraction'],
            'per_row_0.3': d['per_row_fractions']['0.3'] if '0.3' in d['per_row_fractions']
                           else d['per_row_fractions'].get(0.3),
        })
    return runs


def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float('nan'), float('nan'), 0)
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (m, s, len(xs))


def verdict(npc_mean, onf_mean):
    if npc_mean < 0.3 and onf_mean < 0.1:
        return "SUPPORT DIVERGENCE (CE/MTP load different rows; cancellation negligible)"
    if onf_mean > 0.3:
        return "CANCELLATION (high-norm opposed minority sets the aggregate sign)"
    return "MIXED (both channels contribute)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/norm_support')
    ap.add_argument('--out', default='analysis')
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"No norm_support_*.json found in {args.results}")
        sys.exit(1)

    opts = sorted(set(r['optimizer'] for r in runs))
    summary = {}
    lines = ["# Norm / Support measurement — pooled results\n",
             "Deciding statistics per optimizer (mean +/- sd across seeds).\n",
             "- `norm_profile_cos` near 0 => support divergence; near 1 => support overlaps.",
             "- `opposed_norm_fraction` low => cancellation small; high => cancellation drives the sign.\n",
             "| Optimizer | n | norm_profile_cos | opposed_norm_fraction | global_cos | per-row \\|cos\\|>0.3 | Verdict |",
             "|---|---|---|---|---|---|---|"]

    for opt in opts:
        g = [r for r in runs if r['optimizer'] == opt]
        npc = msd([r['norm_profile_cos'] for r in g])
        onf = msd([r['opposed_norm_fraction'] for r in g])
        gc = msd([r['global_cos'] for r in g])
        pr = msd([r['per_row_0.3'] for r in g])
        v = verdict(npc[0], onf[0])
        summary[opt] = {
            'n': npc[2], 'seeds': sorted(r['seed'] for r in g),
            'norm_profile_cos': {'mean': npc[0], 'sd': npc[1]},
            'opposed_norm_fraction': {'mean': onf[0], 'sd': onf[1]},
            'global_cos': {'mean': gc[0], 'sd': gc[1]},
            'per_row_0.3': {'mean': pr[0], 'sd': pr[1]},
            'verdict': v,
        }
        lines.append(
            f"| {opt} | {npc[2]} | {npc[0]:.3f} ± {npc[1]:.3f} | "
            f"{onf[0]:.3f} ± {onf[1]:.3f} | {gc[0]:+.3f} ± {gc[1]:.3f} | "
            f"{pr[0]:.1%} | {v} |")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'norm_support_table.md'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, 'norm_support_summary.json'), 'w') as f:
        json.dump({'runs': runs, 'by_optimizer': summary}, f, indent=2)

    print("\n".join(lines))
    print(f"\nWrote {args.out}/norm_support_table.md and norm_support_summary.json")


if __name__ == '__main__':
    main()
