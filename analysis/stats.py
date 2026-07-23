#!/usr/bin/env python3
"""
stats.py — Reproducible statistics for the per-row gradient interference paper.

Reads the committed result JSONs under results/ and computes, with no manual
numbers anywhere:

  * Per-variant final_val_loss aggregates (n, mean, sd) at 124M (Phase B).
  * Welch's t-test (unequal variance) + Cohen's d (pooled) for the baseline A
    against each other variant.
  * The per-row cosine distribution summary at the final Phase-A measurement,
    with the 47 padding vocab rows (50304 - 50257) masked out.
  * The control calibration (CE vs L1).
  * The 350M single-seed point (reported as n=1, no test).

Outputs:
  stats_results.json   — machine-readable
  stats_table.md       — formatted table for the paper

Usage:
    python3 stats.py --results results --out .
"""
import argparse, glob, json, os
from collections import defaultdict

import numpy as np
from scipy import stats as sps

REAL_VOCAB = 50257           # GPT-2 BPE vocab; rows 50257..50303 are padding
PADDED_VOCAB = 50304


def load(path):
    with open(path) as f:
        return json.load(f)


def cohens_d(x, y):
    """Pooled-SD Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return (np.mean(x) - np.mean(y)) / sp


def welch(x, y):
    """Welch's t-test; returns (t, p, df_welch). NaN-safe for tiny n."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    t, p = sps.ttest_ind(x, y, equal_var=False)
    # Welch–Satterthwaite df
    vx, vy = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    return float(t), float(p), float(df)


def phase_b_by_variant(results_dir):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, "phase_b", "*.json")):
        d = load(f)
        if d.get("crashed") or "final_val_loss" not in d:
            continue
        by[d["variant"]].append((d["seed"], d["final_val_loss"]))
    return {v: sorted(s) for v, s in by.items()}


def phase_d_tuned_10_9(results_dir):
    """Full-length (30517-step) tuned (10,9) G_nce ablation seeds."""
    out = []
    for f in glob.glob(os.path.join(results_dir, "phase_d",
                                    "*layers10-9*30517steps*.json")):
        d = load(f)
        if "final_val_loss" in d:
            out.append(d["final_val_loss"])
    return sorted(out)


def perrow_summary(results_dir):
    """Per-row cosine distribution at the final Phase-A A1 measurement,
    padding rows masked."""
    a1 = load(os.path.join(results_dir, "phase_a", "a1_muon_interference.json"))
    last = a1["measurements"][-1]
    rc = np.asarray(last["row_cosines"], dtype=float)
    full_n = rc.size
    rc_real = rc[:REAL_VOCAB] if full_n >= REAL_VOCAB else rc
    return {
        "step": last["step"],
        "global_cos": last["global_cos"],
        "n_rows_raw": full_n,
        "n_rows_masked": int(rc_real.size),
        "padding_rows_removed": int(full_n - rc_real.size),
        "median_cos": float(np.median(rc_real)),
        "frac_cos_lt_0": float(np.mean(rc_real < 0)),
        "frac_abs_gt_0.3": float(np.mean(np.abs(rc_real) > 0.3)),
        "frac_gt_0.9": float(np.mean(rc_real > 0.9)),
    }


def control_summary(results_dir):
    a3 = load(os.path.join(results_dir, "phase_a", "a3_control.json"))
    return {
        "global_cos": a3.get("global_cos"),
        "per_row_0.3": a3.get("per_row_fractions", {}).get("0.3"),
    }


def scale_350m(results_dir):
    def one(sub):
        fs = glob.glob(os.path.join(results_dir, sub, "*.json"))
        return load(fs[0])["final_val_loss"] if fs else None
    a = one("phase_c_350m_r4_a")
    b = one("phase_c_350m_r4")
    return {"A_seed42": a, "B_seed42": b,
            "delta": (b - a) if (a and b) else None, "n": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    pb = phase_b_by_variant(args.results)
    vals = {v: [x[1] for x in s] for v, s in pb.items()}

    aggregates = {
        v: {"n": len(x), "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "seeds": [s for s, _ in pb[v]]}
        for v, x in vals.items()
    }

    # tuned (10,9) G_nce as an extra "variant"
    td = phase_d_tuned_10_9(args.results)
    if td:
        aggregates["gnce_tuned_10-9"] = {
            "n": len(td), "mean": float(np.mean(td)),
            "sd": float(np.std(td, ddof=1)) if len(td) > 1 else 0.0,
            "seeds": "phase_d 30517-step"}

    # A-vs-each comparisons
    A = vals.get("a", [])
    comparisons = {}
    compare_against = ["b", "b_sg", "gnce", "nextlat"]
    pool = dict(vals)
    if td:
        pool["gnce_tuned_10-9"] = td
        compare_against.append("gnce_tuned_10-9")
    for v in compare_against:
        if v in pool and A:
            t, p, df = welch(A, pool[v])
            comparisons[f"A_vs_{v}"] = {
                "delta_mean": float(np.mean(pool[v]) - np.mean(A)),
                "welch_t": t, "welch_p": p, "welch_df": df,
                "cohens_d": cohens_d(A, pool[v]),
                "n_A": len(A), "n_other": len(pool[v]),
            }

    result = {
        "aggregates_124M": aggregates,
        "A_vs_variant": comparisons,
        "per_row_distribution": perrow_summary(args.results),
        "control_calibration": control_summary(args.results),
        "scale_350M_single_seed": scale_350m(args.results),
        "note": ("Welch's t + pooled Cohen's d. 350M is n=1 (no test). "
                 "Padding rows (50257..50303) masked from per-row stats."),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stats_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # --- Markdown table ---
    lines = ["# Statistics — recomputed from committed JSONs", "",
             "## 124M final_val_loss (Phase B)", "",
             "| Variant | n | mean | sd |", "|---|---|---|---|"]
    order = ["a", "b", "b_sg", "gnce", "gnce_tuned_10-9", "nextlat"]
    for v in order:
        if v in aggregates:
            a = aggregates[v]
            lines.append(f"| {v} | {a['n']} | {a['mean']:.4f} | {a['sd']:.4f} |")
    lines += ["", "## Baseline A vs each variant (Welch's t, Cohen's d)", "",
              "| Comparison | Δmean | t | p | df | Cohen's d | n(A) | n(other) |",
              "|---|---|---|---|---|---|---|---|"]
    for k, c in comparisons.items():
        p = c["welch_p"]
        pstr = f"{p:.2e}" if (p == p and p < 1e-3) else (f"{p:.4f}" if p == p else "n/a")
        lines.append(
            f"| {k} | {c['delta_mean']:+.4f} | {c['welch_t']:.3f} | {pstr} | "
            f"{c['welch_df']:.2f} | {c['cohens_d']:.2f} | {c['n_A']} | {c['n_other']} |")
    pr = result["per_row_distribution"]
    lines += ["", "## Per-row cosine (A1, padding masked)", "",
              f"- rows: {pr['n_rows_masked']:,} (removed {pr['padding_rows_removed']} padding)",
              f"- median cos: **{pr['median_cos']:.4f}**",
              f"- frac cos<0: **{pr['frac_cos_lt_0']*100:.2f}%**",
              f"- frac |cos|>0.3: **{pr['frac_abs_gt_0.3']*100:.2f}%**",
              f"- global cos: {pr['global_cos']:+.4f}",
              f"- control (CE-vs-L1) per_row_0.3: **{result['control_calibration']['per_row_0.3']*100:.2f}%**",
              "", "## 350M (single seed — no test)",
              f"- A={result['scale_350M_single_seed']['A_seed42']:.4f}, "
              f"B={result['scale_350M_single_seed']['B_seed42']:.4f}, "
              f"Δ={result['scale_350M_single_seed']['delta']:+.4f} (n=1)"]
    with open(os.path.join(args.out, "stats_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote stats_results.json and stats_table.md")


if __name__ == "__main__":
    main()
