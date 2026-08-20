#!/usr/bin/env python3
"""
stats.py — Reproducible statistics for the per-row gradient interference paper.

Reads the committed result JSONs under results/ and computes, with no manual
numbers anywhere:

  * Per-variant final_val_loss aggregates (n, mean, sd) at 124M (Phase B).
  * Paired t-test over shared seeds (the manuscript's primary test) plus
    Welch's t-test (unequal variance) and Cohen's d (pooled) for baseline A
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


def paired(x, y):
    """Paired t-test on matched pairs; returns (t, p, df, n_pairs, mean_diff).

    The manuscript's PRIMARY test for the CE-only vs shared-MTP gap is paired
    over the seeds the two conditions share, not Welch over all seeds: the seed
    controls initialization and data order, so pairing removes that variance.
    Welch is reported as the more conservative secondary test because the seed
    sets are unequal (n=5 vs n=3).
    """
    if len(x) < 2 or len(x) != len(y):
        return (float("nan"),) * 3 + (len(x), float("nan"))
    d = np.asarray(x, float) - np.asarray(y, float)
    t, p = sps.ttest_rel(x, y)
    return float(t), float(p), float(len(d) - 1), len(d), float(np.mean(d))


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
            entry = {
                "delta_mean": float(np.mean(pool[v]) - np.mean(A)),
                "welch_t": t, "welch_p": p, "welch_df": df,
                "cohens_d": cohens_d(A, pool[v]),
                "n_A": len(A), "n_other": len(pool[v]),
            }
            # Paired test over the seeds both conditions share. This is the
            # manuscript's primary test; Welch above is the secondary one.
            sa = {s: x for s, x in pb.get("a", [])}
            sb = {s: x for s, x in pb.get(v, [])}
            shared = sorted(set(sa) & set(sb))
            if len(shared) >= 2:
                pt, pp, pdf, npair, mdiff = paired([sb[s] for s in shared],
                                                   [sa[s] for s in shared])
                entry.update({"paired_t": pt, "paired_p": pp, "paired_df": pdf,
                              "paired_n": npair, "paired_mean_diff": mdiff,
                              "paired_seeds": shared})
            comparisons[f"A_vs_{v}"] = entry

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
    ns = norm_support_restatement(args.results)
    result["norm_support_restatement"] = ns
    kl_path = os.path.join(os.path.dirname(args.results.rstrip("/")) or ".",
                           "analysis", "kl_scan_results.json")
    if not os.path.exists(kl_path):
        kl_path = os.path.join("analysis", "kl_scan_results.json")
    if os.path.exists(kl_path):
        result["shape_deviation"] = shape_deviation(kl_path)

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
    lines += ["", "## Inline prose statistics (Sections 4.1 / 4.5)", "",
              "Section 4.1 restatement, sample sd (ddof=1):", "",
              "| optimizer | opposed-norm (full) | opposed-norm (active) | "
              "norm-profile (full) | norm-profile (active) | parallel mass f |",
              "|---|---|---|---|---|---|"]
    for opt in sorted(k for k in ns if not k.startswith('_')):
        a = ns[opt]
        lines.append(
            f"| {opt} | {a['onf_full']['mean']:.4f} ± {a['onf_full']['sd_ddof1']:.4f} "
            f"| {a['onf_active']['mean']:.4f} ± {a['onf_active']['sd_ddof1']:.4f} "
            f"| {a['npc_full']['mean']:.4f} | {a['npc_active']['mean']:.4f} "
            f"| {a['parallel_mass_fraction']['min']*100:.2f}–"
            f"{a['parallel_mass_fraction']['max']*100:.2f}% |")
    for opt in sorted(k for k in ns if not k.startswith("_")):
        for r in ns[opt]["per_seed"]:
            lines.append(
                f"- {opt} seed {r['seed']}: shift "
                f"{r['onf_active'] - r['onf_full']:+.4f}, f = "
                f"{r['parallel_mass_fraction']*100:.2f}%, identity residual "
                f"{r['identity_residual']:+.4f}, opposed-mass-inside-parallel "
                f"{r['opposed_mass_inside_parallel']*100:.3f}%")
    _p = ns["_pooled"]
    lines += ["",
              f"- parallel-mass fraction f: {_p['f_mean_conforming']*100:.2f}% mean "
              f"over the {_p['n_conforming']} conforming runs, "
              f"{_p['f_mean_all']*100:.2f}% over all {_p['n_runs']}",
              f"- Welch t on opposed-norm fraction: {_p['welch_full']['t']:.2f} "
              f"(full vocabulary), {_p['welch_active']['t']:.2f} (active rows)"]
    if "shape_deviation" in result:
        sd_ = result["shape_deviation"]
        lines += ["", "Section 4.5 shape deviation from exact linearity:", ""]
        for s_key, v in sd_["per_scale"].items():
            lines.append(
                f"- s={s_key}: normalized [{v['normalized_min']:.4f}, "
                f"{v['normalized_max']:.4f}] vs linear {v['exact_linear']}; "
                f"max deviation {v['deviation_nats']:.4f} nats = "
                f"{v['deviation_sigma']:.2f}σ (band width, NOT what 4.5 quotes: "
                f"{v['band_width_nats']:.4f} nats = {v['band_width_sigma']:.2f}σ)")

    with open(os.path.join(args.out, "stats_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote stats_results.json and stats_table.md")


# ---------------------------------------------------------------------------
# Inline prose statistics (Sections 4.1 / 4.5).
#
# Every number in these two blocks appears inline in the manuscript. They live
# here so the text is generated from the same source as the figures rather than
# hand-transcribed. Convention: sd is the SAMPLE sd (ddof=1), per Section 3.4.
# ---------------------------------------------------------------------------
ACTIVE_TOL = 0.01            # |mtp_norm/ce_norm - 0.75| > tol  =>  row is active
PARALLEL_RATIO = 0.75        # inactive rows are exact 0.75 scalar multiples


def active_mask(ce, mtp, tol=ACTIVE_TOL):
    """Rows whose CE/MTP norm ratio departs from the 0.75 construction value."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ce > 0, mtp / ce, PARALLEL_RATIO)
    return np.abs(ratio - PARALLEL_RATIO) > tol, ratio


def norm_support_restatement(results_dir):
    """Section 4.1: full-vocab vs active-row norm statistics, per seed and pooled.

    Also returns the parallel-mass fraction f and checks the relation
    active = full / (1 - f). An opposed row must carry a target (rows parallel by
    construction have cos = +1 exactly), so the relation is exact under the TRUE
    target-presence partition; the ratio classifier only approximates that
    partition, and identity_residual measures the opposed mass it misfiles. The
    residual is therefore a diagnostic for the classifier, not a property of the
    data -- see opposed_mass_inside_parallel for the mass responsible.
    """
    per_opt = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "norm_support",
                                              "norm_support_*.json"))):
        d = load(path)
        ce = np.asarray(d["ce_row_norms"]); mtp = np.asarray(d["mtp_row_norms"])
        cos = np.asarray(d["row_cosines"]); w = ce * mtp
        act, _ = active_mask(ce, mtp)
        f_par = float(w[~act].sum() / w.sum())
        row = dict(
            seed=d["seed"],
            # per-row direction statistics on the active-row denominator: these are
            # the Layer-2 quantities the main-text table prints
            active_fraction=float(act.mean()),
            active_median_cos=float(np.median(cos[act])),
            active_opposed_count_frac=float((cos[act] < 0).mean()),
            aggregate_cos=float(d["global_cos"]),
            onf_full=float(w[cos < 0].sum() / w.sum()),
            onf_active=float(w[act][cos[act] < 0].sum() / w[act].sum()),
            npc_full=float(ce @ mtp / (np.linalg.norm(ce) * np.linalg.norm(mtp))),
            npc_active=float(ce[act] @ mtp[act]
                             / (np.linalg.norm(ce[act]) * np.linalg.norm(mtp[act]))),
            parallel_mass_fraction=f_par,
            # mass that is opposed yet classified parallel: the misclassification
            # that makes the classifier a generic rather than exact target proxy
            opposed_mass_inside_parallel=float(
                w[~act][cos[~act] < 0].sum() / w.sum()),
        )
        row["identity_pred_active"] = row["onf_full"] / (1.0 - f_par)
        row["identity_residual"] = row["onf_active"] - row["identity_pred_active"]
        per_opt[d["optimizer"]].append(row)

    out = {}
    for opt, rows in per_opt.items():
        rows.sort(key=lambda r: r["seed"])
        agg = {"per_seed": rows}
        for key in ("onf_full", "onf_active", "npc_full", "npc_active",
                    "parallel_mass_fraction", "active_fraction",
                    "active_median_cos", "active_opposed_count_frac",
                    "aggregate_cos"):
            v = [r[key] for r in rows]
            agg[key] = {"mean": float(np.mean(v)),
                        "sd_ddof1": float(np.std(v, ddof=1)),
                        "min": float(min(v)), "max": float(max(v))}
        shifts = [r["onf_active"] - r["onf_full"] for r in rows]
        agg["shift"] = {"per_seed": shifts, "mean": float(np.mean(shifts)),
                        "all_positive": bool(all(s > 0 for s in shifts))}
        # conforming runs = those whose identity residual is within the quoted 0.008;
        # the prose quotes f both ways because the outlier moves the mean materially
        conf = [r for r in rows if abs(r["identity_residual"]) <= 0.008]
        agg["identity"] = {
            "max_residual_conforming": float(max((abs(r["identity_residual"])
                                                  for r in conf), default=0.0)),
            "n_conforming": len(conf),
            "outlier_seeds": [r["seed"] for r in rows if r not in conf],
            "f_range_conforming": [float(min((r["parallel_mass_fraction"]
                                              for r in conf), default=0.0)),
                                   float(max((r["parallel_mass_fraction"]
                                              for r in conf), default=0.0))],
        }
        out[opt] = agg

    # pooled across optimizers: the two f means the Section 4.1 mass claim quotes
    allrows = [r for rows in per_opt.values() for r in rows]
    conf = [r for r in allrows if abs(r["identity_residual"]) <= 0.008]
    out["_pooled"] = {
        "f_mean_all": float(np.mean([r["parallel_mass_fraction"] for r in allrows])),
        "f_mean_conforming": float(np.mean([r["parallel_mass_fraction"]
                                            for r in conf])),
        "n_runs": len(allrows), "n_conforming": len(conf),
    }
    # Welch on the opposed-norm fraction, both denominators: the appendix names
    # which one its t belongs to, so both are emitted here
    if len(per_opt) == 2:
        (o1, r1), (o2, r2) = sorted(per_opt.items())
        for key, tag in (("onf_full", "welch_full"), ("onf_active", "welch_active")):
            t, pv = sps.ttest_ind([r[key] for r in r1], [r[key] for r in r2],
                                  equal_var=False)
            out["_pooled"][tag] = {"t": float(abs(t)), "p": float(pv),
                                   "groups": [o1, o2]}
    return out


def shape_deviation(kl_scan_path, observed_gap=None, sigma=None):
    """Section 4.5: departure of the predicted KL curve from exact linearity.

    Reported as max |normalized_KL(s) - s| over estimator variants, rescaled to
    the measured s=1 gap. This is the deviation from linearity, NOT the width of
    the band across variants; the two differ and only the former is what the
    sentence in 4.5 claims.
    """
    kl = load(kl_scan_path)
    gap = observed_gap if observed_gap is not None else kl["observed_sweep_gap"]["1.0"]
    sig = sigma if sigma is not None else kl["gap_noise_sd"]
    out = {"observed_gap": gap, "sigma": sig, "per_scale": {}}
    for s_key, s_val in (("0.25", 0.25), ("0.5", 0.50)):
        norm = [r[s_key] / r["1.0"] for r in kl["kl_estimates"]]
        lo, hi = min(norm), max(norm)
        dev = max(abs(hi - s_val), abs(lo - s_val))
        out["per_scale"][s_key] = {
            "normalized_min": lo, "normalized_max": hi,
            "exact_linear": s_val,
            "max_deviation_from_linear": dev,
            "deviation_nats": dev * gap,
            "deviation_sigma": dev * gap / sig,
            "band_width_normalized": hi - lo,
            "band_width_nats": (hi - lo) * gap,
            "band_width_sigma": (hi - lo) * gap / sig,
        }
    return out


if __name__ == "__main__":
    main()
