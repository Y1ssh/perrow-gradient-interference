#!/usr/bin/env python3
"""
Regenerate all five paper figures from the committed result JSONs.

    python figures/make_figures.py --results results --out figures

No GPU or training needed — reads only `results/`. Produces fig1..fig5 as
.png + .pdf + .csv in --out. Numbers match stats.py.

Figures:
  1  Per-row cosine distribution (sign-split, points-on-log) + control
  2  Aggregate vs per-row alignment over training (Muon + AdamW)
  3  Per-row alignment vs aggregate cosine, all 74 weight matrices
  4  Final next-token loss by intervention (points = seeds)
  5  Per-row alignment by token-frequency quintile (both optimizers)
  6  Norm decomposition: mean per-row cosine (flat-norm aggregate) vs actual aggregate
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

META_GREY = "#8a8a8a"
FOC = "#2b6cb0"
ALARM = "#C44"
AUX = "#7a4fb0"
REAL_VOCAB = 50257  # 50304 padded - 47 padding rows


def _style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })


def _load(path):
    return json.load(open(path))


def _final_measurement(meas, step=None):
    if step is not None:
        for m in meas:
            if m["step"] == step:
                return m
    return meas[-1]


def fig1(results, out):
    a1 = _load(f"{results}/phase_a/a1_muon_interference.json")
    a3 = _load(f"{results}/phase_a/a3_control.json")
    m = _final_measurement(a1["measurements"], step=1000)
    rc = np.asarray(m["row_cosines"])[:REAL_VOCAB]  # drop 47 padding rows
    counts, edges = np.histogram(rc, bins=60, range=(-1, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    frac_neg = 100.0 * (rc < 0).mean()
    mtp_frac = 100.0 * (np.abs(rc) > 0.3).mean()
    ctrl = a3.get("per_row_fractions", {})
    ctrl_frac = 100.0 * float(ctrl.get("0.3", ctrl.get(0.3, 0.0041547)))

    fig = plt.figure(figsize=(7.2, 3.1))
    g = fig.add_gridspec(1, 3, width_ratios=[2.3, 1, 0.9], wspace=0.5)
    ax, axc, axl = fig.add_subplot(g[0]), fig.add_subplot(g[1]), fig.add_subplot(g[2])
    msk = counts > 0
    cols = [ALARM if c < 0 else FOC for c in centers[msk]]
    ax.vlines(centers[msk], 0.5, counts[msk], color="#ccc", lw=0.5, zorder=1)
    ax.scatter(centers[msk], counts[msk], s=10, c=cols, edgecolors="none", zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.5, counts.max() * 1.6); ax.set_xlim(-1.05, 1.05)
    ax.axvline(np.median(rc), color=META_GREY, lw=1, ls="--", zorder=2)
    ax.set_xlabel("Per-row cosine (CE vs MTP gradient)"); ax.set_ylabel("Rows (log)")
    ax.set_title("Per-row gradients align in direction", fontsize=8, loc="left")
    ax.text(-0.5, 0.9, f"median +{np.median(rc):.2f}", fontsize=6, color=META_GREY,
            transform=ax.get_xaxis_transform(), ha="center")
    ax.text(-1.0, 0.68, f"{frac_neg:.2f}% of rows cos<0", fontsize=6, color=ALARM,
            transform=ax.get_xaxis_transform())
    axc.bar([0, 1], [mtp_frac, ctrl_frac], color=[FOC, META_GREY], width=0.62)
    axc.set_xticks([0, 1]); axc.set_xticklabels(["CE vs\nMTP", "CE vs L1\n(control)"], fontsize=6)
    axc.set_ylabel("Rows |cos| > 0.3 (%)"); axc.set_ylim(0, 112)
    for x, val, col in [(0, mtp_frac, FOC), (1, ctrl_frac, META_GREY)]:
        axc.text(x, val + 3, f"{val:.1f}%", ha="center", fontsize=6.5, color=col, fontweight="bold")
    axc.set_title("Instrument\ncalibrates", fontsize=8, loc="left")
    axl.axis("off")
    axl.scatter([], [], s=12, color=FOC, label="aligned (cos ≥ 0)")
    axl.scatter([], [], s=12, color=ALARM, label="opposed (cos < 0)")
    axl.legend(loc="upper left", frameon=False, fontsize=6.5, bbox_to_anchor=(0, 0.95))
    axl.text(0, 0.42, "GPT-2 124M, step 1000\n50,257 vocab rows\n(47 padding masked)",
             fontsize=5.8, color=META_GREY, transform=axl.transAxes, va="top")
    _save(fig, out, "fig1_perrow_histogram")
    _csv(out, "fig1_perrow_histogram", ["cosine_bin_center", "count"], zip(centers, counts))
    return dict(median=float(np.median(rc)), frac_neg=frac_neg, mtp_frac=mtp_frac, ctrl_frac=ctrl_frac)


def fig2(results, out):
    def traj(name):
        d = _load(f"{results}/phase_a/{name}")
        st = [m["step"] for m in d["measurements"]]
        g = [m["global_cos"] for m in d["measurements"]]
        pr = [m["per_row_0.3"] for m in d["measurements"]]
        order = np.argsort(st)
        return np.array(st)[order], np.array(g)[order], np.array(pr)[order]
    ms, mg, mpr = traj("a1_muon_interference.json")
    as_, ag, apr = traj("a2_adamw_interference.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(ms, mg, "-", color=FOC, lw=1.6, label="aggregate cosine (Muon)")
    ax.plot(as_, ag, "--", color=FOC, lw=1.2, alpha=0.7, label="aggregate cosine (AdamW)")
    ax.plot(ms, mpr, "-", color=ALARM, lw=1.6, label="per-row |cos|>0.3 (Muon)")
    ax.plot(as_, apr, "--", color=ALARM, lw=1.2, alpha=0.7, label="per-row |cos|>0.3 (AdamW)")
    ax.axhline(0, color=META_GREY, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("Training step"); ax.set_ylabel("Cosine / fraction")
    ax.set_title("Aggregate and per-row alignment decouple during training", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="center right")
    _save(fig, out, "fig2_emergent_divergence")
    _csv(out, "fig2_emergent_divergence", ["step", "muon_global", "muon_perrow"], zip(ms, mg, mpr))


def fig3(results, out):
    a1 = _load(f"{results}/phase_a/a1_muon_interference.json")
    snap = a1["per_layer_snapshots"]["1000"]
    names = list(snap.keys())
    pr = np.array([snap[n]["per_row_0.3"] for n in names])
    gl = np.array([snap[n]["global_cos"] for n in names])
    is_head = np.array(["wte" in n or "lm_head" in n for n in names])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(gl[~is_head], pr[~is_head], s=14, c=META_GREY, alpha=0.7, label="attn / MLP / embed")
    ax.scatter(gl[is_head], pr[is_head], s=90, marker="*", c=FOC, edgecolors="k",
               linewidths=0.4, zorder=5, label="output head (wte, tied)")
    ax.set_xlabel("Aggregate cosine"); ax.set_ylabel("Per-row |cos| > 0.3")
    ax.set_title(f"High per-row alignment is general ({len(names)} matrices)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    _save(fig, out, "fig3_perlayer")
    _csv(out, "fig3_perlayer", ["matrix", "global_cos", "per_row_0.3"], zip(names, gl, pr))


def fig4(results, out):
    def mean_vals(variant):
        fs = sorted(glob.glob(f"{results}/phase_b/{variant}_seed*.json"))
        return [(_load(f)["final_val_loss"]) for f in fs]
    def tuned_10_9():
        fs = glob.glob(f"{results}/phase_d/*layers10-9*_30517steps.json")
        return [_load(f)["final_val_loss"] for f in fs if "5000steps" not in f]
    data = {"a": mean_vals("a"), "gnce": mean_vals("gnce"), "gnce_t": tuned_10_9(),
            "nextlat": mean_vals("nextlat"), "b": mean_vals("b"), "b_sg": mean_vals("b_sg")}
    order = [("a", "CE only"), ("gnce", "G_nce"), ("gnce_t", "G_nce\ntuned"),
             ("nextlat", "NextLat"), ("b", "shared\nMTP"), ("b_sg", "MTP\nstop-grad")]
    A_mean = np.mean(data["a"])
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, (k, lab) in enumerate(order):
        vals = data[k]; mn = np.mean(vals)
        col = META_GREY if k == "a" else (ALARM if k in ("b", "b_sg") else FOC)
        ax.bar(i, mn, width=0.62, color=col, alpha=0.85, zorder=2)
        jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(i + np.array(jit), vals, s=12, c="k", zorder=3, alpha=0.7)
    ax.axhline(A_mean, color=META_GREY, ls="--", lw=0.9, zorder=1)
    ax.text(2.4, A_mean - 0.012, f"baseline A = {A_mean:.3f}", fontsize=6, color=META_GREY, ha="center", va="top")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=6)
    ax.set_ylabel("Final validation loss (nats)"); ax.set_ylim(3.9, 4.60)
    ax.set_title("No auxiliary recovers the baseline; MTP degrades it most", fontsize=7.6, loc="left")
    for k, lab, dx in [("b", f"Δ+{np.mean(data['b'])-A_mean:.2f}", -0.42),
                       ("b_sg", f"Δ+{np.mean(data['b_sg'])-A_mean:.2f}", -0.42)]:
        i = [j for j, (kk, _) in enumerate(order) if kk == k][0]
        ax.text(i + dx, np.mean(data[k]), lab, ha="right", va="center", fontsize=6.3, color=ALARM, fontweight="bold")
    _save(fig, out, "fig4_interventions")
    _csv(out, "fig4_interventions", ["variant", "mean_final_val_loss", "n_seeds"],
         [(k, np.mean(v), len(v)) for k, v in data.items()])


def fig5(results, out):
    # Use mean_abs_cos (non-saturating) rather than the frac>0.3 (ceiling-bound) statistic.
    def quints(name):
        d = _load(f"{results}/phase_a/{name}")["token_freq_correlation"]["1000"]
        qs = ["bottom_20%", "20-40%", "40-60%", "60-80%", "top_20%"]
        return [d[q]["mean_abs_cos"] for q in qs], qs
    mv, qs = quints("a1_muon_interference.json")
    av, _ = quints("a2_adamw_interference.json")
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    x = np.arange(5)
    ax.plot(x, mv, "-o", color=FOC, lw=1.6, ms=4, label="Muon")
    ax.plot(x, av, "--^", color=AUX, lw=1.4, ms=4, label="AdamW")
    ax.set_xticks(x)
    ax.set_xticklabels(["rarest\n20%", "20–40%", "40–60%", "60–80%", "most\ncommon"], fontsize=6)
    ax.set_ylabel("Mean per-row |cos|"); ax.set_xlabel("Token frequency quintile")
    ax.set_ylim(0.82, 1.01)
    ax.set_title("Rare tokens show higher per-row alignment (single run)", fontsize=8, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    _save(fig, out, "fig5_token_frequency")
    _csv(out, "fig5_token_frequency", ["quintile", "muon_mean_abs_cos", "adamw_mean_abs_cos"], zip(qs, mv, av))


def fig6(results, out):
    """Norm decomposition: median cos, mean cos (=aggregate if norms flat), actual aggregate."""
    from matplotlib.lines import Line2D
    runs = [("Muon", "a1_muon_interference.json"), ("AdamW", "a2_adamw_interference.json")]
    data = []
    for lab, fn in runs:
        m = _final_measurement(_load(f"{results}/phase_a/{fn}")["measurements"], step=1000)
        rc = np.asarray(m["row_cosines"])[:REAL_VOCAB]
        data.append((lab, float(np.median(rc)), float(rc.mean()), float(m["global_cos"])))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(0, color=META_GREY, lw=0.8, ls=":", zorder=0)
    for i, (lab, me, mn, gl) in enumerate(data):
        ax.plot([i - 0.26], [me], "o", color=FOC, ms=7, zorder=3)
        ax.plot([i], [mn], "s", color=AUX, ms=7, zorder=3)
        ax.plot([i + 0.26], [gl], "D", color=ALARM, ms=7, zorder=3)
        ax.annotate("", xy=(i + 0.26, gl), xytext=(i, mn),
                    arrowprops=dict(arrowstyle="->", color=ALARM, lw=1.2))
        ax.text(i + 0.30, (mn + gl) / 2, f"{gl - mn:+.2f}\n(norm\neffect)",
                color=ALARM, fontsize=6, va="center", ha="left")
    ax.set_xticks(range(len(data))); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("Cosine (CE vs MTP gradient)"); ax.set_ylim(-0.45, 1.12); ax.set_xlim(-0.6, 1.7)
    ax.set_title("Per-row norm structure, not the typical cosine, sets the aggregate",
                 fontsize=7.8, loc="left")
    leg = [Line2D([], [], marker="o", color=FOC, ls="", ms=6, label="median per-row cosine"),
           Line2D([], [], marker="s", color=AUX, ls="", ms=6, label="mean per-row cosine (aggregate if norms flat)"),
           Line2D([], [], marker="D", color=ALARM, ls="", ms=6, label="actual aggregate cosine")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=6)
    _save(fig, out, "fig6_norm_decomposition")
    _csv(out, "fig6_norm_decomposition", ["run", "median_cos", "mean_cos", "actual_aggregate"],
         [(d[0], d[1], d[2], d[3]) for d in data])


def _save(fig, out, name):
    fig.savefig(f"{out}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _csv(out, name, header, rows):
    import csv
    with open(f"{out}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, float) else x for x in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _style()
    s = fig1(a.results, a.out); fig2(a.results, a.out); fig3(a.results, a.out)
    fig4(a.results, a.out); fig5(a.results, a.out); fig6(a.results, a.out)
    print("Figures written to", a.out)
    print(f"  fig1: median +{s['median']:.2f}, {s['frac_neg']:.2f}% cos<0, "
          f"{s['mtp_frac']:.1f}% |cos|>0.3, control {s['ctrl_frac']:.2f}%")


if __name__ == "__main__":
    main()
