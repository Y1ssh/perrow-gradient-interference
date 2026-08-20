#!/usr/bin/env python3
"""
Regenerate all data figures in the paper from the committed result JSONs.

    python figures/make_figures.py --results results --out figures --analysis analysis

No GPU or training needed; reads only `results/` and `analysis/`. Produces
fig1..fig6 plus fig_token_lorenz and fig_mixture_sweep as .png + .pdf + .csv in
--out. Numbers match stats.py.
fig1/fig5/fig_token_lorenz read results/norm_support/ (per-row norms + cosines);
fig_mixture_sweep reads results/phase_e/ + analysis/kl_scan_results.json.

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


def _norm_support(results, opt="muon", seed=42):
    """Load a norm_support run and return (row_cos, ce_norm, mtp_norm, active) on real rows."""
    d = _load(f"{results}/norm_support/norm_support_{opt}_seed{seed}.json")
    rc = np.asarray(d["row_cosines"])[:REAL_VOCAB]
    ce = np.asarray(d["ce_row_norms"])[:REAL_VOCAB]
    mt = np.asarray(d["mtp_row_norms"])[:REAL_VOCAB]
    ratio = np.where(ce > 0, mt / ce, 0.75)
    active = np.abs(ratio - 0.75) > 0.01   # by-construction target-presence proxy (0.75 identity)
    return rc, ce, mt, active


def fig1(results, out):
    """Active-row per-row cosine histogram (main) with full-vocab spike inset."""
    rc, ce, mt, active = _norm_support(results, "muon", 42)
    a = rc[active]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bins = np.linspace(-1, 1, 61)
    _, _, patches = ax.hist(a, bins=bins, color=FOC, alpha=0.85, edgecolor="none")
    for i, p in enumerate(patches):
        if bins[i] < 0:
            p.set_facecolor(ALARM)
    med = float(np.median(a)); ymax = ax.get_ylim()[1]
    ax.axvline(med, color="k", lw=1.2, ls="--")
    ax.annotate(f"median {med:.2f}", xy=(med, ymax * 0.42), xytext=(med + 0.08, ymax * 0.52),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlabel("Per-row cosine (active rows)"); ax.set_ylabel("Number of rows")
    ax.set_title("On rows that carry supervision, CE/MTP gradients are aligned but not identical",
                 fontsize=8, loc="left")
    ax.text(0.015, 0.42, f"{100*(a<0).mean():.1f}%\nopposed", transform=ax.transAxes,
            color=ALARM, fontsize=7.5, va="top")
    axi = ax.inset_axes([0.10, 0.52, 0.34, 0.40])
    axi.hist(rc, bins=bins, color="#999999", alpha=0.9, edgecolor="none")
    axi.axvline(1.0, color="k", lw=0.8, ls="--")
    axi.set_title("full vocabulary (median +1)", fontsize=6.2)
    axi.set_xticks([-1, 0, 1]); axi.tick_params(labelsize=5.5); axi.set_yticks([])
    axi.text(0.04, 0.82, f"{100*active.mean():.0f}% active\n(rest parallel)",
             transform=axi.transAxes, fontsize=5.5, va="top")
    _save(fig, out, "fig1_perrow_histogram")
    counts, edges = np.histogram(a, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    _csv(out, "fig1_perrow_histogram", ["active_cosine_bin_center", "count"], zip(centers, counts))
    return dict(active_median=med, active_frac_neg=float(100*(a<0).mean()),
                active_fraction=float(100*active.mean()))


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
    ax.set_xscale("log"); ax.set_xlabel("Training step")
    ax.set_ylabel("Aggregate cosine or per-row fraction")
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
    order = [("a", "CE only"), ("gnce", "$G_{\\mathrm{nce}}$"), ("gnce_t", "$G_{\\mathrm{nce}}$\ntuned"),
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
    ax.set_title("No auxiliary demonstrably recovers the baseline", fontsize=7.6, loc="left")
    for k, lab, dx in [("b", f"Δ+{np.mean(data['b'])-A_mean:.2f}", -0.42),
                       ("b_sg", f"Δ+{np.mean(data['b_sg'])-A_mean:.2f}", -0.42)]:
        i = [j for j, (kk, _) in enumerate(order) if kk == k][0]
        ax.text(i + dx, np.mean(data[k]), lab, ha="right", va="center", fontsize=6.3, color=ALARM, fontweight="bold")
    _save(fig, out, "fig4_interventions")
    _csv(out, "fig4_interventions", ["variant", "mean_final_val_loss", "n_seeds"],
         [(k, np.mean(v), len(v)) for k, v in data.items()])


def _freq_quints(results, opt, seed=42):
    """Per token-id quintile (GPT-2 id ~ frequency): full-vocab & active-only mean|cos|, active %."""
    rc, ce, mt, active = _norm_support(results, opt, seed)
    qs = np.array_split(np.arange(REAL_VOCAB), 5)
    full = [float(np.mean(np.abs(rc[q]))) for q in qs]
    actv = [float(np.mean(np.abs(rc[q][active[q]]))) if active[q].any() else np.nan for q in qs]
    frac = [float(100 * active[q].mean()) for q in qs]
    return full, actv, frac


def fig5(results, out):
    """Rare-token effect is a construction artifact: full-vocab apparent effect vs active-row flat."""
    mf, ma, mfr = _freq_quints(results, "muon")
    af, aa, _ = _freq_quints(results, "adamw")
    x = np.arange(5); labels = ["most\ncommon", "", "middle", "", "rarest"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax1.plot(x, mf, "-o", color="#999999", lw=1.6, ms=4, label="Muon")
    ax1.plot(x, af, "--^", color="#999999", lw=1.4, ms=4, alpha=0.6, label="AdamW")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel("Mean per-row |cos|"); ax1.set_ylim(0.80, 1.01)
    ax1.set_title("Full vocabulary: apparent rare-token effect", fontsize=7.5, loc="left")
    ax1.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax1.text(0.5, 0.05, "artifact", transform=ax1.transAxes, fontsize=7, style="italic", color=ALARM)
    ax2b = ax2.twinx()
    ax2b.bar(x, mfr, width=0.5, color="#e0e0e0", zorder=0)
    ax2b.set_ylabel("% active (bars)", fontsize=6.5, color="#888888")
    ax2b.tick_params(labelsize=5.5, colors="#888888"); ax2b.set_ylim(0, 30)
    ax2.plot(x, ma, "-o", color=FOC, lw=1.6, ms=4, label="Muon", zorder=3)
    ax2.plot(x, aa, "--^", color=FOC, lw=1.4, ms=4, alpha=0.6, label="AdamW", zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("Active-row mean |cos|"); ax2.set_ylim(0.0, 1.01)
    ax2.set_title("Active rows: effect vanishes", fontsize=7.5, loc="left")
    ax2.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax2.set_zorder(ax2b.get_zorder() + 1); ax2.patch.set_visible(False)
    fig.text(0.5, 0.005, "Token frequency quintile", ha="center", fontsize=7.5)
    fig.suptitle("The rare-token alignment effect is a construction artifact: rare tokens are ~98% inactive",
                 fontsize=8, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, out, "fig5_token_frequency")
    _csv(out, "fig5_token_frequency",
         ["quintile", "muon_full", "muon_active", "muon_active_pct", "adamw_full", "adamw_active"],
         zip(["common", "q2", "middle", "q4", "rarest"], mf, ma, mfr, af, aa))


def fig_token_lorenz(results, out):
    """Lorenz curve of gradient mass + decoded top-opposed tokens (needs tiktoken)."""
    rc, ce, mt, active = _norm_support(results, "muon", 42)
    mass = ce * mt; opp = rc < 0
    order = np.argsort(mass)[::-1]; cummass = np.cumsum(mass[order]) / mass.sum()
    n_23 = int(np.searchsorted(cummass, 2/3) + 1)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        decode = lambda t: enc.decode([int(t)])
    except Exception:
        decode = lambda t: f"id{t}"
    opp_idx = np.where(opp)[0]; obm = opp_idx[np.argsort(mass[opp_idx])[::-1]]
    top = [(decode(t), float(rc[t]), float(mass[t] / mass.sum() * 100)) for t in obm[:12]]
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)
    axL = fig.add_subplot(gs[0, 0])
    axL.plot(np.arange(1, len(cummass) + 1), cummass * 100, color=FOC, lw=1.8)
    axL.plot([0, REAL_VOCAB], [0, 100], ls=":", color="#aaaaaa", lw=0.9)
    axL.axhline(66.7, color=ALARM, ls="--", lw=0.8)
    axL.annotate(f"{n_23} rows\n= 2/3 of mass", xy=(n_23, 66.7), xytext=(n_23 + 180, 52),
                 fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8, color=ALARM))
    axL.set_xlim(0, 600); axL.set_ylim(0, 101)
    axL.set_xlabel("Number of rows (ranked by gradient mass)")
    axL.set_ylabel("Cumulative % of gradient mass")
    axL.set_title("Gradient mass is extremely concentrated", fontsize=8, loc="left")
    axT = fig.add_subplot(gs[0, 1]); axT.axis("off")
    axT.set_title("Highest-mass opposed rows are frequent function words", fontsize=8, loc="left")
    cell = [[repr(tok), f"{c:+.2f}", f"{m:.1f}"] for tok, c, m in top]
    tbl = axT.table(cellText=cell, colLabels=["token", "cos", "% mass"], loc="center",
                    cellLoc="left", colWidths=[0.5, 0.25, 0.25])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.25)
    for (r, cc), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#dddddd")
        if r == 0:
            cellobj.set_text_props(weight="bold"); cellobj.set_facecolor("#f0f0f0")
        elif float(cell[r - 1][2]) >= 3:
            cellobj.set_facecolor("#fbe9e9")
    fig.suptitle(f"{int(opp.sum())} opposed rows ({100*opp.mean():.2f}% of vocab) carry "
                 f"{100*mass[opp].sum()/mass.sum():.0f}% of the gradient mass", fontsize=8.5, y=1.01)
    _save(fig, out, "fig_token_lorenz")
    _csv(out, "lorenz_curve", ["n_rows", "cumulative_mass_frac"],
         ((n, f"{cummass[n-1]:.5f}") for n in list(range(1, 101)) + list(range(100, 601, 10))))


def fig7(results, out):
    """Norm support: norm-profile cosine + opposed-norm fraction, n=3 seeds/optimizer.

    Regenerates the mechanism figure from the committed results/norm_support/ JSONs
    so a fresh clone can rebuild it. Plots SIGNED quantities throughout: the
    norm-profile cosine of the two per-row norm vectors, and the fraction of
    norm-product mass carried by rows with cos < 0.
    """
    opts = ["muon", "adamw"]
    labels = ["Muon", "AdamW"]
    stat = {}
    for opt in opts:
        npc, onf, gc = [], [], []
        for seed in (42, 123, 456):
            d = _load(f"{results}/norm_support/norm_support_{opt}_seed{seed}.json")
            npc.append(float(d["norm_profile_cos"]))
            onf.append(float(d["opposed_norm_fraction"]))
            gc.append(float(d["global_cos"]))
        stat[opt] = dict(npc=npc, onf=onf, gc=gc)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    x = np.arange(len(opts))
    npc_m = [float(np.mean(stat[o]["npc"])) for o in opts]
    npc_s = [float(np.std(stat[o]["npc"], ddof=1)) for o in opts]
    gc_m = [float(np.mean(stat[o]["gc"])) for o in opts]
    gc_s = [float(np.std(stat[o]["gc"], ddof=1)) for o in opts]
    axL.errorbar(x - 0.08, npc_m, yerr=npc_s, fmt="o", ms=8, capsize=4,
                 color="#2b6cb0", label="norm-profile cosine")
    axL.errorbar(x + 0.08, gc_m, yerr=gc_s, fmt="s", ms=8, capsize=4,
                 color="#c0392b", label="aggregate cosine")
    axL.axhline(0.0, lw=0.8, color="0.6", zorder=0)
    for xi, (a, b) in enumerate(zip(npc_m, gc_m)):
        axL.annotate("", xy=(xi + 0.08, b + 0.06), xytext=(xi - 0.08, a - 0.06),
                     arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_xlim(-0.45, 1.45)
    axL.set_ylim(-0.55, 1.12)
    axL.set_ylabel("cosine")
    axL.set_title("Same rows are loaded, yet the aggregate is negative", fontsize=10)
    axL.legend(frameon=False, fontsize=8, loc="center right")

    onf_m = [float(np.mean(stat[o]["onf"])) for o in opts]
    onf_s = [float(np.std(stat[o]["onf"], ddof=1)) for o in opts]
    axR.errorbar(x, onf_m, yerr=onf_s, fmt="o", ms=8, capsize=4, color="#2b6cb0")
    axR.axhline(0.5, ls="--", lw=1.0, color="0.5")
    axR.text(0.98, 0.52, "half the mass", transform=axR.get_yaxis_transform(),
             fontsize=8, color="0.45", va="bottom", ha="right")
    for xi, (v, e) in enumerate(zip(onf_m, onf_s)):
        axR.annotate(f"{v:.2f}", xy=(xi, v + e), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=9)
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_xlim(-0.45, 1.45)
    axR.set_ylim(0.0, 1.0)
    axR.set_ylabel("opposed-norm fraction (cos < 0)")
    axR.set_title("The opposed minority carries most of the mass", fontsize=10)
    for ax in (axL, axR):
        ax.margins(0.06)
    fig.suptitle("The aggregate is a norm-weighted cancellation, not disjoint support",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "fig7_norm_support")
    _csv(out, "fig7_norm_support",
         ["optimizer", "norm_profile_cos_mean", "norm_profile_cos_sd",
          "opposed_norm_fraction_mean", "opposed_norm_fraction_sd",
          "aggregate_cos_mean", "aggregate_cos_sd"],
         [[labels[i], npc_m[i], npc_s[i], onf_m[i], onf_s[i], gc_m[i], gc_s[i]]
          for i in range(len(opts))])


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


def fig_mixture_sweep(results, out, analysis="analysis"):
    """Measured next-token gap vs auxiliary weight s, with the zero-parameter KL band.

    Reads results/phase_e/sweep_scale*.json (observed gaps) and
    analysis/kl_scan_results.json (predicted band + coverage extrapolation).
    Both are committed, so this regenerates from a fresh clone.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sw = {}
    for f in sorted(glob.glob(f"{results}/phase_e/sweep_scale*_seed42_*steps.json")):
        d = _load(f)
        sw[float(d["mtp_scale"])] = float(d["final_val_loss"])
    if 0.0 not in sw:
        print("  fig_mixture_sweep: no s=0 anchor, skipped"); return
    ce0 = sw[0.0]
    scales = sorted(sw)
    gaps = [sw[s] - ce0 for s in scales]

    kl = _load(f"{analysis}/kl_scan_results.json")
    rows = kl["kl_estimates"]
    kmax = max(e["topk"] for e in rows)
    sig = float(kl.get("gap_noise_sd", 0.0255))
    # band per s = [min over all variants, max over all variants] at the largest k
    band_lo, band_hi = [], []
    for s in scales:
        key = f"{s:g}" if f"{s:g}" in rows[0] else str(s)
        vals = [e[key] for e in rows if e["topk"] == kmax and key in e]
        band_lo.append(min(vals) if vals else 0.0)
        band_hi.append(max(vals) if vals else 0.0)
    # coverage extrapolation of the full-mixture (p3eqp2) variant at s=1
    h0 = sorted([e for e in rows if e["offset3"] == "p3eqp2" and e["val_half"] == 0],
                key=lambda e: e["topk"])
    extrap = None
    if len(h0) >= 2:
        u = [1.0 - e["coverage_mass"] for e in h0]
        v = [e["1.0"] for e in h0]
        slope = (v[-2] - v[-1]) / (u[-2] - u[-1])
        extrap = v[-1] - slope * u[-1]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.fill_between(scales, band_lo, band_hi, color=AUX, alpha=0.30, lw=0, zorder=1)
    ax.plot(scales, band_hi, color=AUX, lw=1.0, zorder=2)
    ax.plot(scales, band_lo, color=AUX, lw=1.0, zorder=2)
    # s=0 is the anchor: its "gap" is the loss minus itself, identically zero, so it
    # carries no run-noise interval. Only the s>0 points are measurements against it.
    pos = [i for i, s in enumerate(scales) if s > 0]
    ax.errorbar([scales[i] for i in pos], [gaps[i] for i in pos], yerr=sig, fmt="o",
                color=FOC, ms=6, lw=0, elinewidth=1.1, capsize=2.5, zorder=4)
    ax.plot([0.0], [0.0], marker="o", mfc="white", mec=FOC, mew=1.2, ms=6, ls="", zorder=4)
    if extrap is not None:
        ax.plot([1.0], [extrap], marker="v", color=ALARM, ms=7, ls="", zorder=5)
        ax.annotate(f"coverage-extrapolated\ntop {extrap:.3f}", xy=(1.0, extrap),
                    xytext=(0.62, extrap - 0.075), fontsize=6, color=ALARM,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=ALARM, lw=0.8))
        ax.annotate(f"residual {gaps[-1] - extrap:+.3f}",
                    xy=(1.0, (gaps[-1] + extrap) / 2), xytext=(1.03, (gaps[-1] + extrap) / 2),
                    fontsize=6, color=FOC, ha="left", va="center")
    ax.set_xlabel("Auxiliary weight $s$")
    ax.set_ylabel("Next-token loss increase (nats)")
    ax.set_title("A zero-parameter KL prediction tracks the measured degradation",
                 fontsize=7.8, loc="left")
    ax.set_xticks(scales)
    ax.margins(0.06)
    ax.set_xlim(-0.06, 1.30)
    leg = [Line2D([], [], marker="o", color=FOC, ls="", ms=6,
                  label=f"measured gap ($\\pm${sig:.4f} run noise)"),
           Line2D([], [], marker="o", mfc="white", mec=FOC, mew=1.2, ls="", ms=6,
                  label="anchor ($s{=}0$, gap $\\equiv 0$)"),
           Patch(facecolor=AUX, alpha=0.30, label=f"predicted KL band, $k{{=}}{kmax}$"),
           Line2D([], [], marker="v", color=ALARM, ls="", ms=6,
                  label="full-coverage extrapolation")]
    ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=6)
    _save(fig, out, "fig_mixture_sweep")
    _csv(out, "fig_mixture_sweep",
         ["s", "final_val", "observed_gap", "band_lo", "band_hi"],
         [(scales[i], sw[scales[i]], gaps[i], band_lo[i], band_hi[i])
          for i in range(len(scales))])
    return {"gaps": gaps, "band_hi": band_hi, "extrap": extrap}


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
    ap.add_argument("--analysis", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _style()
    s = fig1(a.results, a.out); fig2(a.results, a.out); fig3(a.results, a.out)
    fig4(a.results, a.out); fig5(a.results, a.out); fig6(a.results, a.out)
    fig7(a.results, a.out)
    fig_token_lorenz(a.results, a.out)
    fig_mixture_sweep(a.results, a.out, a.analysis)
    print("Figures written to", a.out)
    print(f"  fig1: active median {s['active_median']:.2f}, {s['active_frac_neg']:.2f}% opposed, "
          f"{s['active_fraction']:.1f}% of vocab active")


if __name__ == "__main__":
    main()
