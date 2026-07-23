#!/usr/bin/env python3
"""Regenerate Figure 1, the support-divergence schematic (paper Fig. 1).

This is a *conceptual* figure (hand-composed cartoon, not derived from the
committed result JSONs), so it lives in its own script rather than in
make_figures.py. Run:

    python figures/make_schematic.py --out figures

Produces fig0_schematic.{png,pdf}. The PDF is what paper/main.tex includes.
"""
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

CE_C = "#2b6cb0"   # CE gradient (blue)
MTP_C = "#C77d0a"  # MTP gradient (amber)
FOC = "#2b6cb0"
ALARM = "#C44"
MUTE = "#9aa0a6"


def make(out_dir: str) -> None:
    mpl.rcParams.update({
        "font.size": 8, "axes.linewidth": 0.8,
        "font.family": "DejaVu Sans", "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    # ---- Panel A: per-row directions (aligned) ----
    axA = axes[0]
    rows = 8
    for r in range(rows):
        y = rows - 1 - r
        ang = np.deg2rad(35 + np.random.RandomState(r).uniform(-6, 6))
        dx, dy = 0.7 * np.cos(ang), 0.16 * np.sin(ang)
        axA.annotate("", xy=(0.35 + dx, y + dy), xytext=(0.35, y),
                     arrowprops=dict(arrowstyle="->", color=CE_C, lw=1.4))
        axA.annotate("", xy=(0.35 + dx * 0.9, y + dy * 0.9), xytext=(0.35, y),
                     arrowprops=dict(arrowstyle="->", color=MTP_C, lw=1.4, alpha=0.8))
    axA.set_xlim(0, 1.5); axA.set_ylim(-0.6, rows - 0.2)
    axA.set_title("Per row: aligned\n(median cos $\\approx+1$)", fontsize=7.2)
    axA.axis("off")
    axA.text(0.35, -0.5, "one row per vocab token", fontsize=6, color=MUTE, ha="left")

    # ---- Panel B: per-row MAGNITUDE on disjoint rows ----
    axB = axes[1]
    rng = np.random.RandomState(1)
    ce_mag = np.zeros(rows); mtp_mag = np.zeros(rows)
    for r in [0, 1, 2, 3]: ce_mag[r] = rng.uniform(0.5, 0.95)
    for r in [4, 5, 6, 7]: mtp_mag[r] = rng.uniform(0.5, 0.95)
    yv = np.arange(rows)[::-1]
    axB.barh(yv + 0.16, ce_mag, height=0.32, color=CE_C, label="CE $\\|g\\|$")
    axB.barh(yv - 0.16, mtp_mag, height=0.32, color=MTP_C, label="MTP $\\|g\\|$")
    axB.set_xlim(0, 1.05); axB.set_ylim(-0.6, rows - 0.2)
    axB.set_title("Per row: magnitude on\n disjoint rows (support)", fontsize=7.2)
    axB.axis("off")
    axB.legend(loc="lower right", frameon=False, fontsize=6)

    # ---- Panel C: what each metric reports ----
    axC = axes[2]
    axC.axis("off"); axC.set_xlim(0, 1); axC.set_ylim(0, 1)
    axC.text(0.5, 0.93, "Two metrics, two readings", fontsize=7.4, ha="center", weight="bold")
    axC.text(0.02, 0.72, "per-row cosine", fontsize=6.8, color=FOC)
    axC.add_patch(mpl.patches.FancyBboxPatch((0.02, 0.6), 0.96, 0.08,
                  boxstyle="round,pad=0.005", fc="#eef4fb", ec=FOC, lw=0.8))
    axC.plot([0.02 + 0.96 * 0.97], [0.64], "o", color=FOC, ms=8)
    axC.text(0.98, 0.50, "$\\approx +1$  \"aligned\"", fontsize=7, color=FOC, ha="right")
    axC.text(0.02, 0.33, "aggregate cosine", fontsize=6.8, color=ALARM)
    axC.add_patch(mpl.patches.FancyBboxPatch((0.02, 0.21), 0.96, 0.08,
                  boxstyle="round,pad=0.005", fc="#fdecec", ec=ALARM, lw=0.8))
    axC.plot([0.02 + 0.96 * 0.5], [0.25], "D", color=ALARM, ms=8)
    axC.text(0.98, 0.11, "$\\approx 0$  \"conflict\" (artifact)", fontsize=7, color=ALARM, ha="right")

    fig.suptitle("Support divergence: identical directions, disjoint magnitude, "
                 "and the aggregate misreads it", fontsize=8.0, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig0_schematic.png", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig0_schematic.pdf", bbox_inches="tight")
    print(f"wrote {out_dir}/fig0_schematic.png and .pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    make(ap.parse_args().out)
