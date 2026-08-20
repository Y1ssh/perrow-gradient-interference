#!/usr/bin/env python3
"""Regenerate Figure 1, the mechanism schematic (paper Fig. 1).

This is a *conceptual* figure (hand-composed cartoon), but its proportions are
chosen to match the measured norm/support result (measurement/run_norm_support.py,
n=3 per optimizer): per-row directions are aligned for ~99.7% of rows, yet a tiny
HIGH-NORM opposed minority (~0.3% of rows by count) carries the majority of the
gradient-magnitude mass (opposed_norm_fraction ~0.60-0.69). CE and MTP load the
SAME rows (norm_profile_cos ~0.98 => support OVERLAPS, it does not diverge), so
the aggregate cosine is dragged negative by norm-weighted opposition on a few
dominant rows, not by disjoint support. Run:

    python figures/make_schematic.py --out figures

Produces fig0_schematic.{png,pdf}. The PDF is what the venue main.tex files include
(via paper/sections/intro.tex).
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 1.05]})

    rows = 8
    # One representative high-norm OPPOSED row (row index 2 from top); the rest are
    # low-norm aligned rows. This mirrors the measured 0.3%-by-count / 60%-of-mass split.
    opposed_row = 2

    # ---- Panel A: per-row directions; a high-norm minority is opposed ----
    axA = axes[0]
    for r in range(rows):
        y = rows - 1 - r
        if r == opposed_row:
            # high-norm, OPPOSED: long arrows pointing opposite ways
            axA.annotate("", xy=(0.35 + 0.85, y), xytext=(0.35, y),
                         arrowprops=dict(arrowstyle="->", color=CE_C, lw=2.2))
            axA.annotate("", xy=(0.35 - 0.30, y), xytext=(0.35, y),
                         arrowprops=dict(arrowstyle="->", color=MTP_C, lw=2.2))
            axA.text(0.35 + 0.90, y, "high-norm,\nopposed", fontsize=5.6,
                     color=ALARM, va="center", ha="left")
        else:
            # low-norm, aligned: short arrows same direction
            ang = np.deg2rad(32 + np.random.RandomState(r).uniform(-5, 5))
            dx, dy = 0.42 * np.cos(ang), 0.14 * np.sin(ang)
            axA.annotate("", xy=(0.35 + dx, y + dy), xytext=(0.35, y),
                         arrowprops=dict(arrowstyle="->", color=CE_C, lw=1.1))
            axA.annotate("", xy=(0.35 + dx * 0.9, y + dy * 0.9), xytext=(0.35, y),
                         arrowprops=dict(arrowstyle="->", color=MTP_C, lw=1.1, alpha=0.8))
    axA.set_xlim(-0.5, 1.9); axA.set_ylim(-0.9, rows - 0.2)
    axA.set_title("Per row: direction\n99.7% aligned; a few opposed", fontsize=7.0)
    axA.axis("off")
    axA.text(0.35, -0.7, "one row per vocab token", fontsize=6, color=MUTE, ha="left")

    # ---- Panel B: per-row MAGNITUDE on the SAME rows (support overlaps) ----
    axB = axes[1]
    rng = np.random.RandomState(3)
    # CE and MTP load the SAME rows (overlap); magnitude concentrated on the opposed row.
    base = rng.uniform(0.04, 0.14, rows)
    ce_mag = base.copy(); mtp_mag = base * rng.uniform(0.85, 1.1, rows)
    ce_mag[opposed_row] = 0.95; mtp_mag[opposed_row] = 0.88   # the dominant high-norm row
    yv = np.arange(rows)[::-1]
    bar_colors_ce = [ALARM if i == opposed_row else CE_C for i in range(rows)]
    bar_colors_mt = [ALARM if i == opposed_row else MTP_C for i in range(rows)]
    axB.barh(yv + 0.16, ce_mag, height=0.30, color=bar_colors_ce, label="CE $\\|g\\|$")
    axB.barh(yv - 0.16, mtp_mag, height=0.30, color=bar_colors_mt, label="MTP $\\|g\\|$",
             alpha=0.85)
    axB.set_xlim(0, 1.05); axB.set_ylim(-0.9, rows - 0.2)
    axB.set_title("Per row: magnitude, same rows\n(support overlaps, cos $\\approx0.98$)",
                  fontsize=7.0)
    axB.axis("off")
    axB.legend(loc="lower right", frameon=False, fontsize=6)
    axB.text(0.02, -0.7, "one high-norm row carries the mass", fontsize=5.8,
             color=ALARM, ha="left")

    # ---- Panel C: what each metric reports ----
    axC = axes[2]
    axC.axis("off"); axC.set_xlim(0, 1); axC.set_ylim(0, 1)
    axC.text(0.5, 0.95, "Two metrics, two readings", fontsize=7.4, ha="center", weight="bold")
    axC.text(0.02, 0.74, "per-row cosine (typical row)", fontsize=6.4, color=FOC)
    axC.add_patch(mpl.patches.FancyBboxPatch((0.02, 0.62), 0.96, 0.08,
                  boxstyle="round,pad=0.005", fc="#eef4fb", ec=FOC, lw=0.8))
    axC.plot([0.02 + 0.96 * 0.97], [0.66], "o", color=FOC, ms=8)
    axC.text(0.98, 0.52, "$\\approx +1$  \"aligned\"", fontsize=7, color=FOC, ha="right")
    axC.text(0.02, 0.35, "aggregate cosine (mass-weighted)", fontsize=6.4, color=ALARM)
    axC.add_patch(mpl.patches.FancyBboxPatch((0.02, 0.23), 0.96, 0.08,
                  boxstyle="round,pad=0.005", fc="#fdecec", ec=ALARM, lw=0.8))
    # aggregate sits NEGATIVE now (box spans -1..+1 mapped to 0..0.96; negative => left)
    axC.plot([0.02 + 0.96 * 0.35], [0.27], "D", color=ALARM, ms=8)
    axC.text(0.98, 0.13, "$<0$  misread as \"conflict\"", fontsize=7, color=ALARM, ha="right")
    axC.text(0.5, 0.03, "the opposed minority dominates the sum", fontsize=5.8,
             color=MUTE, ha="center")

    fig.suptitle("Aligned per row, but a high-norm opposed minority drags the aggregate "
                 "negative; the cosine misreads it", fontsize=7.8, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig0_schematic.png", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig0_schematic.pdf", bbox_inches="tight")
    print(f"wrote {out_dir}/fig0_schematic.png and .pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    make(ap.parse_args().out)
