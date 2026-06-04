"""
Nagaki-style CONVERSION contour (2026-05-12 refactor).

Note: the 6.7 min GC peak is the SUBSTRATE (4-bromofluorobenzene), not the
product. So the metric is now substrate conversion = 100 - residual%.
This measures Li-Br exchange RATE (formation only), not decay.

X-ticks: 10^-1.5, 10^-1.0, 10^-0.5, 10^0, 10^0.5  (fixed decades)
Data points placed at actual log10(tR).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

CSV = Path(__file__).parent / "experiment_yield_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_conversion_contour.png"

T_ORDER = [20, 0, -25, -50, -78]
L_ORDER = [3, 10, 25, 50, 100]
X_TICKS_LOG = [-1.5, -1.0, -0.5, 0.0, 0.5]

OUTLIERS = set()
MISSING  = [(20, 50), (20, 100), (-25, 100)]

METRIC = "conversion_pct_ECN"      # what to plot
METRIC_LABEL = "Substrate conversion %  (ECN-est., pending RF calibration)"


def main():
    df = pd.read_csv(CSV).dropna(subset=[METRIC]).copy()
    df["is_outlier"] = df.apply(lambda r: (r.T_C, r.L_cm) in OUTLIERS, axis=1)

    valid_df   = df[~df["is_outlier"]].copy()
    outlier_df = df[df["is_outlier"]].copy()
    print(f"Valid: {len(valid_df)}    Outliers: {len(outlier_df)}    Missing: {len(MISSING)}")
    print(valid_df[["T_C", "L_cm", "tR_s", METRIC]]
          .sort_values(["T_C", "L_cm"], ascending=[False, True])
          .to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = plt.get_cmap("RdYlGn")
    # Conversion is 0–100; pick scale that highlights the dynamic range
    vmin, vmax = 30, 100
    norm = Normalize(vmin=vmin, vmax=vmax)
    tube = np.pi * 0.025 ** 2

    # contour interpolated from valid only
    log_tR = np.log10(valid_df["tR_s"].to_numpy())
    T = valid_df["T_C"].to_numpy()
    Y = valid_df[METRIC].to_numpy()
    xi = np.linspace(-1.6, 0.6, 350)
    yi = np.linspace(-85, 25, 350)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((log_tR, T), Y, (XI, YI), method="linear")

    levels = np.arange(vmin, vmax + 1, 5)
    cf = ax.contourf(XI, YI, ZI, levels=levels, cmap="RdYlGn",
                     extend="both", alpha=0.9)
    cs = ax.contour(XI, YI, ZI, levels=levels[::2], colors="black",
                    linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fmt="%d", fontsize=8)

    # valid data points
    for _, r in valid_df.iterrows():
        x = np.log10(r.tR_s); y = r.T_C
        c = cmap(norm(r[METRIC]))
        ax.scatter(x, y, s=200, marker="o", facecolor=c,
                   edgecolor="black", linewidth=1.5, zorder=6)
        ax.annotate(f"{r[METRIC]:.1f}", xy=(x, y),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="gray", alpha=0.85))

    # outlier (drawn but with X overlay + smaller label)
    for _, r in outlier_df.iterrows():
        x = np.log10(r.tR_s); y = r.T_C
        c = cmap(norm(r[METRIC]))
        ax.scatter(x, y, s=200, marker="X", facecolor=c,
                   edgecolor="red", linewidth=2.0, zorder=6)
        ax.annotate(f"{r[METRIC]:.1f}\n(outlier)",
                    xy=(x, y), xytext=(8, -18),
                    textcoords="offset points", fontsize=8,
                    color="darkred", style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffe8e8",
                              ec="red", alpha=0.85))

    # missing slots
    for T_, L_ in MISSING:
        tR = tube * L_ / (7.5 / 60)
        ax.scatter(np.log10(tR), T_, s=220, marker="o",
                   facecolor="none", edgecolor="gray",
                   linewidth=1.6, linestyle="--", zorder=4)
        ax.annotate("missing", xy=(np.log10(tR), T_),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="gray", style="italic")

    # fixed log10 ticks
    ax.set_xticks(X_TICKS_LOG)
    ax.set_xticklabels([rf"$10^{{{t:g}}}$" for t in X_TICKS_LOG], fontsize=11)
    ax.set_xlabel("Residence time tR / s   (log scale)", fontsize=12)
    ax.set_ylabel("Temperature / °C", fontsize=12)
    ax.set_yticks(T_ORDER)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(-1.6, 0.6)
    ax.set_ylim(-85, 28)

    # secondary L marks at top
    for L_ in L_ORDER:
        tR = tube * L_ / (7.5 / 60)
        ax.axvline(np.log10(tR), ymin=0.97, ymax=1.0,
                   color="black", linewidth=0.6, alpha=0.4)
        ax.text(np.log10(tR), 26.5, f"L={L_}", ha="center",
                fontsize=8, color="gray", style="italic")

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(METRIC_LABEL, fontsize=11)

    elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markeredgecolor="black", markersize=12,
               label=f"valid  (n={len(valid_df)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markersize=12,
               label=f"missing  (n={len(MISSING)})"),
    ]
    if len(outlier_df):
        elems.insert(1,
            Line2D([0], [0], marker="X", color="w", markerfacecolor="lightgray",
                   markeredgecolor="red", markersize=12,
                   label=f"outlier  (n={len(outlier_df)})"))
    ax.legend(handles=elems, fontsize=10, framealpha=0.95,
              loc="upper left", bbox_to_anchor=(1.18, 1.0),
              borderaxespad=0.)

    ax.set_title(
        "4-Bromofluorobenzene + n-BuLi → 4-FC₆H₄Li · substrate conversion contour\n"
        "(GC 6.77 min peak = unreacted substrate; conversion = 100 − residual%)",
        fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
