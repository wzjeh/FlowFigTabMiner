"""
Day-2-only yield contour (drop the day-1 batch: T=0°C entire row + T=-25°C entire row).
Compare side-by-side with the all-points version to show that the
day-1 batch was the source of physical anomalies.
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
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_yield_contour_day2.png"

# day-1 batch: T=0°C (all L) and T=-25°C (all L) → flag for exclusion
DAY1_TEMPS = {0, -25}
REBOUND = {(-50, 100)}    # keep -25, 50 dropped automatically since -25°C all out


def main():
    df = pd.read_csv(CSV).dropna(subset=["yield_pct_ECN"]).copy()
    df["day1"] = df["T_C"].isin(DAY1_TEMPS)
    df["rebnd"] = df.apply(lambda r: (r.T_C, r.L_cm) in REBOUND, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (label, use_df) in enumerate([
        ("All 22 pts (incl. day-1)",        df),
        ("Day-2 only: 13 pts (drop 0/-25°C)", df[~df["day1"]].copy()),
    ]):
        ax = axes[ax_idx]
        valid = use_df[~use_df["rebnd"]]
        flagged = use_df[use_df["rebnd"]]
        # day-1 still shown in left panel as red-X
        d1_pts = use_df[use_df["day1"]] if ax_idx == 0 else use_df.iloc[0:0]

        # interpolate from valid (non-rebound, non-day1) points
        interp_src = valid[~valid["day1"]] if ax_idx == 0 else valid
        log_tR = np.log10(interp_src["tR_s"].to_numpy())
        T = interp_src["T_C"].to_numpy()
        Y = interp_src["yield_pct_ECN"].to_numpy()
        xi = np.linspace(np.log10(0.03), np.log10(2.0), 200)
        yi = np.linspace(-85, 25, 200)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((log_tR, T), Y, (XI, YI), method="linear")

        levels = np.arange(0, 71, 5)
        cf = ax.contourf(XI, YI, ZI, levels=levels, cmap="RdYlGn",
                         extend="both", alpha=0.85)
        cs = ax.contour(XI, YI, ZI, levels=levels[::2], colors="black",
                        linewidths=0.5, alpha=0.4)
        ax.clabel(cs, inline=True, fmt="%d", fontsize=8)

        norm = Normalize(vmin=0, vmax=70)
        cmap = plt.get_cmap("RdYlGn")

        for _, r in use_df.iterrows():
            x = np.log10(r.tR_s); y = r.T_C
            c = cmap(norm(r.yield_pct_ECN))
            if r.day1 and ax_idx == 0:
                ax.scatter(x, y, s=180, marker="X", facecolor=c,
                           edgecolor="red", linewidth=2, zorder=5)
            elif r.rebnd:
                ax.scatter(x, y, s=180, marker="s", facecolor=c,
                           edgecolor="orange", linewidth=2, zorder=5)
            else:
                ax.scatter(x, y, s=180, marker="o", facecolor=c,
                           edgecolor="black", linewidth=1.2, zorder=5)
            label_txt = f"{r.yield_pct_ECN:.1f}"
            ax.annotate(label_txt, xy=(x, y),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="gray", alpha=0.7))

        # missing points
        tube = np.pi * 0.025**2
        miss_pts = [(20, 50), (20, 100), (-25, 100)]
        if ax_idx == 1:
            # in day-2 panel, -25°C is dropped entirely so no need to mark
            miss_pts = [(20, 50), (20, 100)]
        for T_, L_ in miss_pts:
            tR = tube * L_ / (7.5/60)
            ax.scatter(np.log10(tR), T_, s=200, marker="o",
                       facecolor="none", edgecolor="gray",
                       linewidth=1.5, linestyle="--", zorder=4)
            ax.annotate("missing", xy=(np.log10(tR), T_),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=8, color="gray", style="italic")

        L_ticks = [3, 10, 25, 50, 100]
        tR_ticks = [tube*L/(7.5/60) for L in L_ticks]
        ax.set_xticks(np.log10(tR_ticks))
        ax.set_xticklabels([f"{tR:.3f}\n(L={L}cm)" for L, tR in zip(L_ticks, tR_ticks)])
        ax.set_xlabel("Residence time tR / s  (log scale)", fontsize=12)
        ax.set_ylabel("Temperature / °C", fontsize=12)
        ax.set_yticks([20, 0, -25, -50, -78])
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_xlim(np.log10(0.035), np.log10(1.9))
        ax.set_ylim(-85, 28)
        ax.set_title(label, fontsize=13, fontweight="bold")

        cbar = plt.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label("Yield % (ECN-est.)", fontsize=10)

        if ax_idx == 0:
            elems = [
                Line2D([0],[0], marker="o", color="w", markerfacecolor="lightgray",
                       markeredgecolor="black", markersize=10, label="valid"),
                Line2D([0],[0], marker="X", color="w", markerfacecolor="lightgray",
                       markeredgecolor="red", markersize=10, label="day-1 anomaly"),
                Line2D([0],[0], marker="s", color="w", markerfacecolor="lightgray",
                       markeredgecolor="orange", markersize=10, label="rebound"),
                Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
                       markeredgecolor="gray", markersize=10, label="missing"),
            ]
            ax.legend(handles=elems, loc="lower left", fontsize=9, framealpha=0.9)

    plt.suptitle("4-Bromofluorobenzene + n-BuLi flow validation\n"
                 "Effect of dropping day-1 (2026-05-08) batch on contour",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
