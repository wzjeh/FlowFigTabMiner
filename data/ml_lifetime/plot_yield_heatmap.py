"""
Nagaki-style yield contour plot.
  X: residence time tR (log scale)
  Y: temperature (°C)
  Scatter circles, colored by yield%, with yield value labeled next to each point
  Filled contour background interpolated from valid points only

Excludes from interpolation:
  • day-1 anomalous batch (T=0°C entire row)
  • monotonicity-violating rebound points (-25°C L=50, -50°C L=100)
  • 3 missing points
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

CSV = Path(__file__).parent / "experiment_yield_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_yield_contour.png"

FLAGS = {
    (0,    3): "anomaly", (0,   10): "anomaly", (0,   25): "anomaly",
    (0,   50): "anomaly", (0,  100): "anomaly",
    (-25, 50):  "rebound",
    (-50, 100): "rebound",
}


def main():
    df = pd.read_csv(CSV).dropna(subset=["yield_pct_ECN"]).copy()
    df["flag"] = df.apply(lambda r: FLAGS.get((r.T_C, r.L_cm), ""), axis=1)

    valid = df[df["flag"] == ""]
    flagged = df[df["flag"] != ""]

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- contour interpolation from valid points only ---
    log_tR = np.log10(valid["tR_s"].to_numpy())
    T = valid["T_C"].to_numpy()
    Y = valid["yield_pct_ECN"].to_numpy()

    # grid for filled contour
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

    # --- scatter all points ---
    norm = Normalize(vmin=0, vmax=70)
    cmap = plt.get_cmap("RdYlGn")

    for _, r in df.iterrows():
        x = np.log10(r.tR_s)
        y = r.T_C
        c = cmap(norm(r.yield_pct_ECN))
        if r.flag == "anomaly":
            ax.scatter(x, y, s=180, marker="X", facecolor=c,
                       edgecolor="red", linewidth=2, zorder=5)
        elif r.flag == "rebound":
            ax.scatter(x, y, s=180, marker="s", facecolor=c,
                       edgecolor="orange", linewidth=2, zorder=5)
        else:
            ax.scatter(x, y, s=180, marker="o", facecolor=c,
                       edgecolor="black", linewidth=1.2, zorder=5)
        # text label next to point
        label = f"{r.yield_pct_ECN:.1f}"
        if r.flag:
            label += f"\n({r.flag})"
        ax.annotate(label, xy=(x, y),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="gray", alpha=0.7))

    # missing points (T=20 L=50/100, T=-25 L=100) — dashed open markers
    tube_vol_per_cm = np.pi * (0.025 ** 2)
    for T_, L_ in [(20, 50), (20, 100), (-25, 100)]:
        tR = tube_vol_per_cm * L_ / (7.5 / 60)
        ax.scatter(np.log10(tR), T_, s=200, marker="o",
                   facecolor="none", edgecolor="gray",
                   linewidth=1.5, linestyle="--", zorder=4)
        ax.annotate("missing", xy=(np.log10(tR), T_),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="gray", style="italic")

    # axis cosmetics
    L_ticks = [3, 10, 25, 50, 100]
    tR_ticks = [tube_vol_per_cm * L / (7.5 / 60) for L in L_ticks]
    ax.set_xticks(np.log10(tR_ticks))
    ax.set_xticklabels([f"{tR:.3f}\n(L={L}cm)" for L, tR in zip(L_ticks, tR_ticks)])
    ax.set_xlabel("Residence time tR / s  (log scale)", fontsize=12)
    ax.set_ylabel("Temperature / °C", fontsize=12)
    ax.set_yticks([20, 0, -25, -50, -78])
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlim(np.log10(0.035), np.log10(1.9))
    ax.set_ylim(-85, 28)

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Yield % (ECN-estimated, pending RF calibration)", fontsize=11)

    # legend for marker types
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markeredgecolor="black", markersize=12, label="valid"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="lightgray",
               markeredgecolor="red", markersize=12, label="day-1 anomaly"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="lightgray",
               markeredgecolor="orange", markersize=12, label="rebound"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markersize=12, label="missing"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=10,
              framealpha=0.9, title="data quality")

    ax.set_title("4-Bromofluorobenzene + n-BuLi → 4-fluorophenyllithium\n"
                 "Yield contour vs (tR, T) — φ500µm flow reactor, 22/25 measured",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
