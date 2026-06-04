"""
Nagaki-style CONVERSION contour for PhCHO-trap experiment (2026-05-12 batch).

Style 完全参考 plot_yield_contour_clean.py:
  X-ticks: 10^-1.5, 10^-1.0, 10^-0.5, 10^0, 10^0.5  (fixed decades)
  数据点按真实 log10(tR) 放置
  legend 在图外右上
  顶部小 tick 标 L=3/10/25/50/100

数据:
  21 vials (WY-260512-1..21) mapped to:
    1-5  : T=+20°C, L=3,10,25,50,100
    6-10 : T=  0°C, L=3,10,25,50,100
    11-15: T=-25°C, L=3,10,25,50,100
    16-20: T=-50°C, L=3,10,25,50,100
    21   : T=-65°C, L=3   (excluded per Zhao 2026-05-13)

Missing/invalid:
    vial #15 (-25°C, L=100):  substrate peak absent → marked missing
    vial #21 (-65°C, L=3):    Zhao said skip for now
    -65°C × {10,25,50,100}:   still being analyzed
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

CSV = Path(__file__).parent / "experiment_phcho_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_phcho_conversion_contour.png"

T_ORDER = [20, 0, -25, -50, -65]
L_ORDER = [3, 10, 25, 50, 100]
X_TICKS_LOG = [-1.5, -1.0, -0.5, 0.0, 0.5]

# vial → (T, L) map (same as Zhao confirmed)
T_BATCH = [20, 0, -25, -50]
VIAL_MAP = {}
i = 1
for T in T_BATCH:
    for L in L_ORDER:
        VIAL_MAP[i] = (T, L)
        i += 1
VIAL_MAP[21] = (-65, 3)

# Vials to exclude from plot entirely
EXCLUDE_VIALS = {21}      # -65°C/L=3: Zhao said skip
# Slots to mark as "missing" (not measured / pending)
MISSING_SLOTS = [(-65, L) for L in [3, 10, 25, 50, 100]]


def main():
    df = pd.read_csv(CSV)

    # apply Zhao-confirmed mapping and exclude
    df["T_C"]  = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[0])
    df["L_cm"] = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[1])
    df = df[~df["vial"].isin(EXCLUDE_VIALS)].copy()
    df = df.dropna(subset=["conversion_pct", "T_C", "L_cm"]).copy()

    print(f"Valid points: {len(df)}")
    print(df[["vial", "T_C", "L_cm", "tR_s_guess", "conversion_pct"]]
          .sort_values(["T_C", "L_cm"], ascending=[False, True])
          .to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = plt.get_cmap("RdYlGn")
    vmin, vmax = 30, 100      # match plot_yield_contour_clean.py style
    norm = Normalize(vmin=vmin, vmax=vmax)
    tube = np.pi * 0.025 ** 2

    log_tR = np.log10(df["tR_s_guess"].to_numpy())
    T = df["T_C"].to_numpy()
    Y = df["conversion_pct"].to_numpy()
    xi = np.linspace(-1.6, 0.6, 350)
    yi = np.linspace(-72, 25, 350)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((log_tR, T), Y, (XI, YI), method="linear")

    levels = np.arange(vmin, vmax + 1, 5)
    cf = ax.contourf(XI, YI, ZI, levels=levels, cmap="RdYlGn",
                     extend="both", alpha=0.9)
    cs = ax.contour(XI, YI, ZI, levels=levels[::2], colors="black",
                    linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fmt="%d", fontsize=8)

    # valid data points
    for _, r in df.iterrows():
        x = np.log10(r["tR_s_guess"]); y = r["T_C"]
        c = cmap(norm(r["conversion_pct"]))
        ax.scatter(x, y, s=200, marker="o", facecolor=c,
                   edgecolor="black", linewidth=1.5, zorder=6)
        ax.annotate(f"{r['conversion_pct']:.1f}", xy=(x, y),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="gray", alpha=0.85))

    # missing slots
    for T_, L_ in MISSING_SLOTS:
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
    ax.set_ylim(-72, 28)

    # secondary L marks at top
    for L_ in L_ORDER:
        tR = tube * L_ / (7.5 / 60)
        ax.axvline(np.log10(tR), ymin=0.97, ymax=1.0,
                   color="black", linewidth=0.6, alpha=0.4)
        ax.text(np.log10(tR), 26.5, f"L={L_}", ha="center",
                fontsize=8, color="gray", style="italic")

    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Substrate conversion %  (ECN-est., pending RF calibration)",
                   fontsize=11)

    elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markeredgecolor="black", markersize=12,
               label=f"valid  (n={len(df)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markersize=12,
               label=f"missing  (n={len(MISSING_SLOTS)})"),
    ]
    ax.legend(handles=elems, fontsize=10, framealpha=0.95,
              loc="upper left", bbox_to_anchor=(1.18, 1.0),
              borderaxespad=0.)

    ax.set_title(
        "4-Bromofluorobenzene + n-BuLi → 4-FC₆H₄Li · substrate conversion (PhCHO trap exp)\n"
        "(GC 6.75 min peak = unreacted substrate; conversion = 100 − residual%)",
        fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
