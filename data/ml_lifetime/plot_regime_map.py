"""
Damköhler regime map for C2 (remote-EWG) aryllithium FORMATION (Br/Li exchange).

x = k_chem (intrinsic chemical exchange rate, from lit Hammett ρ_f=5.07, Ea_f=32, lnA_f=21)
y = τ_mix  (characteristic micromixing time; single Nagaki platform value, semi-quant band)
diagonals: Da = τ_mix·k_chem = const (slope −1 on log-log)  → chemical / transition / mixing

Conceptual centerpiece: strong-EWG substrates (CN, NO2) have such fast intrinsic exchange
that they cross into the mixing-dominated regime, where the observed rate is set by τ_mix
(an apparatus parameter) rather than molecular structure — explaining why formation Ea_f
cannot be predicted from molecular descriptors, while decomposition (chemically controlled)
can.  Reactor geometry extracted by Gemini (semi-quantitative).
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
BASE = Path(__file__).parent
df = pd.read_csv(BASE / "da_regime_c2.csv")

# τ_mix (s) from the Nagaki platform: V_mixer/Q = d/u ≈ 0.098 ms.  Reverse-check vs Da column:
TAU_MIX = float(np.median(df.Da / df.kf_chem_s))         # = 9.8e-5 s
TAU_LO, TAU_HI = TAU_MIX / 2.5, TAU_MIX * 2.5            # semi-quant platform uncertainty (~factor 2.5)

# group the 6 distinct (σ, k) points by EWG family
FAM = {0.37: ("m-CO$_2$R ester ×4", "#2e86c1", "o"),
       0.45: ("p-CO$_2$R ester ×4", "#5dade2", "o"),
       0.56: ("m-CN",               "#e67e22", "s"),
       0.66: ("p-CN",               "#d35400", "s"),
       0.71: ("m-NO$_2$",           "#c0392b", "^"),
       0.78: ("p-NO$_2$",           "#922b21", "^")}
pts = df.drop_duplicates("sigma").sort_values("sigma")

fig, ax = plt.subplots(figsize=(7.4, 5.6))
kx = np.logspace(2, 6.2, 400)

YLO, YHI = 1e-5, 1e-2
# ---- diagonal Da = const regions (slope −1): chemical / transition / mixing ----
ax.fill_between(kx, YLO, np.clip(0.1/kx, YLO, YHI), color="#27ae60", alpha=0.10)         # Da<0.1
ax.fill_between(kx, np.clip(0.1/kx, YLO, YHI), np.clip(10/kx, YLO, YHI), color="#f1c40f", alpha=0.10)  # 0.1–10
ax.fill_between(kx, np.clip(10/kx, YLO, YHI), YHI, color="#c0392b", alpha=0.10)           # Da>10

# ---- diagonal Da = const guide lines + labels placed mid-chart along each line ----
for Da, lab, klab in [(0.1, "Da=0.1", 1e3), (1.0, "Da=1", 1e4), (10.0, "Da=10", 1e5)]:
    ax.plot(kx, Da / kx, "--", color="0.45", lw=1.1, zorder=1)
    ax.text(klab, (Da / klab) * 1.35, lab, color="0.35", fontsize=9, rotation=-33,
            ha="center", va="bottom")

# ---- τ_mix platform band ----
ax.axhspan(TAU_LO, TAU_HI, color="0.6", alpha=0.18, zorder=0)
ax.axhline(TAU_MIX, color="0.35", lw=1.4, zorder=2)
ax.text(kx.min()*1.15, TAU_MIX*1.04,
        rf"$\tau_{{mix}}\approx{TAU_MIX*1e3:.2f}$ ms (Nagaki T-mixer platform)",
        fontsize=9, color="0.25", va="bottom")

# ---- substrate points ----
seen = set()
for _, r in pts.iterrows():
    lab, col, mk = FAM[round(r.sigma, 2)]
    ax.scatter(r.kf_chem_s, TAU_MIX, s=130, c=col, marker=mk,
               edgecolor="k", lw=1.0, zorder=6,
               label=lab if lab not in seen else None)
    seen.add(lab)

# regime headers placed in their diagonal regions (chemical: bottom-left, mixing: top-right)
ax.text(0.12, 0.05, "CHEMICAL\n(Da << 1)", transform=ax.transAxes, ha="center",
        fontsize=10, fontweight="bold", color="#1e8449", va="bottom")
ax.text(0.50, 0.55, "TRANSITION", transform=ax.transAxes, ha="center", rotation=-33,
        fontsize=10, fontweight="bold", color="#b9770e", va="center")
ax.text(0.86, 0.90, "MIXING-DOMINATED\n(Da >> 1)", transform=ax.transAxes, ha="center",
        fontsize=10, fontweight="bold", color="#922b21", va="top")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(kx.min(), kx.max()); ax.set_ylim(1e-5, 1e-2)
ax.set_xlabel(r"intrinsic exchange rate  $k_{chem}$  (s$^{-1}$)  —  faster EWG →", fontsize=11.5)
ax.set_ylabel(r"micromixing time  $\tau_{mix}$  (s)", fontsize=11.5)
ax.set_title("Formation Damköhler regime map — strong EWG crosses into mixing control",
             fontsize=11.5, fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=2,
          title="C2 remote-EWG ArLi", title_fontsize=8.5)
ax.grid(True, which="both", alpha=0.18)

out = BASE / "analysis_figures" / "fig_regime_map.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
print("✓", out, f"| τ_mix={TAU_MIX*1e3:.3f} ms, Da=0.1 @k={0.1/TAU_MIX:.0f}, Da=10 @k={10/TAU_MIX:.0f}")
