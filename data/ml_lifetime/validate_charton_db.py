"""
Task 2 — VALIDATE the fixed Charton anchor against the database (NOT refit).

Idea (dataset identifiability filtering): the DB per-substrate formation fits (global_arrhenius.csv
Ea_f/lnA_f) are reliable ONLY for substrates whose source experiment was chemistry-controlled
(Da≪1). High-Da (mixing) substrates have k_obs ↛ k_chem, so their fitted k_f is a mixing artifact
and must NOT be used to learn intrinsic chemistry.

We therefore (a) classify each DB substrate by Da = τ_eff·k_chem(σ,T_ref) using the fixed Charton
anchor + a representative flow τ_eff, and (b) compare the fixed Charton k_chem to the DB-fitted k_f.
Expectation: Da-clean (chemical-regime) points lie ON the Charton line; mixing points fall BELOW
(observed capped). This validates the anchor without re-fitting it.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import formation_model as fm

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
BASE = Path(__file__).parent

T_REF = -30.0                  # °C, representative flash-chemistry temperature for the comparison
TAU_FLOW = 0.098e-3            # s, representative Nagaki 250 µm flow platform τ_eff
DA_CLEAN = 0.3                 # Da threshold for "chemistry-controlled"

g = pd.read_csv(BASE / "global_arrhenius.csv")
v = pd.read_csv(BASE / "v60_classified_substrates.csv")[["smi", "class_v60", "sigma_p_sum", "sigma_m_sum"]]
df = g.merge(v, on="smi", how="left")
df["sigma"] = df["sigma_p_sum"].fillna(0) + df["sigma_m_sum"].fillna(0)

R = fm.R_GAS
Tk = T_REF + 273.15
df["kf_obs"]   = np.exp(df["lnA_f"] - df["Ea_f"] / (R * Tk))      # DB-fitted formation rate @T_ref
df["kchem"]    = fm.k_chem(df["sigma"].values, T_REF)             # fixed Charton anchor
df["Da"]       = TAU_FLOW * df["kchem"]
df["regime"]   = np.where(df["Da"] < DA_CLEAN, "chemical",
                  np.where(df["Da"] < 3.0, "transition", "mixing"))

clean = df[df["regime"] == "chemical"]
print(f"{'='*72}\nFixed Charton anchor vs DB-fitted k_f  (T_ref={T_REF}°C, τ_eff={TAU_FLOW*1e3:.3f} ms)")
print(f"{'='*72}")
print(f"  n total = {len(df)} | chemical(Da<{DA_CLEAN}) = {len(clean)} | "
      f"transition = {(df.regime=='transition').sum()} | mixing = {(df.regime=='mixing').sum()}")
# agreement on the Da-clean (chemistry-controlled) subset — in log space
lo = clean[(clean.kf_obs > 0) & (clean.kchem > 0)]
if len(lo) >= 3:
    r = np.corrcoef(np.log10(lo.kf_obs), np.log10(lo.kchem))[0, 1]
    med_ratio = np.median(lo.kf_obs / lo.kchem)
    print(f"  Da-clean parity: Pearson r(log) = {r:+.2f}, median k_obs/k_chem = {med_ratio:.2f}")
print(f"\n  {'substrate':32s} {'σ':>5s} {'Da':>9s} {'k_obs':>10s} {'k_chem':>10s}  regime")
for _, x in df.sort_values("Da").iterrows():
    print(f"  {x['intermediate'][:32]:32s} {x['sigma']:5.2f} {x['Da']:9.2f} "
          f"{x['kf_obs']:10.1f} {x['kchem']:10.1f}  {x['regime']}")

# ---- parity figure ----
fig, ax = plt.subplots(figsize=(6.2, 5.6))
cmap = {"chemical": "#1e8449", "transition": "#b9770e", "mixing": "#c0392b"}
for reg, c in cmap.items():
    s = df[(df.regime == reg) & (df.kf_obs > 0)]
    ax.scatter(s.kchem, s.kf_obs, s=55, c=c, edgecolor="k", lw=0.7, label=f"{reg} (n={len(s)})", zorder=5)
lims = [1e0, 1e6]
ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="k_obs = k_chem")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lims); ax.set_ylim(*lims)
ax.set_xlabel("fixed Charton anchor  $k_{chem}$ (/s)", fontsize=11)
ax.set_ylabel("DB-fitted  $k_{f,obs}$ (/s)", fontsize=11)
ax.set_title(f"Da-clean points validate the Charton anchor; mixing points fall below\n"
             f"(T_ref={T_REF}°C, τ_eff={TAU_FLOW*1e3:.2f} ms)", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.2, which="both")
out = BASE / "analysis_figures" / "fig_charton_db_validation.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
df.to_csv(BASE / "formation_anchor_validation.csv", index=False)
print(f"\n✓ {out}\n✓ formation_anchor_validation.csv")
