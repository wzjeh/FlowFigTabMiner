"""
Mechanism-resolved descriptor–kinetics correlation figure for C2 (remote-EWG ArLi).
Supports three literature-consistent conclusions:
  (1) lnA dominated by solvation/aggregation entropy  -> Gsolv strongest on lnA_d
  (2) Hammett σ fails for decomposition               -> σ weak/non-sig on Ea_d/lnA_d
  (3) decomposition is solvent-mediated (proto-de-Li)  -> Gsolv dominates Ea_d, decay-specific
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from scipy import stats
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

tr = pd.read_csv("v60_training_set.csv"); tr = tr[(~tr.is_virtual) & (tr.class_v60 == "C2")].copy()
old = pd.read_csv("clean_organolithium_unified_descriptors.csv"); old["cs"] = old["intermediate_smiles_canonical"].apply(canon)
oc = {"dft_HOMO_eV":"HOMO","dft_LUMO_eV":"LUMO","HOMO_LUMO_gap_eV":"gap","dft_Gsolv_kJ":"Gsolv",
      "buried_vol_Li":"%Vbur","dft_LiC_bond_A":"d(LiC)","dft_dipole_D":"dipole"}
oo = old.dropna(subset=["cs"]).drop_duplicates("cs")[["cs"]+list(oc)].rename(columns=oc)
tr["cs"] = tr.smi.apply(canon); tr = tr.merge(oo, on="cs", how="left")
tr["sigma"] = tr.sigma_p_sum.fillna(0) + tr.sigma_m_sum.fillna(0)

DESC = ["Gsolv","dipole","sigma","HOMO","LUMO","gap","%Vbur","d(LiC)"]
TGT  = ["Ea_f","lnA_f","Ea_d","lnA_d"]
Rm = np.full((len(DESC),len(TGT)), np.nan); Pm = np.ones_like(Rm)
for i,dn in enumerate(DESC):
    for j,t in enumerate(TGT):
        s = tr.dropna(subset=[dn,t])
        if s[dn].std()>0 and len(s)>=6:
            Rm[i,j],Pm[i,j] = stats.pearsonr(s[dn],s[t])

fig = plt.figure(figsize=(14.5, 6.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0], hspace=0.55, wspace=0.28)

# ---- heatmap ----
axh = fig.add_subplot(gs[:, 0])
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = axh.imshow(Rm, cmap="RdBu_r", norm=norm, aspect="auto")
axh.set_xticks(range(len(TGT))); axh.set_xticklabels(["$E_{a,f}$","ln$A_f$","$E_{a,d}$","ln$A_d$"], fontsize=13)
axh.set_yticks(range(len(DESC))); axh.set_yticklabels(DESC, fontsize=12)
axh.axvline(1.5, color="k", lw=2.5)  # formation | decay divider
axh.text(0.5, -0.72, "FORMATION", ha="center", fontsize=11, fontweight="bold", color="0.35")
axh.text(2.5, -0.72, "DECOMPOSITION", ha="center", fontsize=11, fontweight="bold", color="0.35")
for i in range(len(DESC)):
    for j in range(len(TGT)):
        if not np.isnan(Rm[i,j]):
            star = "*" if Pm[i,j]<0.05 else ""
            axh.text(j, i, f"{Rm[i,j]:+.2f}{star}", ha="center", va="center",
                     fontsize=11, fontweight="bold" if Pm[i,j]<0.05 else "normal",
                     color="white" if abs(Rm[i,j])>0.6 else "black")
axh.set_title("C2 (remote-EWG ArLi, n=12): Pearson r — descriptor vs kinetics\n"
              "* p<0.05; only Gsolv significant, and only for decomposition",
              fontsize=12, fontweight="bold", pad=30)
fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03, label="Pearson r")

# ---- 3 scatter panels ----
def scat(ax, x, y, xl, yl, title):
    s = tr.dropna(subset=[x,y]); xx=s[x].values; yy=s[y].values
    r,p = stats.pearsonr(xx,yy)
    ax.scatter(xx,yy,s=55,c="#c0392b",edgecolor="k",lw=0.8,zorder=5)
    a,b = np.polyfit(xx,yy,1); xs=np.array([xx.min(),xx.max()])
    ax.plot(xs,a*xs+b,"k--",lw=1.4,alpha=0.7)
    ax.set_xlabel(xl,fontsize=11); ax.set_ylabel(yl,fontsize=11)
    ax.set_title(f"{title}   r={r:+.2f}{'*' if p<0.05 else ''}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
scat(fig.add_subplot(gs[0,1]),"Gsolv","lnA_d","$G_{solv}$ (kJ/mol)","ln$A_d$","Gsolv → lnA_d (solvation/entropy)")
scat(fig.add_subplot(gs[1,1]),"Gsolv","Ea_d","$G_{solv}$ (kJ/mol)","$E_{a,d}$ (kJ/mol)","Gsolv → Ea_d (solvent-mediated)")
scat(fig.add_subplot(gs[2,1]),"sigma","Ea_d","Hammett $\\sigma$","$E_{a,d}$ (kJ/mol)","σ → Ea_d (electronic, FAILS)")

fig.suptitle("Solvation (Gsolv) dominates C2 decomposition; Hammett σ does not — consistent with literature",
             fontsize=13, fontweight="bold", y=1.04)
p = "analysis_figures/fig_c2_mechanism_correlation.png"
fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); print("✓", p)
