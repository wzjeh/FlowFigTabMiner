"""
Builder for `layered_kinetics_story.ipynb`.
Holds every notebook cell (markdown + code) as text. When run, it (1) execs the code cells
in order in a shared namespace to VERIFY they run and regenerate all figures, then (2) writes
the .ipynb JSON. Zero extra deps (no nbformat/nbconvert needed).
"""
import json, sys, traceback
from pathlib import Path

BASE = Path(__file__).parent
NB = BASE / "layered_kinetics_story.ipynb"

# ───────────────────────── cell sources ─────────────────────────
MD_TITLE = r"""# 有机锂生成动力学：分层模型 (intrinsic chemistry × observation layer)

这个 notebook 沿**新分层模型**的逻辑一步步分析、并产出讲故事所需的全部图。

**核心 4 点故事**
1. **本征化学** `k_chem` 由均相溶液 LFER（Charton eq.36）决定。
2. **流动观测** `k_obs = f(k_chem, τ_eff)` 决定可观测动力学。
3. **Da = τ_eff·k_chem 控制 identifiability**：Da≪1 化学可见、Da≫1 被混合主导。
4. **历史 flow 数据混淆了这两层** —— 这正是表观 `Ea_f` 看似可拟合却无物理意义的根源。

故事起点：我们先有一个 5 参数动力学模型；随后发现 formation 反常，才引入 Da 矫正。
"""

CODE_SETUP = r'''
# ===== Sec 0 · setup：统一风格 + 调色板 + 数据/模型引擎 =====
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")            # 出图存盘；下方各 cell 用 Image() 内联显示，与后端无关
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats
from IPython.display import Image
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import formation_model as fm                       # k_chem / observation_model / da / tau_eff
import draw_paper_figures as dpf                    # 复用结构图/实验验证图
from draw_paper_figures import VALIDATION, _load_exp, load_modeling_set

BASE = Path.cwd()
FIG = BASE / "analysis_figures"; FIG.mkdir(exist_ok=True)

# —— 统一科研风格 ——
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": True, "axes.spines.right": True,      # 全包围边框
    "axes.linewidth": 1.0, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
})
PALETTE = {"4BrFC6H4": "#1f6fb2", "5Br2FCN": "#c0392b",
           "ester": "#2e86c1", "CN": "#e67e22", "NO2": "#c0392b"}
T_CMAP = plt.get_cmap("coolwarm")
SIGMA = {"4BrFC6H4": 0.06, "5Br2FCN": 0.62}        # σ = σ_p(F) ; σ_m(CN)+σ_p(F)
TAU_REF = 15.9e-3                                  # s — 500 µm 验证装置 τ_eff@20°C；τ_eff(T) 由 THF 粘度标定(E_η=7.5)
def tau_obs(Tc): return fm.tau_eff_T(TAU_REF, Tc)  # 粘度修正的有效混合时标（低温更大）

def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

LAB = {"dft_Gsolv_kJ": "$G_{solv}$", "dft_HOMO_eV": "HOMO", "dft_LUMO_eV": "LUMO",
       "HOMO_LUMO_gap_eV": "gap", "dft_dipole_D": "dipole", "dft_charge_C_ipso": "q(C)",
       "dft_charge_Li": "q(Li)", "dft_LiC_bond_A": "d(Li-C)", "dft_LiC_BDE_kJ": "BDE(Li-C)",
       "dft_wiberg_LiC": "WBI(Li-C)", "buried_vol_Li": "%V$_{bur}$", "mol_volume": "V$_{mol}$",
       "sigma_hammett": "$\\sigma$", "Es_taft": "$E_s$"}

def desc_table():
    g = pd.read_csv(BASE / "global_arrhenius.csv"); g["cs"] = g["smi"].apply(canon)
    d = pd.read_csv(BASE / "clean_organolithium_unified_descriptors.csv")
    d["cs"] = d["intermediate_smiles_canonical"].apply(canon)
    cols = ["dft_Gsolv_kJ","dft_HOMO_eV","dft_LUMO_eV","HOMO_LUMO_gap_eV","dft_dipole_D",
            "dft_charge_C_ipso","dft_charge_Li","dft_LiC_bond_A","dft_LiC_BDE_kJ",
            "dft_wiberg_LiC","buried_vol_Li","mol_volume","sigma_hammett","Es_taft"]
    dd = d.dropna(subset=["cs"]).drop_duplicates("cs")[["cs"] + cols]
    return g.merge(dd, on="cs", how="left"), cols

print("setup OK · 建模集:", len(load_modeling_set()), "化合物 · τ_ref =", TAU_REF*1e3, "ms (τ_eff(-70°C)=%.0f ms)" % (tau_obs(-70)*1e3))
'''

MD_SEC1 = r"""## 1 · 故事开端：动力学模型

竞争一级反应：Ar–Br $\xrightarrow{k_f}$ Ar–Li $\xrightarrow{k_d}$ 分解。可观测产率

$$y(t_R,T)=y_{max}\,(1-e^{-k_f t_R})\,e^{-k_d t_R}$$

是一个 **5 参数全局 Arrhenius 模型** $\{E_{a,f},\ln A_f,E_{a,d},\ln A_d,y_{max}\}$。bell 形曲线随温度移动 —— 低温生成慢、高温分解快。
"""

CODE_SEC1 = r'''
# ===== Sec 1 · 动力学模型示意 + bell 形曲线 =====
def fig_kinetic_model():
    p = dict(Ea_f=35.78, lnA_f=22.52, Ea_d=28.83, lnA_d=9.88, y_max=85.1)  # p-CN-PhLi 全局拟合
    R = fm.R_GAS
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
    axa.set_title("(a)  Competing first-order kinetics", loc="left", fontweight="bold")
    axa.annotate("Ar-Br", xy=(0.06, 0.74), fontsize=14, ha="center")
    axa.annotate("", xy=(0.34, 0.74), xytext=(0.14, 0.74),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1f6fb2"))
    axa.text(0.24, 0.81, "$k_f$", color="#1f6fb2", fontsize=13, ha="center")
    axa.text(0.21, 0.65, "n-BuLi\n(formation)", fontsize=9, ha="center", color="0.4")
    axa.annotate("Ar-Li", xy=(0.44, 0.74), fontsize=14, ha="center")
    axa.annotate("", xy=(0.72, 0.74), xytext=(0.54, 0.74),
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#c0392b"))
    axa.text(0.63, 0.81, "$k_d$", color="#c0392b", fontsize=13, ha="center")
    axa.text(0.63, 0.65, "decomposition", fontsize=9, ha="center", color="0.4")
    axa.annotate("decomp.", xy=(0.85, 0.74), fontsize=11, ha="center", color="0.5")
    axa.annotate("", xy=(0.44, 0.50), xytext=(0.44, 0.66),
                 arrowprops=dict(arrowstyle="-|>", lw=1.8, color="0.35"))
    axa.text(0.50, 0.55, "quench → ArH / ArE", fontsize=9, ha="left", color="0.35")
    axa.text(0.5, 0.30, r"$y(t_R,T)=y_{max}\,(1-e^{-k_f t_R})\,e^{-k_d t_R}$", fontsize=14, ha="center",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f7", ec="0.7"))
    axa.text(0.5, 0.09, "5-parameter global Arrhenius model:\n"
             r"$\{E_{a,f},\ \ln A_f,\ E_{a,d},\ \ln A_d,\ y_{max}\}$", fontsize=10, ha="center", color="0.35")
    axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    axb = fig.add_subplot(gs[0, 1])
    axb.set_title("(b)  Yield-residence-time profiles", loc="left", fontweight="bold")
    tR = np.logspace(-3, 2.5, 400); Ts = [-78, -50, -25, 0, 25]
    for i, Tc in enumerate(Ts):
        Tk = Tc + 273.15
        kf = np.exp(p["lnA_f"] - p["Ea_f"]/(R*Tk)); kd = np.exp(p["lnA_d"] - p["Ea_d"]/(R*Tk))
        y = p["y_max"]*(1-np.exp(-kf*tR))*np.exp(-kd*tR); c = T_CMAP(i/(len(Ts)-1))
        axb.plot(tR, y, color=c, lw=2.2, label=f"{Tc} °C")
        tmax = np.log((kf+kd)/kd)/kf
        axb.scatter([tmax], [p["y_max"]*(1-np.exp(-kf*tmax))*np.exp(-kd*tmax)], color=c, s=30,
                    zorder=6, edgecolor="k", lw=0.5)
    axb.set_xscale("log"); axb.set_xlim(1e-3, 3e2); axb.set_ylim(0, 100)
    axb.set_xlabel("residence time $t_R$ (s)"); axb.set_ylabel("yield (%)")
    axb.text(0.02, 0.93, "formation ↑", transform=axb.transAxes, color="#1f6fb2", fontsize=9)
    axb.text(0.70, 0.93, "↓ decomposition", transform=axb.transAxes, color="#c0392b", fontsize=9)
    axb.legend(title="temperature", loc="center right", framealpha=0.9)
    fig.suptitle("A kinetic model for flow ArLi generation — the starting point",
                 fontsize=12.5, fontweight="bold", y=1.02)
    out = FIG / "fig_kinetic_model.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return out

Image(str(fig_kinetic_model()))
'''

MD_SEC2 = r"""## 2 · 因果正确的描述符：生成 ← 前体，分解 ← 中间体

**因果原则**：生成（Br/Li 交换）发生在 **ArBr 前体**上、此时 ArLi 尚不存在，所以 $E_{a,f}/\ln A_f$ 必须用**前体 ArBr 的描述符**预测；分解发生在 **ArLi 中间体**上，所以 $E_{a,d}/\ln A_d$ 才用**中间体 ArLi 的描述符**。**不能用中间体描述符预测它自己的生成。**

下图（C2, n=12）左块=ArBr 前体描述符×生成、右块=ArLi 中间体描述符×分解：
- **分解 ← ArLi**：$G_{solv}$ 主导 $E_{a,d}/\ln A_d$、Hammett σ 失效 → solvation-mediated，**可预测**。
- **生成 ← ArBr**：即便用因果正确的前体描述符，也**没有任何描述符显著**（C-Br BDE/LUMO/电荷都不相关）→ 生成不由分子化学决定，而是受**传递/混合**主导（呼应 Sec 3 的 Damköhler 分析）。

（对照：把所有机理混在一起、不分层时整体偏弱 —— 见下方第二张全机理热图。）
"""

CODE_SEC2A = r'''
# ===== Sec 2 · 因果正确：生成←ArBr前体描述符 / 分解←ArLi中间体描述符 =====
def fig_c2_correlation():
    tr = pd.read_csv(BASE/"v60_training_set.csv"); tr = tr[(~tr.is_virtual)&(tr.class_v60=="C2")].copy()
    tr["cs"] = tr.smi.apply(canon)
    old = pd.read_csv(BASE/"clean_organolithium_unified_descriptors.csv")
    old["cs"] = old["intermediate_smiles_canonical"].apply(canon)
    oc = {"dft_HOMO_eV":"HOMO","dft_LUMO_eV":"LUMO","HOMO_LUMO_gap_eV":"gap","dft_Gsolv_kJ":"Gsolv",
          "buried_vol_Li":"%Vbur","dft_LiC_bond_A":"d(LiC)","dft_dipole_D":"dipole"}
    oo = old.dropna(subset=["cs"]).drop_duplicates("cs")[["cs"]+list(oc)].rename(columns=oc)
    tr = tr.merge(oo, on="cs", how="left"); tr["sigma"] = tr.sigma_p_sum.fillna(0)+tr.sigma_m_sum.fillna(0)
    arbr = pd.read_csv(BASE/"arbr_c2_descriptors.csv")          # ArBr 前体描述符 + Ea_f/lnA_f
    # 生成 = ArBr 前体（Br/Li 交换时 ArLi 尚不存在）；分解 = ArLi 中间体
    F_DESC=["LUMO_br","gap_br","BDE_CBr","qCipso_br","qBr","dCBr","dip_br","sig"]
    F_LAB ={"LUMO_br":"LUMO(ArBr)","gap_br":"gap(ArBr)","BDE_CBr":"BDE(C-Br)","qCipso_br":"q(C$_{ipso}$)",
            "qBr":"q(Br)","dCBr":"d(C-Br)","dip_br":"dipole(ArBr)","sig":"$\\sigma$"}
    D_DESC=["Gsolv","HOMO","dipole","sigma","LUMO","gap","%Vbur","d(LiC)"]
    D_LAB ={"Gsolv":"$G_{solv}$","HOMO":"HOMO","dipole":"dipole","sigma":"$\\sigma$","LUMO":"LUMO",
            "gap":"gap","%Vbur":"%V$_{bur}$","d(LiC)":"d(Li-C)"}
    def cmat(df,ds,ts):
        R=np.full((len(ds),len(ts)),np.nan); P=np.ones_like(R)
        for i,d in enumerate(ds):
            for j,t in enumerate(ts):
                s=df.dropna(subset=[d,t])
                if len(s)>=6 and s[d].std()>0: R[i,j],P[i,j]=stats.pearsonr(s[d],s[t])
        return R,P
    Rf,Pf=cmat(arbr,F_DESC,["Ea_f","lnA_f"]); Rd,Pd=cmat(tr,D_DESC,["Ea_d","lnA_d"])
    def hm(ax,R,P,ds,lab,tlabs,title,yright=False):
        im=ax.imshow(R,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect="auto")
        ax.set_xticks(range(len(tlabs))); ax.set_xticklabels(tlabs,fontsize=12)
        ax.set_yticks(range(len(ds))); ax.set_yticklabels([lab[d] for d in ds],fontsize=10.5)
        if yright: ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
        for i in range(len(ds)):
            for j in range(R.shape[1]):
                if not np.isnan(R[i,j]):
                    st="*" if P[i,j]<0.05 else ""
                    ax.text(j,i,f"{R[i,j]:+.2f}{st}",ha="center",va="center",fontsize=10,
                            fontweight="bold" if P[i,j]<0.05 else "normal",
                            color="white" if abs(R[i,j])>0.6 else "black")
        ax.set_title(title,fontsize=10.5,fontweight="bold",pad=8); ax.grid(False); return im
    fig=plt.figure(figsize=(13.2,6.4))
    # FORMATION | DECOMPOSITION 拼成一块（中间挨着；分解 y 轴在右侧）
    gs_hm=fig.add_gridspec(1,2,left=0.10,right=0.47,top=0.80,bottom=0.24,wspace=0.015)
    gs_sc=fig.add_gridspec(3,1,left=0.73,right=0.975,top=0.90,bottom=0.09,hspace=0.62)
    axf=fig.add_subplot(gs_hm[0,0])
    im=hm(axf,Rf,Pf,F_DESC,F_LAB,["$E_{a,f}$","ln$A_f$"],"FORMATION  ←  ArBr precursor")
    axd=fig.add_subplot(gs_hm[0,1])
    hm(axd,Rd,Pd,D_DESC,D_LAB,["$E_{a,d}$","ln$A_d$"],"DECOMPOSITION  ←  ArLi intermediate",yright=True)
    cax=fig.add_axes([0.135,0.115,0.365,0.022])
    fig.colorbar(im,cax=cax,orientation="horizontal",label="Pearson r")
    def scat(ax,x,y,xl,yl,title):
        s=tr.dropna(subset=[x,y]); xx=s[x].values; yy=s[y].values; r,pp=stats.pearsonr(xx,yy)
        ax.scatter(xx,yy,s=46,c="#c0392b",edgecolor="k",lw=0.8,zorder=5)
        a,b=np.polyfit(xx,yy,1); xs=np.array([xx.min(),xx.max()]); ax.plot(xs,a*xs+b,"k--",lw=1.3,alpha=0.7)
        ax.set_xlabel(xl,fontsize=9.5); ax.set_ylabel(yl,fontsize=9.5)
        ax.set_title(f"{title}  r={r:+.2f}{'*' if pp<0.05 else ''}",fontsize=9.5,fontweight="bold"); ax.grid(alpha=0.3)
    scat(fig.add_subplot(gs_sc[0,0]),"Gsolv","lnA_d","$G_{solv}$ (kJ/mol)","ln$A_d$","decomp: $G_{solv}$→ln$A_d$")
    scat(fig.add_subplot(gs_sc[1,0]),"Gsolv","Ea_d","$G_{solv}$ (kJ/mol)","$E_{a,d}$","decomp: $G_{solv}$→$E_{a,d}$")
    scat(fig.add_subplot(gs_sc[2,0]),"sigma","Ea_d","Hammett $\\sigma$","$E_{a,d}$","decomp: σ→$E_{a,d}$ (FAILS)")
    fig.suptitle("Causal descriptors: formation ← PRECURSOR (ArBr)  |  decomposition ← INTERMEDIATE (ArLi)   —   C2, n=12",
                 fontsize=12,fontweight="bold",y=0.955)
    out=FIG/"fig_c2_mechanism_correlation.png"; fig.savefig(out,dpi=160); plt.close(fig); return out

Image(str(fig_c2_correlation()))
'''

CODE_SEC2B = r'''
# ===== Sec 2 · 全机理（未分层）描述符 × 动力学：整体弱，formation 尤其弱 =====
def fig_descriptor_mechanism_heatmap():
    m, cols = desc_table()
    tgts = ["Ea_d","lnA_d","Ea_f","lnA_f"]; tl = ["$E_{a,d}$","ln$A_d$","$E_{a,f}$","ln$A_f$"]
    Rm = np.full((len(cols),len(tgts)),np.nan); Pm = np.ones_like(Rm)
    for i,c in enumerate(cols):
        for j,t in enumerate(tgts):
            s = m.dropna(subset=[c,t])
            if len(s)>=6 and s[c].std()>0: Rm[i,j],Pm[i,j] = stats.pearsonr(s[c],s[t])
    fig, ax = plt.subplots(figsize=(5.6,7.2))
    im = ax.imshow(Rm,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect=0.7)
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels(tl,fontsize=12)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([LAB[c] for c in cols],fontsize=11); ax.axvline(1.5,color="k",lw=2)
    ax.text(0.5,-0.9,"DECOMPOSITION",ha="center",fontsize=9.5,fontweight="bold",color="0.35")
    ax.text(2.5,-0.9,"FORMATION",ha="center",fontsize=9.5,fontweight="bold",color="0.35")
    for i in range(len(cols)):
        for j in range(len(tgts)):
            if not np.isnan(Rm[i,j]):
                st = "*" if Pm[i,j]<0.05 else ""
                ax.text(j,i,f"{Rm[i,j]:+.2f}{st}",ha="center",va="center",fontsize=9,
                        fontweight="bold" if Pm[i,j]<0.05 else "normal",
                        color="white" if abs(Rm[i,j])>0.6 else "black")
    ax.set_title("Descriptor → kinetics across all mechanisms (Pearson r, * p<0.05)\n"
                 "formation column weak (mixing-contaminated); decomposition retains signal",
                 fontsize=10.5,fontweight="bold",pad=26)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="Pearson r"); ax.grid(False)
    out = FIG/"descriptor_mechanism_heatmap.png"; fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(fig); return out

Image(str(fig_descriptor_mechanism_heatmap()))
'''

MD_SEC3 = r"""## 3 · 发现 formation 反常 → 引入 Damköhler 观测层（统一框架）

formation 的表观 $E_{a,f}$ **无法**用分子描述符预测：数据库逐底物拟合的 $k_f$ 跨约 10 个数量级、与文献 Charton 化学速率几乎零相关。原因不是化学，而是 **flow 观测把化学与混合混为一谈**。

**统一因子化**（无底物专属拟合）：$k_{obs}=k_{chem}(\sigma,T)/(1+\mathrm{Da})$、$\mathrm{Da}=\tau_{eff}(T)\,k_{chem}$，全底物共享同一装置 $\tau_{eff}$（且 $\tau_{eff}(T)$ 随 THF 粘度变化——低温粘度大、混合慢、$\tau_{eff}$ 更大）。
- **弱/中等 σ（chemistry-identifiable，如 4BrF）**：$\mathrm{Da}<1$ → Charton 零(底物)拟合即预测对。
- **强 EWG（5Br2FCN）**：Charton 外推给极大 $k_{chem}$ → $\mathrm{Da}\gg1$ → $k_{obs}\approx1/\tau_{eff}(T)$，低温因粘度更低 → 低温低 yield 上升形状。

下方 regime map（Charton $k_{chem}$ 为 homogeneous-LFER 外推，强 EWG 处是速率**上界**）+ 两底物逐温曲线。**注意 5Br2FCN 低温/短-$t_R$ 的系统残差**（模型平坦高、实验低）——见结论的诚实边界说明。
"""

CODE_SEC3A = r'''
# ===== Sec 3 · Damköhler regime map：每底物固定 −50°C 单点（aryl-only）=====
def fig_regime_map():
    g = pd.read_csv(BASE/"global_arrhenius.csv")
    v = pd.read_csv(BASE/"v60_classified_substrates.csv")[["smi","class_v60","sigma_p_sum","sigma_m_sum"]]
    m = g.merge(v, on="smi", how="left").dropna(subset=["class_v60"]).copy()
    m["sigma"] = m["sigma_p_sum"].fillna(0) + m["sigma_m_sum"].fillna(0)
    m["grp"] = m["class_v60"].astype(str).map(dpf.to_display)
    groups = ["C1", "C2", "C3"]                        # 仅芳基类：Charton 远程-σ 适用
    m = m[m["grp"].isin(groups)].copy()
    T_REF, TAU = -50.0, 0.098e-3                       # 固定 −50°C；Nagaki 250 µm 平台 τ_eff
    m["Da"] = TAU * fm.k_chem(m["sigma"].values, T_REF)
    LBL = {"C1": "C1 · inert / weak-EWG", "C2": "C2 · remote-EWG",
           "C3": "C3 · ortho-chelation"}
    gcol = {x: dpf.DISPLAY_COLOR[x] for x in groups}; cnt = {x: int((m.grp == x).sum()) for x in groups}
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(5.9, 6.0)); YLO, YHI = 1e-4, 3e2   # 再瘦一点、砍掉上半空白
    ax.axhspan(YLO, 0.1, color="#27ae60", alpha=0.08)
    ax.axhspan(0.1, 10, color="#f1c40f", alpha=0.08)
    ax.axhspan(10, YHI, color="#c0392b", alpha=0.08)
    ax.axhline(0.1, color="0.55", ls="--", lw=0.9); ax.axhline(10, color="0.55", ls="--", lw=0.9)
    sg = np.linspace(-0.32, 0.85, 100)
    ax.plot(sg, TAU*fm.k_chem(sg, T_REF), "k-", lw=1.2, alpha=0.35, zorder=2)   # Charton 趋势(淡)
    for grp in groups:
        s = m[m.grp == grp]
        x = s["sigma"].values + rng.uniform(-0.012, 0.012, len(s))
        ax.scatter(x, s["Da"].values, s=120, c=[gcol[grp]], edgecolor="k", lw=0.7, zorder=5,
                   label=f"{LBL[grp]}  (n={cnt[grp]})")
    ax.set_yscale("log"); ax.set_xlim(-0.42, 0.9); ax.set_ylim(YLO, YHI)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.set_xlabel(r"Hammett $\sigma$", fontsize=14)
    # ax.set_xlabel(r"Hammett $\sigma$  (electron-withdrawing strength) →", fontsize=14)
    # ax.set_ylabel(r"Da $=\tau_{eff}\,k_{chem}$  (at $-50\,°$C)", fontsize=14)
    ax.set_ylabel(r"Da", fontsize=14)
    ax.text(0.88, 3e-4, "CHEMICAL (Da<0.1)", ha="right", va="bottom", color="#1e8449", fontweight="bold", fontsize=12)
    ax.text(0.88, 1.0, "TRANSITION", ha="right", va="center", color="#b9770e", fontweight="bold", fontsize=12)
    ax.text(0.88, 1.6e2, "MIXING (Da>10)", ha="right", va="top", color="#922b21", fontweight="bold", fontsize=12)
    # ax.set_title("Aryllithium Damköhler regime map (each substrate at $-50\\,°$C)\n"
    #              "stronger EWG → faster exchange → mixing control",
    #              fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", framealpha=0.93, fontsize=11)
    ax.grid(axis="y", which="major", alpha=0.12)
    out = FIG/"fig_regime_map.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return out

Image(str(fig_regime_map()))
'''

CODE_SEC3B = r'''
# ===== Sec 3 · 两底物：化学(虚) vs 分层(实, ×Da) 的 yield-tR 拟合 =====
def fig_formation_regime(key):
    cfg = next(c for c in VALIDATION if c["key"] == key)
    p = cfg["pred"]; sig = SIGMA[key]; R = fm.R_GAS
    df = _load_exp(cfg); tR = df["tR_s"].values; T = df["T_C"].values; y = df["val"].values
    Ts = sorted(set(np.round(T).astype(int)))
    fig, axs = plt.subplots(1, len(Ts), figsize=(3.0*len(Ts), 3.0), sharey=True); axs = np.atleast_1d(axs)
    tg = np.logspace(np.log10(tR.min()*0.7), np.log10(tR.max()*1.2), 200); col = PALETTE[key]
    for ax, Tc in zip(axs, Ts):
        m = np.round(T).astype(int) == Tc
        ax.scatter(tR[m], y[m], s=42, c="k", zorder=5, label="exp")
        kc = fm.k_chem(sig, Tc); kd = np.exp(p["lnA_d"] - p["Ea_d"]/(R*(Tc+273.15)))
        ax.plot(tg, p["y_max"]*(1-np.exp(-kc*tg))*np.exp(-kd*tg), "--", c="0.6", lw=1.5, label="chemistry only")
        keff = fm.observation_model(kc, tau_obs(Tc), form="series")
        ax.plot(tg, p["y_max"]*(1-np.exp(-keff*tg))*np.exp(-kd*tg), "-", c=col, lw=2.2, label="layered ×Da, τ$_{eff}$(T)")
        Da, reg = fm.da(sig, Tc, tau_obs(Tc))
        ax.set_xscale("log"); ax.set_title(f"{Tc}°C  Da={Da:.0f} ({reg})", fontsize=9)
        ax.set_xlabel("$t_R$ (s)"); ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)  # 半框（按 Zhao）
    axs[0].set_ylabel("yield (%)"); axs[0].set_ylim(0, 100); axs[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{cfg['name']} — layered model (fixed Charton χ × Da observation layer)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIG/f"fig_formation_regime_{key}.png"; fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig); return out

p1 = fig_formation_regime("5Br2FCN"); p2 = fig_formation_regime("4BrFC6H4")
Image(str(p1))
'''

CODE_SEC3C = r'''
Image(str(p2))   # 4BrF：低 σ → 化学控制（Da 小），固定 Charton 化学零拟合即贴合
'''

MD_SEC4 = r"""## 4 · 实验验证（两个 in-house 盲测底物）

`fig_experimental_validation`（实验热图 vs 预测热图 + parity）来自集中模型入口 `draw_paper_figures`；下方 parity 用分层模型（固定 Charton × Da 观测层）重算，化学层零每底物拟合。
"""

CODE_SEC4 = r'''
# ===== Sec 4 · 实验验证大图 + 分层模型 parity =====
dpf.fig_experimental_validation()                 # -> analysis_figures/fig_experimental_validation.png

def fig_final_parity():
    R = fm.R_GAS; fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for cfg in VALIDATION:
        p = cfg["pred"]; sig = SIGMA[cfg["key"]]; df = _load_exp(cfg)
        Tk = df["T_C"].values + 273.15
        keff = fm.observation_model(fm.k_chem(sig, df["T_C"].values), tau_obs(df["T_C"].values), form="series")
        kd = np.exp(p["lnA_d"] - p["Ea_d"]/(R*Tk))
        yp = p["y_max"]*(1-np.exp(-keff*df["tR_s"].values))*np.exp(-kd*df["tR_s"].values)
        ya = df["val"].values; mae = np.mean(np.abs(yp-ya))
        ax.scatter(ya, yp, s=50, c=PALETTE[cfg["key"]], edgecolor="k", lw=0.6,
                   label=f"{cfg['name'][:26]} (MAE {mae:.1f})", zorder=5)
    ax.plot([0,100],[0,100],"k--",lw=1.2,alpha=0.6)
    for off in (10,-10): ax.plot([0,100],[off,100+off],":",c="0.6",lw=0.8)
    ax.set_xlim(20,100); ax.set_ylim(20,100)
    ax.set_xlabel("measured yield (%)"); ax.set_ylabel("predicted yield (%)")
    ax.set_title("Layered model — prediction vs measured (±10 pp dotted)", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left"); ax.grid(alpha=0.25)
    out = FIG/"fig_final_model_parity.png"; fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig); return out

Image(str(fig_final_parity()))
'''

MD_SEC5 = r"""## 5 · 结论（统一框架）

1. **本征化学** `k_chem` = 均相 LFER（Charton eq.36，σ-in-lnA，固定锚点，零底物拟合）。
2. **观测层** `k_obs = k_chem/(1+τ_eff(T)·k_chem)`（唯象 resistance-in-series）；**`τ_eff(T)` 由 THF 粘度标定**——低温粘度大→混合慢→`τ_eff` 更大（$E_\eta$≈7.5 kJ/mol 文献值，`τ_ref`=15.9 ms@20°C 为唯一装置拟合量）。
3. **Da = τ_eff(T)·k_chem 控制 identifiability**：Da≪1 化学可见（4BrF 低 σ，Charton 盲预测）；Da≫1 被混合主导（5Br2FCN 高 σ，`k_obs`≈1/τ_eff(T)，低温因粘度更低 → 重现实验的低温低 yield 上升）。
4. **历史 flow 数据混淆这两层** —— 表观 `Ea_f` 看似可拟合却是混合/传递（粘度）活化能、非化学。核心 insight：**flow kinetics 不再直接代表 intrinsic chemistry。**

**诚实边界**：统一观测层（单一 `τ_ref` + 粘度律）修好了 5Br2FCN 低温上升形状（MAE 7.7→5.4），代价是化学控制的 4BrF 在低温被略微过度封顶（MAE 4.1→4.8）——**同一 τ_eff 无法同时完美拟合两种 regime**，这是"有边界的统一框架"而非 case-by-case 拟合。强 EWG 低温残差另含 n-BuLi/ArLi 聚集等本框架外的 cryogenic chemistry（SI：`fig_formation_refit` 的 substrate-specific 有效拟合可进一步部分恢复，但属描述、非盲预测）。
"""

MD_SI_A = r"""## SI-A · 机理分类总览 + 各类 yield 曲线

`fig_7c1`（各机理类的代表结构）与 `fig_7c2`（各类 yield-vs-tR 曲线，全局 Arrhenius 拟合）由 `draw_paper_figures` 生成。
"""

CODE_SI_A = r'''
# ===== SI-A · 机理分类结构 + 各类曲线 =====
df_mod = load_modeling_set()
dpf.fig_class_structures(df_mod)        # -> fig_7c1_class_structures.png
dpf.fig_class_yield_curves(df_mod)      # -> fig_7c2_yield_*.png (每类一张)
Image(str(FIG / "fig_7c1_class_structures.png"))
'''

MD_SI_B = r"""## SI-B · 全描述符相关性矩阵

GFN2-xTB 描述符之间的 Pearson 相关（多重共线性）—— 解释为何需要机理分层 + 单描述符建模而非堆全部描述符。
"""

CODE_SI_B = r'''
# ===== SI-B · 全描述符 × 描述符相关性热图 =====
def fig_descriptor_correlation_heatmap():
    m, cols = desc_table(); sub = m[cols].dropna(); C = sub.corr().values; n = len(cols)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(C, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels([LAB[c] for c in cols], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels([LAB[c] for c in cols], fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(C[i,j])>0.6 else "0.25")
    ax.set_title(f"Descriptor-descriptor correlation matrix (n={len(sub)} substrates)", fontsize=11.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r"); ax.grid(False)
    out = FIG/"descriptor_correlation_heatmap.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return out

Image(str(fig_descriptor_correlation_heatmap()))
'''

CELLS = [
    ("md", MD_TITLE), ("code", CODE_SETUP),
    ("md", MD_SEC1), ("code", CODE_SEC1),
    ("md", MD_SEC2), ("code", CODE_SEC2A), ("code", CODE_SEC2B),
    ("md", MD_SEC3), ("code", CODE_SEC3A), ("code", CODE_SEC3B), ("code", CODE_SEC3C),
    ("md", MD_SEC4), ("code", CODE_SEC4),
    ("md", MD_SEC5),
    ("md", MD_SI_A), ("code", CODE_SI_A),
    ("md", MD_SI_B), ("code", CODE_SI_B),
]

def to_nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n") + "\n"
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": lines})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}

def verify(cells):
    g = {"__name__": "__nb__"}
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), g)
        except Exception:
            print(f"\n!!! cell {i} FAILED:\n", traceback.format_exc())
            return False
    print("\nAll code cells executed OK.")
    return True

if __name__ == "__main__":
    ok = verify(CELLS)
    if ok or "--force" in sys.argv:
        json.dump(to_nb(CELLS), open(NB, "w"), ensure_ascii=False, indent=1)
        print("Wrote", NB)
    else:
        print("Verification failed — notebook NOT written.")
