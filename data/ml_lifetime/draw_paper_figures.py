#!/usr/bin/env python
"""
draw_paper_figures.py — FlowFigTabMiner organolithium-lifetime paper figures
(mechanism-classified version).

All model knobs live in the MODEL ENTRY block at the top of this file.
After changing the model / prediction parameters / classification, just rerun:

    python draw_paper_figures.py

Outputs (analysis_figures/):
  fig_7c1_class_structures.png          — structure overview grouped by mechanism (one big figure)
  fig_7c2_yield_<CLASS>.png             — yield-vs-tR curves per mechanism class (global Arrhenius fit)
  fig_experimental_validation.png       — 2 in-house substrates: experiment vs v6.2 prediction (3x2)
"""
import math
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from scipy.interpolate import Rbf
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "analysis_figures"
OUT.mkdir(exist_ok=True)

# ---- Global font: Arial for ALL figure text incl. tick labels ----
# (Nature/Springer figures use a sans-serif face; body text uses Times. Arial is the
#  conventional choice. Switch to "Times New Roman" below if a serif look is preferred.)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "axes.unicode_minus": False,
})

# ════════════════════════════════════════════════════════════════════════
#                        MODEL ENTRY  (集中模型入口)
#  改模型 / 预测参数 / 机理分类，只需修改本区块，然后重跑脚本即可更新所有图。
# ════════════════════════════════════════════════════════════════════════
R_GAS = 8.314e-3  # kJ/(mol·K)
import formation_model as _fm   # layered formation engine (Charton k_chem × Da observation layer)


def yld(p, tR, T_C):
    """Competing first-order kinetics: y = y_max·(1-exp(-kf·tR))·exp(-kd·tR).
    If p carries 'sigma', formation uses the LAYERED model
    k_f,obs = k_chem(σ,T)/(1+τ_eff·k_chem) (Charton intrinsic chemistry × Da observation
    layer); otherwise the plain Arrhenius k_f."""
    Tk = np.asarray(T_C, dtype=float) + 273.15
    if "sigma" in p:
        tau = _fm.tau_eff_T(p.get("tau_eff", 15.9e-3), T_C)   # viscosity-scaled (cold→larger τ_eff)
        kf = _fm.observation_model(_fm.k_chem(p["sigma"], T_C), tau, form="series")
    else:
        kf = np.exp(p["lnA_f"] - p["Ea_f"] / (R_GAS * Tk))
    kd = np.exp(p["lnA_d"] - p["Ea_d"] / (R_GAS * Tk))
    return p["y_max"] * (1 - np.exp(-kf * np.asarray(tR, dtype=float))) * np.exp(-kd * np.asarray(tR, dtype=float))


# ---- Mechanism classes (v6.0 == v6.2): order + display label + colour ----
MECH_ORDER = ["C1", "C2", "C3", "C4", "C5", "BENZYL", "CARBENOID", "OXI", "OTHER"]
MECH_LABEL = {
    "C1": "C1 · inert / proto-de-Li  (no EWG, no ortho group)",
    "C2": "C2 · remote-EWG  (p/m-CN, NO$_2$, CO$_2$R, CF$_3$)",
    "C3": "C3 · ortho-chelation  (o-CO$_2$R / OR / NR$_2$, Li$\\cdots$X)",
    "C4": "C4 · ortho-5-exo  (o-CN / NO$_2$ / CHO / C(=O)R)",
    "C5": "C5 · ortho-benzyne  (o-Br / o-I, 1,2-elimination)",
    "BENZYL": "Benzyllithium  (sp$^3$ Li–CH$_2$–Ar)",
    "CARBENOID": "Carbenoid  (sp$^3$ Li–CHX)",
    "OXI": "Oxiranyllithium  (epoxide, ring-opening)",
    "OTHER": "Other  (heteroaryl, etc.)",
}
MECH_COLOR = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(MECH_ORDER)}

# ---- Display grouping used in the figures (Zhao): C1 / C2 / C3 / OXI / Others ----
DISPLAY_ORDER = ["C1", "C2", "C3", "OXI", "OTHERS"]
DISPLAY_LABEL = {
    "C1": "C1 · inert / proto-de-Li",
    "C2": "C2 · remote-EWG  (p/m-CN, NO$_2$, CO$_2$R, CF$_3$)",
    "C3": "C3 · ortho-chelation  (Li$\\cdots$X)",
    "OXI": "Oxiranyllithium  (epoxide, ring-opening)",
    "OTHERS": "Others  (ortho-5-exo / benzyne / benzyl / carbenoid / heteroaryl)",
}
DISPLAY_COLOR = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(DISPLAY_ORDER)}
# descriptive filename slug per display group (class label moved out of the figure into the name)
GROUP_SLUG = {"C1": "C1_inert", "C2": "C2_remote-EWG", "C3": "C3_ortho-chelation",
              "OXI": "OXI_oxiranyllithium", "OTHERS": "OTHERS"}


def to_display(c):
    """Collapse the 9 mechanism classes into the 5 display groups."""
    return c if c in ("C1", "C2", "C3", "OXI") else "OTHERS"


def _anchor(cls, ewg):
    """v6.2 categorical anchor (class-mean) — used for classes without a descriptor model."""
    a = pd.read_csv(DATA_DIR / "v60_subclass_anchors.csv")
    r = a[(a["class"] == cls) & (a["ewg_type"] == ewg)].iloc[0]
    return dict(Ea_f=float(r["Ea_f_mean"]), lnA_f=float(r["lnA_f_mean"]),
                Ea_d=float(r["Ea_d_mean"]), lnA_d=float(r["lnA_d_mean"]),
                y_max=float(r["y_max_mean"]))


def _bayes_5Br2FCN():
    """v6.2 Bayesian descriptor prediction (σ+EWG_type+mol_volume) for the C2/CN substrate."""
    r = pd.read_csv(DATA_DIR / "v62_prediction_5Br2FCN.csv").iloc[0]
    return dict(Ea_f=float(r["Ea_f"]), lnA_f=float(r["lnA_f"]),
                Ea_d=float(r["Ea_d"]), lnA_d=float(r["lnA_d"]), y_max=float(r["y_max"]))


# ---- The two in-house validation substrates (experiment vs v6.2 prediction) ----
_TAU_REF = 15.9e-3   # s — 500 µm validation-rig effective mixing time at 20 °C (viscosity-scaled τ_eff(T))
VALIDATION = [
    dict(key="4BrFC6H4", name="1-bromo-4-fluorobenzene", smi="Fc1ccc(Br)cc1",
         exp_csv="experiment_phcho_summary_v2.csv", loader="phcho", quench="PhCHO",
         # tau_eff = 0 → Da→0 limit: k_obs = k_chem (chemistry-only). Substrate 1 sits in the
         # chemically-controlled regime; its data lies entirely on the trap-limited plateau and
         # carries no information about τ_eff, so reporting it via the chemistry-only baseline
         # is the physically faithful presentation. See SI Fig S14 for the diagnostic strip.
         pred={**_anchor("C1", "none"), "sigma": 0.06, "tau_eff": 0.0},
         pred_label="chemistry-only (Da$\\to$0): fixed Charton $k_{chem}$, no $\\tau_{eff}$ layer"),
    dict(key="5Br2FCN", name="5-bromo-2-fluorobenzonitrile", smi="N#Cc1cc(Br)ccc1F",
         # 2026-06-05: switch experimental source to the D-product-calibrated yields
         # (purified fluorobenzonitrile-5-d standard, y = 0.9092x − 0.1201). The
         # original non-D calibration over-counted product mass by a roughly 5 %
         # systematic offset; the D-cal CSV is the physically correct yield surface.
         exp_csv="experiment_fbrcn_d_calibrated_summary.csv", loader="fbrcn", quench="MeOD",
         pred={**_bayes_5Br2FCN(), "sigma": 0.62, "tau_eff": _TAU_REF},
         pred_label="strong-EWG: Charton $\\chi$ × Da, viscosity-scaled $\\tau_{eff}(T)$ ($\\tau_{ref}$=15.9 ms@20°C, $E_\\eta$=7.5)"),
]
# ════════════════════════════════════════════════════════════════════════


def load_modeling_set():
    """Mechanism-classified modeling set = v60_classified_substrates with reliability +
    tier filtering (identical to notebook Step 5/6)."""
    df = pd.read_csv(DATA_DIR / "v60_classified_substrates.csv")
    df = df[(df["r2_global"] > 0.6) & (df["Ea_d"] < 140)].copy()
    tier = pd.read_csv(DATA_DIR / "model_comparison_L1_L2.csv")
    tier["tc"] = tier["tier"].apply(
        lambda x: "A" if "Tier A" in str(x) else ("B" if "Tier B" in str(x) else "C"))
    excl = set(tier[tier["tc"].isin(["A", "B"])]["smi"])
    df = df[~df["smi"].isin(excl)].copy()
    df["class_v60"] = pd.Categorical(df["class_v60"], categories=MECH_ORDER, ordered=True)
    df["disp"] = pd.Categorical(df["class_v60"].astype(str).map(to_display),
                                categories=DISPLAY_ORDER, ordered=True)
    return df.sort_values(["disp", "Ea_d"]).reset_index(drop=True)


def mol_img(smi, size=(420, 300)):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().bondLineWidth = 2
    d.drawOptions().padding = 0.08
    d.DrawMolecule(m)
    d.FinishDrawing()
    return Image.open(BytesIO(d.GetDrawingText()))


def short(name):
    name = str(name).strip()
    for k, v in [("monolithiated ", ""), (" anion", ""), (" intermediate", ""),
                 ("lithium ", ""), ("lithiated ", ""), ("methyl ", "Me-"),
                 ("ethyl ", "Et-"), ("isopropyl ", "iPr-"), ("tert-butyl ", "tBu-")]:
        name = name.replace(k, v)
    return name


# ════════════════ 7c.1 — structure overview by mechanism (one big figure) ════════════════
def fig_class_structures(df, ncol=5):
    classes = [c for c in DISPLAY_ORDER if (df["disp"] == c).any()]
    nrows = {c: math.ceil(int((df["disp"] == c).sum()) / ncol) for c in classes}
    total = sum(nrows.values())
    fig = plt.figure(figsize=(ncol * 3.4, total * 3.1 + len(classes) * 0.3))
    sfigs = np.atleast_1d(fig.subfigures(len(classes), 1,
                          height_ratios=[nrows[c] for c in classes]))
    for gi, (sf, c) in enumerate(zip(sfigs, classes)):
        sub = df[df["disp"] == c].reset_index(drop=True)
        # thin separator at the top; panel letter sits above the first molecule, near the rule
        sf.add_artist(Line2D([0.004, 0.996], [0.985, 0.985], transform=sf.transSubfigure,
                             color="0.4", lw=1.3))
        axs = np.atleast_2d(sf.subplots(nrows[c], ncol, squeeze=False,
                            gridspec_kw=dict(wspace=0.05, hspace=0.12)))
        axs[0][0].text(-0.02, 1.07, f"({chr(97 + gi)})  {DISPLAY_LABEL[c]}   (n={len(sub)})",
                       transform=axs[0][0].transAxes,
                       fontsize=17, fontweight="bold", va="bottom", ha="left")
        for k, (_, row) in enumerate(sub.iterrows()):
            ax = axs[k // ncol][k % ncol]
            im = mol_img(row["smi"], size=(620, 460))
            if im is not None:
                ax.imshow(im)
            ax.axis("off")
        for k in range(len(sub), nrows[c] * ncol):
            axs[k // ncol][k % ncol].axis("off")
    p = OUT / "fig_7c1_class_structures.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓", p)
    return p


# ════════════════ 7c.2 — yield-vs-tR curves per mechanism class ════════════════
def fig_class_yield_curves(df, ncol=2):
    raw = pd.read_csv(DATA_DIR / "clean_organolithium_unified.csv")
    norm = Normalize(vmin=-78, vmax=25)
    cmap = plt.get_cmap("coolwarm")
    classes = [c for c in DISPLAY_ORDER if (df["disp"] == c).any()]
    paths = []
    for c in classes:
        sub = df[df["disp"] == c].reset_index(drop=True)
        n = len(sub)
        nc = ncol  # fixed columns -> every class figure has identical width
        nrows = math.ceil(n / nc)
        fig, axes = plt.subplots(nrows, nc, figsize=(6.6 * nc, 6.3 * nrows), squeeze=False)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax = axes[i // nc][i % nc]
            smi = row["smi"]
            pr = dict(Ea_f=row["Ea_f"], lnA_f=row["lnA_f"],
                      Ea_d=row["Ea_d"], lnA_d=row["lnA_d"], y_max=row["y_max"])
            r2g = row.get("r2_global", np.nan)
            rd = raw[raw["intermediate_smiles_canonical"] == smi][["tR1_s", "T1_C", "yield_pct"]].dropna()
            rd = rd[(rd["tR1_s"] > 0) & rd["yield_pct"].notna()]
            temps = sorted(rd["T1_C"].unique())
            for T in temps:
                dT = rd[rd["T1_C"] == T]
                ax.scatter(dT["tR1_s"], dT["yield_pct"], s=80, color=cmap(norm(T)),
                           edgecolor="black", linewidth=1.0, alpha=0.85, zorder=5)
            if len(rd) > 0:
                lo = max(rd["tR1_s"].min() * 0.5, 1e-4)
                hi = rd["tR1_s"].max() * 2
            else:
                lo, hi = 1e-3, 100
            tfit = np.logspace(np.log10(lo), np.log10(hi), 200)
            for T in temps:
                ax.plot(tfit, yld(pr, tfit, T), "-", color=cmap(norm(T)), lw=2.2,
                        alpha=0.9, label=f"T = {T:+.0f}°C")
            ax.set_xscale("log")
            ax.set_xlabel(r"$t_R$  (s)", fontsize=13)
            ax.set_ylabel("Yield  (%)", fontsize=13)
            ax.set_ylim(-5, 105)
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(alpha=0.3, linestyle="--")
            ax.text(-0.02, 1.25, f"({chr(ord('a') + i)})", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="top", ha="left")
            r2txt = f"R² = {r2g:.3f}" if not pd.isna(r2g) else "R² = N/A"
            txt = (f"Formation ($k_f$): $E_{{a,f}}$={pr['Ea_f']:.1f}, ln$A_f$={pr['lnA_f']:.2f}\n"
                   f"Decomp. ($k_d$): $E_{{a,d}}$={pr['Ea_d']:.1f}, ln$A_d$={pr['lnA_d']:.2f}\n"
                   f"$y_{{max}}$={pr['y_max']:.1f} %    {r2txt}")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=8.5, va="bottom",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                                 edgecolor="black", linewidth=1.0, alpha=0.95))
            if temps:
                ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85,
                          ncol=1 if len(temps) <= 5 else 2)
            # molecule name removed from the figure (kept commented for easy restore)
            # ax.set_title(short(row["intermediate"]), fontsize=11, fontweight="bold", pad=5)
            # structure inset at the top of each panel
            im = mol_img(smi, size=(360, 250))
            if im is not None:
                ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=0.30), (0.5, 1.26),
                              xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5)))
        for j in range(n, nrows * nc):
            axes[j // nc][j % nc].axis("off")
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.07,
                            hspace=0.72, wspace=0.22)
        p = OUT / f"fig_7c2_yield_{GROUP_SLUG[c]}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("✓", p)
        paths.append(p)
    return paths


# ════════════════ Experimental validation — 3×2 (cols = molecules) ════════════════
def _load_exp(cfg):
    df = pd.read_csv(DATA_DIR / cfg["exp_csv"])
    if cfg["loader"] == "phcho":
        df = df[df["date"] == 260512].copy()
        df["val"] = df["Yeild"] * 100
        df = df[~((df["L_cm"] == 2.5) & (df["T_C"] == -65))].copy()
        m = (df["L_cm"] == 2.5)
        df.loc[m, "L_cm"] = 10
        df.loc[m, "tR_s"] = 0.1571
    else:
        df["val"] = df["yield_pct"]
    df = df.dropna(subset=["tR_s", "T_C", "val"]).copy()
    df["pred"] = yld(cfg["pred"], df["tR_s"].values, df["T_C"].values)
    return df


def _fmt_logx(ax):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))


def _corner(ax, letter):
    """Panel letter as a bold left-aligned title, sitting OUTSIDE the axes in the
    top-left margin (Zhao). Coexists with a centred per-panel title if present."""
    if letter:
        ax.set_title(f"({letter})", loc="left", fontsize=15, fontweight="bold")


def _wada(ax, df, zcol, title=None, letter=None):
    d = df.dropna(subset=["tR_s", "T_C", zcol])
    x = np.log10(d["tR_s"].values); y = d["T_C"].values; z = d[zcol].values
    xi = np.linspace(x.min() - 0.05, x.max() + 0.05, 400)
    yi = np.linspace(y.min() - 3, y.max() + 3, 400)
    XI, YI = np.meshgrid(xi, yi)
    sx, sy = np.std(x) or 1, np.std(y) or 1
    ZI = np.clip(Rbf(x / sx, y / sy, z, function="linear", smooth=0.3)(XI / sx, YI / sy), 0, 110)
    cs = ax.contourf(XI, YI, ZI, levels=np.linspace(0, 100, 21), cmap="bwr",
                     norm=Normalize(0, 100), extend="both")
    _fmt_logx(ax)
    ax.scatter(x, y, s=28, facecolor="white", edgecolors="black", zorder=5, clip_on=False)
    xm, ym = x.mean(), y.mean()
    for _, r in d.iterrows():
        xv = np.log10(r["tR_s"]); yv = r["T_C"]; zv = r[zcol]
        if pd.isna(zv):
            continue
        ha = "left" if xv < xm else "right"; va = "top" if yv > ym else "bottom"
        ax.text(xv + (0.04 if ha == "left" else -0.04), yv + (1 if va == "bottom" else -1),
                f"{zv:.0f}", ha=ha, va=va, fontsize=8, clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    ax.set_xlabel("$t_R$ (s)"); ax.set_ylabel("$T$ (°C)")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    _corner(ax, letter)
    return cs


def _pred_heatmap(ax, df, pred, title=None, letter=None):
    Tv = np.linspace(df["T_C"].min() - 3, df["T_C"].max() + 3, 90)
    tRv = np.logspace(np.log10(df["tR_s"].min() * 0.8), np.log10(df["tR_s"].max() * 1.2), 90)
    Tg, tRg = np.meshgrid(Tv, tRv)
    Y = yld(pred, tRg, Tg)
    cs = ax.contourf(np.log10(tRg), Tg, Y, levels=np.linspace(0, 100, 21), cmap="bwr",
                     norm=Normalize(0, 100), extend="both")
    _fmt_logx(ax)
    ax.scatter(np.log10(df["tR_s"]), df["T_C"], s=28, facecolor="white",
               edgecolors="black", zorder=5, clip_on=False)
    xm = np.log10(df["tR_s"]).mean(); ym = df["T_C"].mean()
    for _, r in df.iterrows():
        xv = np.log10(r["tR_s"]); yv = r["T_C"]; pv = r["pred"]
        ha = "left" if xv < xm else "right"; va = "top" if yv > ym else "bottom"
        ax.text(xv + (0.04 if ha == "left" else -0.04), yv + (1 if va == "bottom" else -1),
                f"{pv:.0f}", ha=ha, va=va, fontsize=8, clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    ax.set_xlabel("$t_R$ (s)"); ax.set_ylabel("$T$ (°C)")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    _corner(ax, letter)
    return cs


def _parity(ax, df, label="", name="", smi=None, as_title=True):
    yo = df["val"].values; yp = df["pred"].values
    err = yo - yp
    mae = np.abs(err).mean(); rmse = np.sqrt((err ** 2).mean())
    ss = np.sum((yo - yo.mean()) ** 2)
    r2 = 1 - np.sum(err ** 2) / ss if ss > 0 else np.nan
    sc = ax.scatter(yo, yp, c=df["T_C"], cmap="coolwarm_r", vmin=-78, vmax=25, s=55,
                    edgecolor="black", linewidth=0.8, zorder=5)
    ax.plot([0, 110], [0, 110], "k--", lw=1.3, alpha=0.6, label="y = x")
    ax.fill_between([0, 110], [-10, 100], [10, 120], alpha=0.1, color="gray", label="±10 pp")
    ax.set_xlim(0, 110); ax.set_ylim(0, 110)
    ax.set_xlabel("Experimental yield (%)"); ax.set_ylabel("Predicted yield (%)")
    if as_title:
        ax.set_title(f"({label}) Parity — {name}\nR²={r2:+.2f}, MAE={mae:.1f}, RMSE={rmse:.1f} pp",
                     fontsize=10, fontweight="bold")
    else:
        _corner(ax, label)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    # substrate structure in the empty top-left corner of the parity plot
    if smi:
        im = mol_img(smi, (320, 230))
        if im is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=0.34), (0.27, 0.78),
                          xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5)))
    return sc  # colorbar drawn once, shared by both parity panels


def fig_experimental_validation(letters_only=True):
    """2×3 validation block (native proportions, figsize 18.5×11). Top component of the
    combined figure. `letters_only=True` → no in-panel titles, just a bold (letter) in the
    top-left margin (outside the axes); the 4 heatmaps share a yield colorbar and the 2
    parity panels share a T colorbar. Returns (path, metrics)."""
    fig = plt.figure(figsize=(18.5, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.32, right=0.88)
    exps = [_load_exp(cfg) for cfg in VALIDATION]
    letters = iter("abcdef")
    cs_heat = sc_par = None
    metrics = []
    for ri, cfg in enumerate(VALIDATION):
        exp = exps[ri]
        err = exp["val"] - exp["pred"]
        mae = float(err.abs().mean()); bias = float(err.mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        ss = float(((exp["val"] - exp["val"].mean()) ** 2).sum())
        r2 = 1 - float((err ** 2).sum()) / ss if ss > 0 else float("nan")
        metrics.append((cfg["name"], mae, bias, rmse, r2))
        if letters_only:
            _wada(fig.add_subplot(gs[ri, 0]), exp, "val", letter=next(letters))
            cs_heat = _pred_heatmap(fig.add_subplot(gs[ri, 1]), exp, cfg["pred"], letter=next(letters))
            sc_par = _parity(fig.add_subplot(gs[ri, 2]), exp, label=next(letters),
                             name=cfg["name"], smi=cfg["smi"], as_title=False)
        else:
            _wada(fig.add_subplot(gs[ri, 0]), exp, "val",
                  f"({next(letters)}) {cfg['name']}\nexperiment — {cfg['quench']} quench")
            cs_heat = _pred_heatmap(fig.add_subplot(gs[ri, 1]), exp, cfg["pred"],
                                    f"({next(letters)}) layered prediction\nMAE={mae:.1f}, bias={bias:+.1f} pp")
            sc_par = _parity(fig.add_subplot(gs[ri, 2]), exp,
                             label=next(letters), name=cfg["name"], smi=cfg["smi"])
    cax_y = fig.add_axes([0.905, 0.55, 0.013, 0.33])
    fig.colorbar(cs_heat, cax=cax_y, ticks=np.arange(0, 101, 20)).set_label(
        "yield (%)", rotation=270, labelpad=16)
    cax_t = fig.add_axes([0.905, 0.11, 0.013, 0.33])
    fig.colorbar(sc_par, cax=cax_t).set_label("T (°C)", rotation=270, labelpad=16)
    p = OUT / "fig_experimental_validation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓", p)
    return p, metrics


# ════════════════ Combined validation figure (a–h) ════════════════
PALETTE = {"4BrFC6H4": "#1f6fb2", "5Br2FCN": "#c0392b"}


def _regime_strip_fig(cfg, letter, tau_override=None, title_style="da_full"):
    """One substrate's per-temperature yield-vs-tR strip as a STANDALONE flat figure
    (native proportions, 3.0×3.0 per sub-panel → wide/flat strip). Chemistry-only (dashed)
    vs layered ×Da (solid); per-T centred title; (letter) in the top-left margin of
    the first sub-panel. Returns the saved path.

    Parameters
    ----------
    tau_override : float | None
        τ_eff (s) to use for the *layered* comparison curve. If None, falls back to
        ``cfg["pred"]["tau_eff"]``. Required when cfg's own τ_eff has been set to 0
        (Da→0 chemistry-only configuration) but a diagnostic strip still needs to
        contrast against a non-trivial layered curve.
    title_style : {"da_full", "t_only"}
        ``"da_full"`` → ``"{T}°C  Da={N} (regime)"`` (default; used for substrate 2).
        ``"t_only"`` → ``"{T}°C"`` only; suitable when the printed Da would be
        misleading (e.g., substrate 1, whose data does not constrain τ_eff)."""
    p = cfg["pred"]; sig = p["sigma"]; R = R_GAS
    tau_ref = tau_override if tau_override is not None else p.get("tau_eff", 15.9e-3)
    df = _load_exp(cfg)
    tR = df["tR_s"].values; T = df["T_C"].values; y = df["val"].values
    Ts = sorted(set(np.round(T).astype(int)))
    fig, axs = plt.subplots(1, len(Ts), figsize=(3.0 * len(Ts), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    tg = np.logspace(np.log10(tR.min() * 0.7), np.log10(tR.max() * 1.2), 200)
    col = PALETTE.get(cfg["key"], "#1f6fb2")
    for ax, Tc in zip(axs, Ts):
        m = np.round(T).astype(int) == Tc
        ax.scatter(tR[m], y[m], s=42, c="k", zorder=5, label="exp")
        kc = _fm.k_chem(sig, Tc); kd = np.exp(p["lnA_d"] - p["Ea_d"] / (R * (Tc + 273.15)))
        ax.plot(tg, p["y_max"] * (1 - np.exp(-kc * tg)) * np.exp(-kd * tg),
                "--", c="0.6", lw=1.5, label="chemistry only")
        tau = _fm.tau_eff_T(tau_ref, Tc)
        keff = _fm.observation_model(kc, tau, form="series")
        ax.plot(tg, p["y_max"] * (1 - np.exp(-keff * tg)) * np.exp(-kd * tg),
                "-", c=col, lw=2.2, label="layered ×Da, $\\tau_{eff}$(T)")
        if title_style == "t_only":
            ax.set_title(f"{Tc}°C", fontsize=9)
        else:
            Da, reg = _fm.da(sig, Tc, tau)
            ax.set_title(f"{Tc}°C  Da={Da:.0f} ({reg})", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("$t_R$ (s)"); ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("yield (%)"); axs[0].set_ylim(0, 100)
    _corner(axs[0], letter)                       # (g)/(h) in the top-left margin
    axs[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = OUT / f"_strip_{cfg['key']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_validation_composite():
    """Combined figure replacing fig5-result2: the 2×3 validation block (a–f) stacked above
    the two flat per-temperature regime strips (g, h). Each component is rendered at its own
    native proportions, then the three PNGs are scaled to a common width and stacked — so no
    sub-panel aspect ratio is altered and the strips stay flat (Zhao)."""
    val_png, metrics = fig_experimental_validation(letters_only=True)
    # only the strongly-activated substrate (5Br2FCN) keeps a per-temperature regime strip,
    # labelled (g); the chemically-controlled 4BrF strip is dropped (Zhao).
    g_png = _regime_strip_fig(VALIDATION[1], "g")
    imgs = [Image.open(q).convert("RGB") for q in (val_png, g_png)]
    W = max(im.width for im in imgs)
    scaled = [im if im.width == W else
              im.resize((W, round(im.height * W / im.width)), Image.LANCZOS) for im in imgs]
    gap = 24
    H = sum(im.height for im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (W, H), "white")
    yoff = 0
    for im in scaled:
        canvas.paste(im, (0, yoff)); yoff += im.height + gap
    p = OUT / "fig5-result2_composite.png"
    canvas.save(p)
    print("✓", p)
    for name, mae, bias, rmse, r2 in metrics:
        print(f"   {name}: MAE={mae:.1f}  bias={bias:+.1f}  RMSE={rmse:.1f}  R2={r2:+.2f}")
    return p


if __name__ == "__main__":
    import shutil
    df = load_modeling_set()
    print(f"Modeling set: {len(df)} compounds")
    print(df["class_v60"].value_counts().reindex(MECH_ORDER).dropna().to_string())
    print("-" * 60)
    fig_class_structures(df)
    fig_class_yield_curves(df)
    fig_validation_composite()   # also (re)writes fig_experimental_validation.png internally
    # ---- SI Fig S14: substrate 1 diagnostic strip ----
    # tau_override = _TAU_REF: VALIDATION[0]["pred"]["tau_eff"] is 0 (chemistry-only); the
    # diagnostic strip must contrast chemistry-only against the *global* layered curve, so we
    # force the global τ_eff for the layered comparison here.
    sub1_strip = _regime_strip_fig(VALIDATION[0], "", tau_override=_TAU_REF, title_style="t_only")
    si_dst = Path(__file__).resolve().parents[2] / "pub" / "pub-sn" / "figures" / "si" / "fig_substrate1_regime.png"
    shutil.copy2(sub1_strip, si_dst)
    print("✓", si_dst)
    print("All figures written to", OUT)
