"""
Task 3 — blind prediction of the two in-house substrates with the LAYERED model
(fixed Charton chemistry × transferable observation layer), + mandatory leave-one-substrate-out
cross-validation of τ_eff (the rig fingerprint).

  yield(tR,T) = y_max · (1 − exp(−k_f_obs·tR)) · exp(−kd·tR)
  k_f_obs     = observation_model(k_chem(σ,T), τ_eff)          # fixed Charton k_chem, no chem fit

Decomposition (kd) and y_max are kept from the current v6.2 model (unchanged). The ONLY free
quantity is the apparatus τ_eff. Two τ_eff forms are compared:
  (A) constant τ_eff
  (B) viscosity-T-dependent τ_eff(T) = τ_eff_ref · exp[(E_τ/R)(1/T − 1/T_ref)]   (E_τ ≈ THF
      transport activation; cold THF mixes slower). E_τ fixed (8 kJ/mol) → still ONE free rig param.
LOO: fit τ_eff on ONE substrate, predict the OTHER (tests transferability → publishable iff it works).
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path
import formation_model as fm
from draw_paper_figures import VALIDATION, _load_exp

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
BASE = Path(__file__).parent
R = fm.R_GAS

# σ for the two substrates (Charton input): 4BrF = para-F (σp=0.06); 5Br2FCN = σm(CN)+σp(F)=0.56+0.06
SIGMA = {"4BrFC6H4": 0.06, "5Br2FCN": 0.62}
E_TAU = 8.0          # kJ/mol — THF transport (viscosity) activation; fixes τ_eff(T) shape
T_REF_K = 273.15     # reference 0°C for τ_eff(T)
TAU_GEOM = fm.tau_eff(0.5, 7.5)   # 500 µm rig geometric bracket

def tau_of_T(tau_ref, T_C, tdep):
    if not tdep:
        return tau_ref
    return tau_ref * np.exp((E_TAU / R) * (1.0 / (np.asarray(T_C, float) + 273.15) - 1.0 / T_REF_K))

def predict(d, tau_ref, tdep):
    sig = SIGMA[d["key"]]
    kc = fm.k_chem(sig, d["T"])
    tau = tau_of_T(tau_ref, d["T"], tdep)
    keff = fm.observation_model(kc, tau, form="series")
    kd = np.exp(d["p"]["lnA_d"] - d["p"]["Ea_d"] / (R * (d["T"] + 273.15)))
    return d["p"]["y_max"] * (1 - np.exp(-keff * d["tR"])) * np.exp(-kd * d["tR"])

def mae(sets, tau_ref, tdep):
    e = [np.abs(predict(d, tau_ref, tdep) - d["y"]) for d in sets]
    return float(np.mean(np.concatenate(e)))

def fit_tau(sets, tdep):
    r = minimize_scalar(lambda lt: mae(sets, 10**lt, tdep), bounds=(-4, 0.5), method="bounded")
    return 10**r.x, r.fun

# ---- load both substrates ----
DS = []
for cfg in VALIDATION:
    df = _load_exp(cfg)
    DS.append(dict(key=cfg["key"], name=cfg["name"], p=cfg["pred"],
                   tR=df["tR_s"].values, T=df["T_C"].values, y=df["val"].values))
by = {d["key"]: d for d in DS}

print("="*74)
print(f"τ_eff geometric bracket (500 µm @7.5 mL/min): "
      f"{TAU_GEOM['lo_ms']:.2f}–{TAU_GEOM['hi_ms']:.2f} ms (eng~{TAU_GEOM['engulfment_good_ms']:.2f})")
print("="*74)
for tdep, tag in [(False, "constant τ_eff"), (True, f"τ_eff(T), E_τ={E_TAU} kJ/mol")]:
    tau_g, mae_g = fit_tau(DS, tdep)
    print(f"\n[{tag}]  global fit: τ_eff(0°C)={tau_g*1e3:7.2f} ms  →  global MAE={mae_g:5.2f} pp")
    for d in DS:
        print(f"    {d['key']:9s}  MAE(shared)={mae([d], tau_g, tdep):5.2f} pp")
    # LOO across substrates
    print("    leave-one-substrate-out (fit on A, predict B):")
    for tr_key in by:
        te_key = "5Br2FCN" if tr_key == "4BrFC6H4" else "4BrFC6H4"
        tau_tr, _ = fit_tau([by[tr_key]], tdep)
        loo = mae([by[te_key]], tau_tr, tdep)
        print(f"      fit {tr_key:9s} (τ={tau_tr*1e3:6.2f} ms) → predict {te_key:9s}: MAE={loo:5.2f} pp")

# baseline (old v6.2, no layered model) for reference
def baseline_mae(d):
    from draw_paper_figures import yld
    return float(np.mean(np.abs(yld(d["p"], d["tR"], d["T"]) - d["y"])))
print("\nbaseline v6.2 (no layered model):  " +
      "  ".join(f"{d['key']}={baseline_mae(d):.2f}" for d in DS))

# ---- figure: per-T curves, constant vs T-dep τ_eff, both substrates ----
tau_c, _ = fit_tau(DS, False)
tau_t, _ = fit_tau(DS, True)
for d in DS:
    Ts = sorted(set(np.round(d["T"]).astype(int)))
    fig, axs = plt.subplots(1, len(Ts), figsize=(3.0*len(Ts), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    tg = np.logspace(np.log10(d["tR"].min()*0.7), np.log10(d["tR"].max()*1.2), 200)
    for ax, Tc in zip(axs, Ts):
        m = np.round(d["T"]).astype(int) == Tc
        ax.scatter(d["tR"][m], d["y"][m], s=42, c="k", zorder=5, label="exp")
        for tau_ref, tdep, c, lab in [(tau_c, False, "#888", "const τ_eff"),
                                       (tau_t, True, "#c0392b", "τ_eff(T) visc.")]:
            sig = SIGMA[d["key"]]; kc = fm.k_chem(sig, Tc)
            keff = fm.observation_model(kc, tau_of_T(tau_ref, Tc, tdep), form="series")
            kd = np.exp(d["p"]["lnA_d"] - d["p"]["Ea_d"]/(R*(Tc+273.15)))
            ax.plot(tg, d["p"]["y_max"]*(1-np.exp(-keff*tg))*np.exp(-kd*tg), "-" if tdep else "--",
                    c=c, lw=1.8, label=lab)
        Da, reg = fm.da(SIGMA[d["key"]], Tc, tau_of_T(tau_t, Tc, True))
        ax.set_xscale("log"); ax.set_title(f"{Tc}°C  Da={Da:.0f}\n{reg}", fontsize=9)
        ax.set_xlabel("$t_R$ (s)"); ax.grid(alpha=0.25)
    axs[0].set_ylabel("yield (%)"); axs[0].set_ylim(0, 100); axs[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{d['name']} — layered model (fixed Charton χ × Da observation layer)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = BASE / "analysis_figures" / f"fig_formation_regime_{d['key']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("✓", out)
