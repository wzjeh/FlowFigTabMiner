"""
Layered formation-kinetics model for ArLi generation by halogen–lithium (Br/Li) exchange.

    observed formation  =  intrinsic chemistry (k_chem)  ×  observation layer (Da, τ_eff)

Rationale (see plan / project memory): historical flow kinetics conflate intrinsic exchange
chemistry with micromixing-limited *observation*. We factorize them:

  • k_chem(σ,T)  — intrinsic, pure-chemical exchange rate. ANCHORED (fixed) to the homogeneous
    -solution LFER of Batalov–Rostokin compiled in Charton 2004 (eq. 36):
        log10 k = 5.07·σ − 2080/T + 6.84        (PhLi + ArBr, n=15, R²=95.55%)
    The σ term sits in the INTERCEPT (lnA), the −2080/T term carries NO σ → activation energy is
    constant and the substituent effect lives entirely in lnA (physically faithful to eq. 36):
        Ea_f = const,   lnA_f(σ) = lnA0 + (5.07·ln10)·σ = lnA0 + 11.67·σ
    nBuLi-calibrated baseline (Leroux/Schlosser n-BuLi vs PhLi): Ea_f = 32 kJ/mol, lnA0 = 21.
    This anchor is FIXED — not re-fit to Da-contaminated flow data.

  • observation layer — a MINIMAL, PHENOMENOLOGICAL "resistance-in-series" approximation
    (NOT a rigorous micromixing theory; real micromixing may follow engulfment / striation /
    segregation / Villermaux–Dushman forms). τ_eff is an EFFECTIVE (phenomenological) mixing
    timescale, τ_eff ≠ τ_hydrodynamic — it lumps engulfment + local concentration equilibration
    + injection heterogeneity + quench lag + finite segmentation.

Da = τ_eff · k_chem.  Da≪1 chemistry-controlled (k_obs→k_chem); Da≫1 mixing-capped (k_obs→1/τ_eff).
"""
import numpy as np

# ── intrinsic chemistry anchor (FIXED — Charton 2004 eq. 36, n-BuLi-calibrated) ──────────────
R_GAS    = 8.314e-3      # kJ/(mol·K)
EA_F     = 32.0          # kJ/mol — constant activation energy (σ-independent, eq. 36)
LNA0     = 21.0          # ln A baseline at σ=0 (n-BuLi/ArBr)
RHO_LNA  = 5.07 * np.log(10.0)   # = 11.67 — Charton ρ projected into lnA (σ slope)

# ── solvent / transport constants for the τ_eff geometric estimator (THF, approximate) ───────
MU_THF   = 0.48e-3      # Pa·s   (≈0.48 cP)
RHO_THF  = 889.0        # kg/m³
NU_THF   = MU_THF / RHO_THF       # m²/s kinematic viscosity
D_MOL    = 1.0e-9       # m²/s   molecular diffusivity (order of magnitude)

# ── temperature dependence of τ_eff via THF viscosity (cold THF mixes slower) ───────────────
# τ_eff(T) = τ_ref · exp(E_eta/R · (1/T − 1/T_ref)).  E_eta = THF viscosity activation energy
# (≈7.5 kJ/mol, literature; NOT a free parameter). τ_ref is the rig fingerprint at T_ref.
E_ETA       = 7.5       # kJ/mol — THF viscosity (transport) activation energy
T_REF_VISC  = 293.15    # K (20 °C) reference for τ_ref


def tau_eff_T(tau_ref, T_C):
    """Effective mixing time at temperature T_C, viscosity-scaled from τ_ref (defined at 20 °C).
    Higher viscosity at low T → larger τ_eff → smaller capped k_obs (lower low-T yield)."""
    Tk = np.asarray(T_C, float) + 273.15
    return tau_ref * np.exp(E_ETA / R_GAS * (1.0 / Tk - 1.0 / T_REF_VISC))


def k_chem(sigma, T_C):
    """Intrinsic pure-chemical Br/Li exchange rate (pseudo-first-order, /s). Fixed Charton anchor."""
    Tk = np.asarray(T_C, float) + 273.15
    lnA = LNA0 + RHO_LNA * np.asarray(sigma, float)
    return np.exp(lnA - EA_F / (R_GAS * Tk))


def tau_eff(mixer_id_mm, Q_mL_min):
    """Geometric ESTIMATE / bracket of the effective mixing timescale τ_eff (s) for a T-mixer.

    Returns a dict: convective (d/u), engulfment (good & poor mixer), and a recommended value
    + a plausible [lo, hi] bracket. NOTE: this is an order-of-magnitude estimator used only to
    sanity-check a τ_eff that is otherwise CALIBRATED from data. τ_eff is phenomenological.
    """
    d = mixer_id_mm * 1e-3                       # m
    A = np.pi * (d / 2) ** 2
    Q = Q_mL_min / 60.0 * 1e-6                   # m³/s
    u = Q / A
    Re = RHO_THF * u * d / MU_THF
    convective = d / u
    # engulfment τ_E = 17.24·sqrt(ν/ε); ε ≈ C·u³/d  (C=1 ideal mixer, C≪1 poor mixer)
    eng_good = 17.24 * np.sqrt(NU_THF / (1.0 * u**3 / d))
    eng_poor = 17.24 * np.sqrt(NU_THF / (0.02 * u**3 / d))
    return dict(u_cm_s=u * 100, Re=Re,
                convective_ms=convective * 1e3,
                engulfment_good_ms=eng_good * 1e3,
                engulfment_poor_ms=eng_poor * 1e3,
                lo_ms=convective * 1e3, hi_ms=eng_poor * 1e3,
                recommended_ms=eng_good * 1e3)


def observation_model(kc, tau, tR=None, form="series"):
    """Map intrinsic rate kc → OBSERVED formation rate k_f,obs through the observation layer.

    Generic interface so the functional form can be swapped later (saturation / logistic /
    engulfment-law / Villermaux). Default `series` = phenomenological resistance-in-series:
        k_f,obs = kc / (1 + τ_eff·kc)              (Da≪1 → kc;  Da≫1 → 1/τ_eff)
    `tR` accepted for future forms that need residence time; unused by `series`.
    """
    kc = np.asarray(kc, float)
    if form == "series":
        return kc / (1.0 + tau * kc)
    raise NotImplementedError(f"observation_model form='{form}' not implemented "
                              f"(reserved: saturation/logistic/engulfment/villermaux)")


def da(sigma, T_C, tau):
    """Damköhler number and regime label from the fixed chemistry anchor + an effective τ_eff."""
    Da = tau * k_chem(sigma, T_C)
    Da_arr = np.atleast_1d(Da)
    lab = np.where(Da_arr < 0.3, "chemical",
          np.where(Da_arr < 3.0, "transition", "mixing"))
    return (Da, lab) if Da_arr.size > 1 else (float(Da), str(lab[0]))


if __name__ == "__main__":
    # quick self-check: his 500 µm validation rig
    te = tau_eff(0.5, 7.5)
    print("τ_eff geometric estimate (500 µm @7.5 mL/min):")
    for k, v in te.items():
        print(f"  {k:20s} = {v:.3f}")
    print("\nDa across T for σ=0.06 (4BrF) and σ=0.60 (5Br2FCN), τ_eff=25 ms:")
    for sig, name in [(0.06, "4BrF"), (0.60, "5Br2FCN")]:
        for T in (-70, -50, -25, 0, 20):
            kc = float(k_chem(sig, T)); Da, lab = da(sig, T, 25e-3)
            print(f"  {name:8s} {T:+4d}°C: k_chem={kc:9.2f}/s  Da={Da:7.2f}  {lab}")
