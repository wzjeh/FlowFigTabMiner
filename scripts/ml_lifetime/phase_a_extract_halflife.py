"""
Phase A: Extract organolithium intermediate half-lives from yield-vs-tR decay curves.

Kinetic model (competing formation + decomposition):
  yield(tR) = y_max * (k_f / (k_d - k_f)) * (exp(-k_f*tR) - exp(-k_d*tR))     [general]
  yield(tR) ≈ y_max * (1 - exp(-k_f*tR)) * exp(-k_d*tR)                        [k_f >> k_d]

For each (intermediate, T) pair, fits the model and extracts:
  - k_f (formation rate, s⁻¹)
  - k_d (decomposition rate, s⁻¹)
  - t_half = ln(2) / k_d (half-life, s)
  - t_peak (tR at max yield)
  - y_max (peak yield %)
  - fit_r2

Input:  data/final_output/organolithium_tr_subdataset_vlm_enriched.csv
Output: data/ml_lifetime/phase_a_halflives.csv
        data/ml_lifetime/phase_a_curves/  (per-intermediate PNG plots)

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/phase_a_extract_halflife.py
"""

import os
import csv
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data/final_output/organolithium_tr_subdataset_vlm_enriched.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_a_halflives.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_a_curves")

# Minimum points per (intermediate, T) to attempt fitting
MIN_POINTS = 4
# Minimum yield drop from peak to end to consider "decay visible"
MIN_DECAY_DROP = 10  # percentage points


def load_tr1_data():
    """Load tR1 data, grouped by (intermediate_smiles, T)."""
    groups = defaultdict(list)
    with open(INPUT_CSV) as f:
        for r in csv.DictReader(f):
            if r["tR_step"] != "tR1":
                continue
            smi = r.get("intermediate_smiles", "").strip()
            t_str = r.get("T1_C", "").strip()
            tr_str = r.get("tR1_s", "").strip()
            y_str = r.get("yield_pct", "").strip()
            if not (smi and t_str and tr_str and y_str):
                continue
            T_C = float(t_str)
            tR = float(tr_str)
            y = float(y_str)
            inter_name = r.get("intermediate", "")
            paper = r.get("paper", "")[:50]
            groups[(smi, T_C)].append({
                "tR": tR, "yield": y,
                "intermediate": inter_name, "paper": paper,
            })
    return groups


# ── Kinetic models ──

def _competing_kinetics(tR, k_f, k_d, y_max):
    """Competing formation + decomposition model.
    yield = y_max * (1 - exp(-k_f * tR)) * exp(-k_d * tR)
    """
    with np.errstate(over="ignore"):
        formation = 1.0 - np.exp(-k_f * tR)
        decay = np.exp(-k_d * tR)
    return y_max * formation * decay


def _formation_only(tR, k_f, y_max):
    """Formation-only model (no visible decay).
    yield = y_max * (1 - exp(-k_f * tR))
    """
    return y_max * (1.0 - np.exp(-k_f * tR))


def fit_curve(tR_arr, y_arr):
    """Fit competing kinetics model. Returns dict of fitted params or None."""
    tR = np.array(tR_arr, dtype=float)
    y = np.array(y_arr, dtype=float)

    # Sort by tR
    order = np.argsort(tR)
    tR = tR[order]
    y = y[order]

    peak_idx = np.argmax(y)
    y_peak = y[peak_idx]
    tR_peak = tR[peak_idx]

    # Check if decay is visible
    has_decay = (peak_idx < len(y) - 1) and (y_peak - y[-1] > MIN_DECAY_DROP)

    if has_decay:
        # Fit competing kinetics
        # Initial guesses from data
        k_f_init = 1.0 / max(tR_peak, 1e-6)  # formation completes around peak
        k_d_init = 0.1 * k_f_init  # decay slower than formation
        y_max_init = y_peak * 1.1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    _competing_kinetics, tR, y,
                    p0=[k_f_init, k_d_init, y_max_init],
                    bounds=([1e-6, 1e-10, 1.0], [1e8, 1e6, 150.0]),
                    maxfev=10000,
                )
            k_f, k_d, y_max = popt

            # Compute R²
            y_pred = _competing_kinetics(tR, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Compute t_peak from model: d/dtR = 0 → tR_peak = ln(k_f/k_d) / (k_f - k_d)
            if k_f > k_d and k_f != k_d:
                t_peak_model = np.log(k_f / k_d) / (k_f - k_d)
            else:
                t_peak_model = tR_peak

            t_half = np.log(2) / k_d

            return {
                "model": "competing",
                "k_f": k_f,
                "k_d": k_d,
                "t_half": t_half,
                "t_peak": t_peak_model,
                "y_max": y_max,
                "r2": r2,
                "tR_data": tR,
                "y_data": y,
            }
        except (RuntimeError, ValueError):
            pass

    # Fallback: formation-only (no measurable decay)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                _formation_only, tR, y,
                p0=[1.0 / max(tR_peak, 1e-6), y_peak * 1.1],
                bounds=([1e-6, 1.0], [1e8, 150.0]),
                maxfev=10000,
            )
        k_f, y_max = popt
        y_pred = _formation_only(tR, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "model": "formation_only",
            "k_f": k_f,
            "k_d": None,
            "t_half": None,  # no measurable decay
            "t_peak": None,
            "y_max": y_max,
            "r2": r2,
            "tR_data": tR,
            "y_data": y,
        }
    except (RuntimeError, ValueError):
        return None


def plot_fits(results_by_intermediate):
    """Generate per-intermediate plots showing data + fitted curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    for inter_name, fits in results_by_intermediate.items():
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        ax = axes

        # Sort by temperature
        fits_sorted = sorted(fits, key=lambda f: f["T_C"])
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(fits_sorted)))

        for fit, color in zip(fits_sorted, colors):
            T = fit["T_C"]
            tR = fit["tR_data"]
            y = fit["y_data"]
            label = f"T={T:.0f}°C"

            ax.scatter(tR, y, color=color, s=20, alpha=0.7)

            # Plot fitted curve
            tR_fine = np.geomspace(max(tR.min(), 1e-4), tR.max(), 200)
            if fit["model"] == "competing":
                y_fit = _competing_kinetics(tR_fine, fit["k_f"], fit["k_d"], fit["y_max"])
                t_h = fit["t_half"]
                label += f" (t₁/₂={t_h:.3g}s, R²={fit['r2']:.2f})"
            else:
                y_fit = _formation_only(tR_fine, fit["k_f"], fit["y_max"])
                label += f" (no decay, R²={fit['r2']:.2f})"
            ax.plot(tR_fine, y_fit, color=color, linewidth=1.5, label=label)

        ax.set_xscale("log")
        ax.set_xlabel("tR₁ (s)", fontsize=12)
        ax.set_ylabel("Yield (%)", fontsize=12)
        ax.set_title(inter_name[:60], fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.set_ylim(-5, 110)
        ax.grid(True, alpha=0.3)

        safe_name = inter_name.replace(" ", "_").replace("/", "_")[:40]
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{safe_name}.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved {len(results_by_intermediate)} plots to {PLOT_DIR}")


def main():
    groups = load_tr1_data()
    print(f"[Phase A] Loaded {len(groups)} (intermediate, T) groups")

    # Fit each group
    output_rows = []
    results_by_inter = defaultdict(list)  # for plotting
    n_fit = 0
    n_decay = 0
    n_formation_only = 0
    n_skip = 0

    for (smi, T_C), points in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(points) < MIN_POINTS:
            n_skip += 1
            continue

        tR_arr = [p["tR"] for p in points]
        y_arr = [p["yield"] for p in points]
        inter_name = points[0]["intermediate"]
        paper = points[0]["paper"]

        result = fit_curve(tR_arr, y_arr)
        if result is None:
            n_skip += 1
            continue

        n_fit += 1
        row = {
            "intermediate": inter_name,
            "intermediate_smiles": smi,
            "T_C": T_C,
            "T_K": T_C + 273.15,
            "n_points": len(points),
            "model": result["model"],
            "k_f": f"{result['k_f']:.6g}",
            "k_d": f"{result['k_d']:.6g}" if result["k_d"] is not None else "",
            "t_half_s": f"{result['t_half']:.6g}" if result["t_half"] is not None else "",
            "t_peak_s": f"{result['t_peak']:.6g}" if result["t_peak"] is not None else "",
            "y_max_pct": f"{result['y_max']:.1f}",
            "fit_r2": f"{result['r2']:.4f}",
            "paper": paper,
        }
        output_rows.append(row)

        # Filter: if k_d hit lower bound (t_half > 1e6), reclassify as formation_only
        if result["model"] == "competing" and result["t_half"] is not None and result["t_half"] > 1e6:
            row["model"] = "formation_only"
            row["k_d"] = ""
            row["t_half_s"] = ""
            row["t_peak_s"] = ""
            result["model"] = "formation_only"  # for plotting
            result["k_d"] = None
            result["t_half"] = None

        if result["model"] == "competing":
            n_decay += 1
        else:
            n_formation_only += 1

        # For plotting
        fit_plot = {
            "T_C": T_C,
            "model": result["model"],
            "k_f": result["k_f"],
            "k_d": result["k_d"],
            "y_max": result["y_max"],
            "t_half": result["t_half"],
            "r2": result["r2"],
            "tR_data": result["tR_data"],
            "y_data": result["y_data"],
        }
        results_by_inter[inter_name].append(fit_plot)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [
        "intermediate", "intermediate_smiles", "T_C", "T_K", "n_points",
        "model", "k_f", "k_d", "t_half_s", "t_peak_s", "y_max_pct",
        "fit_r2", "paper",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Report
    print(f"\n{'=' * 65}")
    print(f"PHASE A RESULTS: {OUTPUT_CSV}")
    print(f"{'=' * 65}")
    print(f"Total (intermediate, T) groups: {len(groups)}")
    print(f"  Fitted:         {n_fit}")
    print(f"    with decay:   {n_decay}")
    print(f"    formation only: {n_formation_only}")
    print(f"  Skipped:        {n_skip}")

    # Summary per intermediate
    print(f"\n── Half-lives by intermediate ──")
    for inter_name, fits in sorted(results_by_inter.items()):
        decay_fits = [f for f in fits if f["model"] == "competing"]
        if decay_fits:
            t_halves = [(f["T_C"], f["t_half"]) for f in decay_fits]
            t_halves.sort()
            print(f"\n  {inter_name}:")
            for T, th in t_halves:
                print(f"    T={T:>6.0f}°C: t₁/₂ = {th:.4g} s")
        else:
            print(f"\n  {inter_name}: no measurable decay")

    # Validation: compare with known lifetimes
    print(f"\n── Validation vs. known lifetimes ──")
    known = {
        "[Li]CF": {"lifetime_s": 0.013, "T_C": -60, "name": "fluoromethyllithium"},
        "[Li]C(F)I": {"lifetime_s": 0.082, "T_C": -40, "name": "iodofluoromethyllithium"},
    }
    for smi, info in known.items():
        matched = [r for r in output_rows
                   if r["intermediate_smiles"] == smi
                   and float(r["T_C"]) == info["T_C"]
                   and r["t_half_s"]]
        if matched:
            extracted = float(matched[0]["t_half_s"])
            literature = info["lifetime_s"]
            ratio = extracted / literature
            print(f"  {info['name']}:")
            print(f"    Literature:  {literature*1000:.1f} ms at {info['T_C']}°C")
            print(f"    Extracted:   {extracted*1000:.1f} ms at {info['T_C']}°C")
            print(f"    Ratio:       {ratio:.2f}x")
        else:
            print(f"  {info['name']}: no matching fit at {info['T_C']}°C")

    # Generate plots
    plot_fits(results_by_inter)

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
