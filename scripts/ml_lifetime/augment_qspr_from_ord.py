"""
Augment QSPR training data with ArLi intermediates inferred from ORD batch reactions.

Strategy:
  1. Filter ORD for halogen-metal exchange reactions (ArX + RLi → ArLi → product)
  2. Infer ArLi intermediate SMILES: ArX with X→Li replacement
  3. Label stability: yield>80% at -78°C → "batch_stable", <30% → "batch_unstable"
  4. Deduplicate by canonical SMILES
  5. Compute molecular descriptors (reuse from phase_c_qspr.py)
  6. Train augmented QSPR: 11 flow (Ea) + N_ord (binary label)

Input:  other dataset/ord/organolithium/ord_organolithium_reactions_rich.csv
Output: data/ml_lifetime/ord_inferred_intermediates.csv
        data/ml_lifetime/augmented_qspr_results.csv
        data/ml_lifetime/analysis_figures/augmented_qspr_*.png

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/augment_qspr_from_ord.py
"""

import os, sys, json, csv, re, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/ml_lifetime"))
from phase_c_qspr import compute_descriptors

ORD_CSV = os.path.join(PROJECT_ROOT, "other dataset/ord/organolithium/ord_organolithium_reactions_rich.csv")
FLOW_ARRHENIUS = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime")
PLOT_DIR = os.path.join(OUTPUT_DIR, "analysis_figures")

os.makedirs(PLOT_DIR, exist_ok=True)


def strip_atom_map(smiles):
    """Remove atom-mapping from SMILES: [C:1] → [C], :2 → ''."""
    s = re.sub(r':(\d+)', '', smiles)
    # Also convert [CH3] → C, [CH2] → C etc. for cleaner SMILES
    mol = Chem.MolFromSmiles(s)
    if mol:
        return Chem.MolToSmiles(mol)
    return None


def infer_arli_from_arx(arx_smiles, halide='Br'):
    """Replace the first Br or I with [Li] to infer ArLi intermediate."""
    mol = Chem.MolFromSmiles(arx_smiles)
    if mol is None:
        return None

    target_num = 35 if halide == 'Br' else 53  # Br=35, I=53
    from rdkit.Chem import RWMol, Atom

    rw = RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == target_num:
            atom.SetAtomicNum(3)  # Li=3
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            break
    else:
        return None

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except:
        return None


def extract_ord_intermediates():
    """Extract ArLi intermediates from ORD halogen-metal exchange reactions."""
    print("Loading ORD data...")
    df = pd.read_csv(ORD_CSV, low_memory=False)
    print(f"  {len(df)} total reactions")

    records = []
    n_with_yield = 0
    n_arx_found = 0
    n_arli_valid = 0

    for i in range(len(df)):
        rxn = str(df['reaction_smiles'].iloc[i])
        oc = df['outcomes_json'].iloc[i]
        cj = df['conditions_json'].iloc[i]
        mc = df['matched_components_json'].iloc[i]

        # Get yield
        yield_pct = None
        if pd.notna(oc):
            try:
                for outcome in json.loads(oc):
                    if isinstance(outcome, dict):
                        for prod in outcome.get('products', []):
                            for m in prod.get('measurements', []):
                                if m.get('type') == 'YIELD':
                                    v = m.get('percentage', {}).get('value')
                                    if v is not None and 0 <= v <= 100:
                                        yield_pct = v
            except: pass

        # Get temperature
        temp_C = None
        if pd.notna(cj):
            try:
                cd = json.loads(cj)
                tp = cd.get('temperature', {}).get('setpoint', {})
                if tp.get('value') is not None and 'CELSIUS' in str(tp.get('units', '')):
                    temp_C = tp['value']
            except: pass

        if yield_pct is None or temp_C is None:
            continue
        n_with_yield += 1

        # Get OrgLi reagent name
        orgLi_name = ""
        if pd.notna(mc):
            try:
                for comp in json.loads(mc):
                    val = comp.get('matched_value', '').lower()
                    if 'lithium' in val:
                        orgLi_name = val
                        break
            except: pass

        # Parse reaction SMILES → find ArX substrate
        if '>>' in rxn:
            reactant_part = rxn.split('>>')[0]
        elif '>' in rxn:
            reactant_part = rxn.split('>')[0]
        else:
            continue

        # Find ArX (aromatic halide) among reactants
        best_arx = None
        best_halide = None
        for frag in reactant_part.split('.'):
            if '[Li]' in frag:
                continue
            clean = strip_atom_map(frag)
            if clean is None:
                continue
            mol = Chem.MolFromSmiles(clean)
            if mol is None:
                continue

            has_br = any(a.GetAtomicNum() == 35 for a in mol.GetAtoms())
            has_i = any(a.GetAtomicNum() == 53 for a in mol.GetAtoms())
            has_ring = mol.GetRingInfo().NumRings() > 0
            n_heavy = mol.GetNumHeavyAtoms()

            if (has_br or has_i) and has_ring and n_heavy >= 5:
                # Prefer I over Br (faster halogen-metal exchange)
                if has_i:
                    best_arx = clean
                    best_halide = 'I'
                elif best_arx is None:
                    best_arx = clean
                    best_halide = 'Br'

        if best_arx is None:
            continue
        n_arx_found += 1

        # Infer ArLi
        arli_smiles = infer_arli_from_arx(best_arx, best_halide)
        if arli_smiles is None:
            continue
        n_arli_valid += 1

        records.append({
            'arx_smiles': best_arx,
            'arli_smiles': arli_smiles,
            'halide': best_halide,
            'yield_pct': yield_pct,
            'temp_C': temp_C,
            'orgLi_reagent': orgLi_name[:40],
        })

    print(f"  With yield+temp: {n_with_yield}")
    print(f"  ArX found: {n_arx_found}")
    print(f"  Valid ArLi inferred: {n_arli_valid}")
    return records


def deduplicate_and_label(records):
    """Deduplicate by canonical ArLi SMILES, assign stability label."""
    by_arli = defaultdict(list)
    for r in records:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(r['arli_smiles']))
        by_arli[canonical].append(r)

    results = []
    for smi, recs in by_arli.items():
        yields = [r['yield_pct'] for r in recs]
        temps = [r['temp_C'] for r in recs]

        # Focus on reactions at cryogenic temperatures (-78 to -50°C)
        cryo = [r for r in recs if -85 <= r['temp_C'] <= -50]
        if cryo:
            mean_yield = np.mean([r['yield_pct'] for r in cryo])
            ref_temp = np.mean([r['temp_C'] for r in cryo])
        else:
            mean_yield = np.mean(yields)
            ref_temp = np.mean(temps)

        # Stability label
        if mean_yield >= 80:
            label = "batch_stable"
        elif mean_yield >= 30:
            label = "batch_moderate"
        else:
            label = "batch_unstable"

        mol = Chem.MolFromSmiles(smi)
        results.append({
            'arli_smiles': smi,
            'arx_smiles': recs[0]['arx_smiles'],
            'n_reactions': len(recs),
            'n_cryo': len(cryo),
            'mean_yield': mean_yield,
            'ref_temp_C': ref_temp,
            'stability_label': label,
            'n_heavy_atoms': mol.GetNumHeavyAtoms() if mol else 0,
        })

    return results


def compute_all_descriptors(intermediates):
    """Compute molecular descriptors for all intermediates."""
    for inter in intermediates:
        desc = compute_descriptors(inter['arli_smiles'])
        if desc:
            inter.update(desc)
    return intermediates


def load_flow_data():
    """Load the 11 flow intermediates with Ea."""
    rows = []
    with open(FLOW_ARRHENIUS) as f:
        for r in csv.DictReader(f):
            rows.append({
                'arli_smiles': r['intermediate_smiles'],
                'intermediate': r['intermediate'],
                'Ea': float(r['Ea_decomp_kJ_mol']),
                't_half_m40': float(r['t_half_m40C_s']),
                'source': 'flow',
            })
    return rows


def train_augmented_model(flow_data, ord_data):
    """Train augmented QSPR with flow (Ea) + ORD (binary label)."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Get common descriptor columns
    desc_keys = [k for k in ord_data[0].keys()
                 if k not in ('arli_smiles', 'arx_smiles', 'n_reactions', 'n_cryo',
                              'mean_yield', 'ref_temp_C', 'stability_label', 'n_heavy_atoms')]

    # Filter ORD data: only those with valid descriptors
    ord_valid = [d for d in ord_data if all(k in d for k in desc_keys[:5])]
    if not ord_valid:
        print("  No valid ORD descriptors!")
        return

    # Prepare ORD classification data
    X_ord = np.array([[d.get(k, 0) for k in desc_keys] for d in ord_valid])
    y_ord_label = [d['stability_label'] for d in ord_valid]
    y_ord_binary = np.array([1 if l == 'batch_stable' else 0 for l in y_ord_label])

    # Prepare flow regression data
    flow_with_desc = []
    for fd in flow_data:
        desc = compute_descriptors(fd['arli_smiles'])
        if desc:
            fd.update(desc)
            flow_with_desc.append(fd)

    X_flow = np.array([[d.get(k, 0) for k in desc_keys] for d in flow_with_desc])
    y_flow_ea = np.array([d['Ea'] for d in flow_with_desc])
    y_flow_logt = np.log10(np.array([d['t_half_m40'] for d in flow_with_desc]))

    # Convert flow Ea to binary labels for combined classification
    flow_labels = []
    for d in flow_with_desc:
        if d['t_half_m40'] > 60:
            flow_labels.append('batch_stable')
        elif d['t_half_m40'] > 1:
            flow_labels.append('batch_moderate')
        else:
            flow_labels.append('batch_unstable')
    y_flow_binary = np.array([1 if l == 'batch_stable' else 0 for l in flow_labels])

    # Remove zero-variance features
    variances = np.var(np.vstack([X_ord, X_flow]), axis=0)
    keep = variances > 1e-10
    X_ord_f = X_ord[:, keep]
    X_flow_f = X_flow[:, keep]
    desc_names_f = [k for k, kp in zip(desc_keys, keep) if kp]

    print(f"\n  ORD: {len(ord_valid)} intermediates ({sum(y_ord_binary)} stable, {len(y_ord_binary)-sum(y_ord_binary)} unstable)")
    print(f"  Flow: {len(flow_with_desc)} intermediates")
    print(f"  Features: {len(desc_names_f)}")

    # 1. Classification on ORD alone
    scaler = StandardScaler()
    X_ord_s = scaler.fit_transform(X_ord_f)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_features='sqrt')
    rf.fit(X_ord_s, y_ord_binary)

    # Predict flow intermediates
    X_flow_s = scaler.transform(X_flow_f)
    flow_pred = rf.predict(X_flow_s)
    flow_prob = rf.predict_proba(X_flow_s)[:, 1]

    print(f"\n  === ORD-trained classifier → Flow predictions ===")
    print(f"  {'Intermediate':<45s} {'Actual':>12s} {'Predicted':>10s} {'P(stable)':>10s}")
    for i, d in enumerate(flow_with_desc):
        actual = flow_labels[i]
        pred = 'stable' if flow_pred[i] else 'unstable'
        print(f"  {d['intermediate'][:45]:<45s} {actual:>12s} {pred:>10s} {flow_prob[i]:>10.2f}")

    correct = sum(1 for i in range(len(flow_pred)) if flow_pred[i] == y_flow_binary[i])
    print(f"\n  Flow accuracy: {correct}/{len(flow_pred)} ({correct/len(flow_pred)*100:.0f}%)")

    # 2. Combined classification (ORD + flow)
    X_all = np.vstack([X_ord_f, X_flow_f])
    y_all = np.concatenate([y_ord_binary, y_flow_binary])
    scaler2 = StandardScaler()
    X_all_s = scaler2.fit_transform(X_all)

    rf2 = RandomForestClassifier(n_estimators=200, random_state=42, max_features='sqrt')
    rf2.fit(X_all_s, y_all)

    # Feature importances from combined model
    importances = rf2.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  === Combined model (ORD+Flow) feature importances ===")
    for idx in sorted_idx[:10]:
        print(f"    {desc_names_f[idx]:30s}: {importances[idx]:.4f}")

    # 3. LOOCV on flow using ORD-pretrained features
    print(f"\n  === Transfer learning: ORD→Flow LOOCV for Ea ===")
    # Use top RF features from combined model to select features for Ea regression
    top_k = min(3, len(desc_names_f))
    top_idx = sorted_idx[:top_k]
    top_names = [desc_names_f[i] for i in top_idx]

    X_flow_top = X_flow_f[:, top_idx]
    n = len(y_flow_ea)
    y_pred_ea = np.zeros(n)
    for i in range(n):
        X_tr = np.delete(X_flow_top, i, axis=0)
        y_tr = np.delete(y_flow_ea, i)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_flow_top[i:i+1])
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr)
        y_pred_ea[i] = model.predict(X_te_s)[0]

    ss_res = np.sum((y_flow_ea - y_pred_ea) ** 2)
    ss_tot = np.sum((y_flow_ea - np.mean(y_flow_ea)) ** 2)
    q2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_flow_ea - y_pred_ea) ** 2))
    print(f"  Features used: {top_names}")
    print(f"  LOOCV Q² = {q2:.3f}, RMSE = {rmse:.1f} kJ/mol")

    return {
        'ord_n': len(ord_valid),
        'flow_n': len(flow_with_desc),
        'flow_accuracy': correct / len(flow_pred),
        'q2': q2,
        'rmse': rmse,
        'top_features': top_names,
        'flow_predictions': list(zip(
            [d['intermediate'] for d in flow_with_desc],
            y_flow_ea.tolist(),
            y_pred_ea.tolist(),
            flow_prob.tolist(),
        )),
        'ord_stable_frac': sum(y_ord_binary) / len(y_ord_binary),
        'importances': dict(zip(desc_names_f, importances.tolist())),
    }


def plot_results(results, ord_data, flow_data):
    """Generate analysis figures."""
    import matplotlib.pyplot as plt

    # Plot 1: ORD stability distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = Counter(d['stability_label'] for d in ord_data)
    colors_map = {'batch_stable': '#31A354', 'batch_moderate': '#FDB863', 'batch_unstable': '#E6550D'}
    ax1.bar(labels.keys(), labels.values(),
            color=[colors_map[k] for k in labels.keys()], edgecolor='white')
    ax1.set_ylabel('Number of unique ArLi intermediates', fontsize=11)
    ax1.set_title(f'(a) ORD Batch Stability Labels\n({sum(labels.values())} unique intermediates)', fontsize=12, fontweight='bold')
    for k, v in labels.items():
        ax1.text(k, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Transfer learning — predicted vs actual Ea
    preds = results['flow_predictions']
    names = [p[0][:25] for p in preds]
    actual = [p[1] for p in preds]
    predicted = [p[2] for p in preds]

    ax2.scatter(actual, predicted, s=100, c='#2171B5', edgecolors='navy', zorder=5)
    for i, name in enumerate(names):
        ax2.annotate(name, (actual[i], predicted[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=7)

    lo = min(min(actual), min(predicted)) - 5
    hi = max(max(actual), max(predicted)) + 5
    ax2.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax2.set_xlabel('Actual Ea (kJ/mol)', fontsize=11)
    ax2.set_ylabel('Predicted Ea (kJ/mol)', fontsize=11)
    ax2.set_title(f'(b) ORD-Augmented QSPR\nLOOCV Q² = {results["q2"]:.3f}', fontsize=12, fontweight='bold')
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'augmented_qspr_results.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot 3: P(stable) from ORD model vs actual t½
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    t_halves = [d['t_half_m40'] for d in flow_data if 'Ea' in d]
    probs = [p[3] for p in preds]

    ax3.scatter(np.log10(t_halves), probs, s=120, c='#2171B5', edgecolors='navy', zorder=5)
    for i, name in enumerate(names):
        ax3.annotate(name, (np.log10(t_halves[i]), probs[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=7)

    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5, label='t½ = 1s')
    ax3.set_xlabel('log₁₀(t½ at -40°C) [s]', fontsize=11)
    ax3.set_ylabel('P(batch_stable) from ORD model', fontsize=11)
    ax3.set_title('ORD Classifier: Predicted Stability vs Actual Half-life', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    fig2.tight_layout()
    path2 = os.path.join(PLOT_DIR, 'ord_prob_vs_halflife.png')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")


def main():
    # Step 1: Extract ORD intermediates
    records = extract_ord_intermediates()

    # Step 2: Deduplicate and label
    print("\nDeduplicating and labeling...")
    ord_intermediates = deduplicate_and_label(records)
    print(f"  Unique ArLi intermediates: {len(ord_intermediates)}")
    labels = Counter(d['stability_label'] for d in ord_intermediates)
    for label, cnt in labels.most_common():
        print(f"    {label}: {cnt}")

    # Step 3: Compute descriptors
    print("\nComputing molecular descriptors...")
    ord_intermediates = compute_all_descriptors(ord_intermediates)
    valid = [d for d in ord_intermediates if 'MW' in d]
    print(f"  Valid descriptors: {len(valid)}/{len(ord_intermediates)}")

    # Save ORD intermediates CSV
    csv_path = os.path.join(OUTPUT_DIR, 'ord_inferred_intermediates.csv')
    if valid:
        keys = ['arli_smiles', 'arx_smiles', 'n_reactions', 'n_cryo',
                'mean_yield', 'ref_temp_C', 'stability_label', 'n_heavy_atoms']
        desc_keys = [k for k in valid[0].keys() if k not in keys]
        all_keys = keys + sorted(desc_keys)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(valid)
        print(f"  Saved: {csv_path}")

    # Step 4: Load flow data
    flow_data = load_flow_data()
    print(f"\nFlow intermediates: {len(flow_data)}")

    # Step 5: Train augmented model
    print("\nTraining augmented QSPR model...")
    results = train_augmented_model(flow_data, valid)

    if results:
        # Step 6: Plot
        print("\nGenerating plots...")
        plot_results(results, valid, flow_data)

        # Summary
        print(f"\n{'='*65}")
        print(f"AUGMENTED QSPR SUMMARY")
        print(f"{'='*65}")
        print(f"ORD intermediates: {results['ord_n']} (stable: {results['ord_stable_frac']*100:.0f}%)")
        print(f"Flow intermediates: {results['flow_n']} (with Ea)")
        print(f"ORD classifier → Flow accuracy: {results['flow_accuracy']*100:.0f}%")
        print(f"Transfer LOOCV Q²: {results['q2']:.3f} (RMSE={results['rmse']:.1f} kJ/mol)")
        print(f"Top features: {results['top_features']}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
