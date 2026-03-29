"""
Quality evaluator for FlowFigTabMiner final output JSONs.
Scores each reaction record and produces data/robustness/quality_report.csv.

Scoring (per record, 0-100):
  +20  has reactant (SMILES or name)
  +20  has product (SMILES or name)
  +20  product SMILES is chemically valid (RDKit)
  +20  has yield or conversion (numeric)
  +20  has ≥1 flow condition (T, time, flow rate, solvent, catalyst, pressure)
"""
import os
import sys
import glob
import json
import csv
import argparse

sys.path.insert(0, os.getcwd())

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to GBK and choke on accented paper names.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "data/robustness"
FINAL_DIR = "data/final_output"
REPORT_CSV = os.path.join(OUTPUT_DIR, "quality_report.csv")

# Try to import RDKit for SMILES validation
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[Warning] RDKit not available, SMILES validity check disabled.")


def is_valid_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        return False
    if not HAS_RDKIT:
        return len(smiles) > 2  # fallback: non-empty
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def _get(record, *keys):
    """Safely get a nested value from record."""
    val = record
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def score_record(record):
    """Score a single reaction record. Returns (score, breakdown_dict)."""
    score = 0
    breakdown = {}

    # +20: has reactant
    has_reactant = bool(
        _get(record, "reactant_smiles") or
        _get(record, "reactant_name") or
        _get(record, "Reactants") or       # legacy schema
        _get(record, "intermediate") or
        _get(record, "electrophile_smiles")
    )
    breakdown["has_reactant"] = has_reactant
    if has_reactant:
        score += 20

    # +20: has product
    product_smiles = (
        _get(record, "product_smiles") or
        _get(record, "electrophile_smiles")  # some legacy records
    )
    has_product = bool(
        product_smiles or
        _get(record, "product_name") or
        _get(record, "Products")
    )
    breakdown["has_product"] = has_product
    if has_product:
        score += 20

    # +20: product SMILES valid
    smiles_valid = is_valid_smiles(product_smiles)
    breakdown["product_smiles_valid"] = smiles_valid
    if smiles_valid:
        score += 20

    # +20: has yield, conversion, OR selectivity (any numeric outcome metric)
    yield_val = (
        _get(record, "yield_pct") or
        _get(record, "yield_external_quenching_flow_isolated") or
        _get(record, "yield_external_quenching_flow") or
        _get(record, "conversion_pct") or
        _get(record, "selectivity_pct") or
        _get(record, "ee_pct") or
        _get(record, "er") or
        _get(record, "dr")
    )
    has_yield = yield_val is not None and yield_val != ""
    breakdown["has_yield"] = has_yield
    if has_yield:
        score += 20

    # +20: has ≥1 flow condition
    cond = _get(record, "conditions") or _get(record, "conditions_flow") or {}
    if not isinstance(cond, dict):
        cond = {}
    condition_fields = [
        cond.get("temperature_C"), cond.get("T_R1_degC"), cond.get("T_R2_degC"),
        cond.get("residence_time_s"), cond.get("t_R1_ms"), cond.get("t_R2_s"),
        cond.get("flow_rate_mL_min"), cond.get("flow_rate_mL_per_min"),
        cond.get("solvent"), cond.get("catalyst"), cond.get("pressure_bar"),
        _get(record, "solvent"), _get(record, "catalyst"),
    ]
    has_conditions = any(v is not None and v != "" for v in condition_fields)
    breakdown["has_conditions"] = has_conditions
    if has_conditions:
        score += 20

    return score, breakdown


def evaluate_json(json_path):
    """Evaluate a single final JSON file."""
    basename = os.path.splitext(os.path.basename(json_path))[0]
    basename = basename.replace("_final", "")

    try:
        with open(json_path, encoding="utf-8") as f:
            content = f.read().strip()
        data = json.loads(content)
    except Exception as e:
        return {
            "pdf": basename,
            "n_records": 0,
            "avg_score": 0,
            "pct_has_smiles": 0,
            "pct_has_yield": 0,
            "pct_has_conditions": 0,
            "low_score_examples": f"JSON parse error: {e}",
        }

    if isinstance(data, dict) and "dataset" in data:
        records = data["dataset"]
    elif isinstance(data, list):
        records = data
    else:
        records = []

    if not records:
        return {
            "pdf": basename,
            "n_records": 0,
            "avg_score": 0,
            "pct_has_smiles": 0,
            "pct_has_yield": 0,
            "pct_has_conditions": 0,
            "low_score_examples": "No records extracted",
        }

    scores = []
    breakdowns = []
    for r in records:
        s, bd = score_record(r)
        scores.append(s)
        breakdowns.append(bd)

    avg = sum(scores) / len(scores)
    pct_smiles = sum(1 for bd in breakdowns if bd.get("has_product")) / len(breakdowns) * 100
    pct_yield = sum(1 for bd in breakdowns if bd.get("has_yield")) / len(breakdowns) * 100
    pct_cond = sum(1 for bd in breakdowns if bd.get("has_conditions")) / len(breakdowns) * 100

    # Collect low-score examples (score < 60) for manual review
    low_examples = []
    for i, (r, s) in enumerate(zip(records, scores)):
        if s < 60:
            snippet = json.dumps(r, ensure_ascii=False)[:300]
            low_examples.append(f"[score={s}] {snippet}")
    low_str = " || ".join(low_examples[:3])  # max 3 examples in CSV

    return {
        "pdf": basename,
        "n_records": len(records),
        "avg_score": round(avg, 1),
        "pct_has_smiles": round(pct_smiles, 1),
        "pct_has_yield": round(pct_yield, 1),
        "pct_has_conditions": round(pct_cond, 1),
        "low_score_examples": low_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlowFigTabMiner output quality")
    parser.add_argument("--dir", default=FINAL_DIR, help="Directory with *_final.json files")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.dir, "*_final.json")))
    if not json_files:
        print(f"No *_final.json files found in {args.dir}")
        return

    print(f"Evaluating {len(json_files)} output files...")
    rows = []
    for jf in json_files:
        row = evaluate_json(jf)
        rows.append(row)
        print(f"  {row['pdf']:20s} | records={row['n_records']:3d} | avg_score={row['avg_score']:5.1f} "
              f"| smiles={row['pct_has_smiles']:5.1f}% | yield={row['pct_has_yield']:5.1f}% "
              f"| cond={row['pct_has_conditions']:5.1f}%")

    fieldnames = ["pdf", "n_records", "avg_score", "pct_has_smiles", "pct_has_yield",
                  "pct_has_conditions", "low_score_examples"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nQuality report saved to: {REPORT_CSV}")

    if rows:
        overall_avg = sum(r["avg_score"] * r["n_records"] for r in rows if r["n_records"] > 0)
        total_records = sum(r["n_records"] for r in rows)
        if total_records > 0:
            print(f"Overall weighted avg score: {overall_avg/total_records:.1f}/100 "
                  f"({total_records} total records across {len(rows)} PDFs)")


if __name__ == "__main__":
    main()
