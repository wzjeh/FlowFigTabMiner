import csv
import glob
import json
import os
import re
from collections import Counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_DIR = os.path.join(ROOT, "data", "final_output")
OUT_DIR = os.path.join(ROOT, "downstream", "lifetime_prediction", "data")
OUT_JSONL = os.path.join(OUT_DIR, "intermediate_lifetime_candidates.jsonl")
OUT_CSV = os.path.join(OUT_DIR, "intermediate_lifetime_candidates.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "dataset_summary.json")

IGNORE_FILES = {
    "example_normalized.json",
    "example1_normalized.json",
    "80_normalized.json",
    "zz_codex_smoke_80_normalized.json",
    "zz_codex_smoke_80_fix_normalized.json",
    "zz_codex_smoke_80_finalcheck_normalized.json",
}


def _norm_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def _batch_or_flow(record):
    reactor_type = _norm_text((record.get("conditions") or {}).get("reactor_type")).lower()
    notes = _norm_text(record.get("notes")).lower()
    text = " ".join(x for x in (reactor_type, notes) if x)
    if "batch" in text or "flask" in text or "canula" in text:
        return "batch"
    if any(k in text for k in ("flow", "microreactor", "capillary", "coil", "t-mixer", "continuous")):
        return "flow"
    return "unknown"


def _find_named_intermediate(record):
    fields = [
        record.get("reactant1_name"),
        record.get("reactant2_name"),
        record.get("product_name"),
        record.get("product_label"),
        record.get("notes"),
    ]
    joined = " || ".join(_norm_text(v) for v in fields if v)
    patterns = [
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*aryllithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*heteroaryllithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*alkyllithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*vinyllithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*oxiranyllithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*ynolate[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*carbamoyl anion[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*organolithium[A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
        r"\b([A-Za-z0-9,\-\(\)\[\]\/ ]*lithiated [A-Za-z0-9,\-\(\)\[\]\/ ]*)\b",
    ]
    for pat in patterns:
        m = re.search(pat, joined, re.I)
        if m:
            return _norm_text(m.group(1))
    return None


def _classify_intermediate(record):
    text = " ".join(
        _norm_text(x).lower()
        for x in [
            record.get("reactant1_name"),
            record.get("reactant2_name"),
            record.get("product_name"),
            record.get("product_label"),
            record.get("notes"),
            record.get("reaction_class"),
        ]
        if x
    )
    rules = [
        ("heteroaryllithium", r"heteroaryllith"),
        ("aryllithium", r"aryllith"),
        ("vinyllithium", r"vinyllith"),
        ("alkyllithium", r"alkyllith"),
        ("oxiranyllithium", r"oxiranyllith"),
        ("ynolate", r"\bynolate\b|lithium ynolate"),
        ("carbamoyl anion", r"carbamoyl anion|carbamoyllith"),
        ("organolithium_generic", r"organolith|lithiated"),
    ]
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    if "lithium" in text:
        return "organolithium_generic"
    return None


def _infer_generation_mode(record):
    text = " ".join(
        _norm_text(x).lower()
        for x in [
            record.get("reaction_class"),
            record.get("notes"),
            record.get("source_table_or_figure"),
        ]
        if x
    )
    if "halogen-metal exchange" in text or "halogen-lithium exchange" in text or "br - li exchange" in text:
        return "halogen_lithium_exchange"
    if "directed metalation" in text or "deprotonation" in text:
        return "deprotonation"
    if "reductive lithiation" in text:
        return "reductive_lithiation"
    if "ynolate" in text:
        return "ynolate_generation"
    if "carbamoyl" in text:
        return "carbamoyl_anion_generation"
    return None


def _extract_stability_note(record):
    notes = _norm_text(record.get("notes"))
    if not notes:
        return None
    keywords = [
        "unstable",
        "stable",
        "decompose",
        "decomposition",
        "racemization",
        "racemize",
        "must be trapped",
        "trapped immediately",
        "residence-time control",
        "flash chemistry",
        "outpace",
        "short-lived",
    ]
    low = notes.lower()
    if any(k in low for k in keywords):
        return notes
    return None


def _infer_stability_class(record):
    text = " ".join(
        _norm_text(x).lower()
        for x in [record.get("notes"), record.get("reaction_class"), record.get("source_table_or_figure")]
        if x
    )
    if any(k in text for k in ("must be trapped", "trapped immediately", "outpace", "milliseconds", "ms)")):
        return "must_trap_immediately"
    if any(k in text for k in ("highly unstable", "extremely unstable", "short-lived", "decompose")):
        return "highly_unstable"
    if any(k in text for k in ("sensitive", "racemization", "residence-time control", "unstable intermediate")):
        return "moderately_sensitive"
    if "stable" in text:
        return "stable"
    return "unknown"


def _infer_lifetime_bucket(record):
    text = _norm_text(record.get("notes")).lower()
    if not text:
        return "unknown"
    if re.search(r"\b\d+(\.\d+)?\s*ms\b", text):
        return "<1 s"
    s_matches = [float(m.group(1)) for m in re.finditer(r"\b(\d+(\.\d+)?)\s*s\b", text)]
    if s_matches:
        val = min(s_matches)
        if val < 1:
            return "<1 s"
        if val <= 10:
            return "1-10 s"
        if val <= 300:
            return "10-300 s"
        return ">300 s"
    min_matches = [float(m.group(1)) for m in re.finditer(r"\b(\d+(\.\d+)?)\s*min\b", text)]
    if min_matches:
        val = min(min_matches) * 60
        if val <= 300:
            return "10-300 s"
        return ">300 s"
    if any(k in text for k in ("unstable", "stable", "decompose", "short-lived", "residence-time control", "must be trapped")):
        return "qualitative_only"
    return "unknown"


def _record_to_candidate(record):
    conds = record.get("conditions") or {}
    candidate = {
        "paper_basename": record.get("paper_basename"),
        "paper_doi": record.get("paper_doi"),
        "paper_year": record.get("paper_year"),
        "source_table_or_figure": record.get("source_table_or_figure"),
        "reaction_class": record.get("reaction_class"),
        "ml_tier": record.get("ml_tier"),
        "ml_exclusion_reason": record.get("ml_exclusion_reason"),
        "structure_resolution_status": record.get("structure_resolution_status"),
        "structure_resolution_source": record.get("structure_resolution_source"),
        "reactant1_name": record.get("reactant1_name"),
        "reactant1_smiles": record.get("reactant1_smiles"),
        "reactant2_name": record.get("reactant2_name"),
        "reactant2_smiles": record.get("reactant2_smiles"),
        "product_name": record.get("product_name"),
        "product_smiles": record.get("product_smiles"),
        "temperature_C": conds.get("temperature_C"),
        "residence_time_s": conds.get("residence_time_s"),
        "flow_rate_mL_min": conds.get("flow_rate_mL_min"),
        "flow_rate_stream1_mL_min": conds.get("flow_rate_stream1_mL_min"),
        "flow_rate_stream2_mL_min": conds.get("flow_rate_stream2_mL_min"),
        "solvent": conds.get("solvent"),
        "additive": conds.get("additive"),
        "catalyst": conds.get("catalyst"),
        "pressure_bar": conds.get("pressure_bar"),
        "reactor_type": conds.get("reactor_type"),
        "batch_or_flow": _batch_or_flow(record),
        "notes": record.get("notes"),
        "yield_pct": record.get("yield_pct"),
        "conversion_pct": record.get("conversion_pct"),
        "ee_pct": record.get("ee_pct"),
    }
    candidate["intermediate_name"] = _find_named_intermediate(record)
    candidate["intermediate_class"] = _classify_intermediate(record)
    candidate["generation_mode"] = _infer_generation_mode(record)
    candidate["stability_note"] = _extract_stability_note(record)
    candidate["stability_class"] = _infer_stability_class(record)
    candidate["lifetime_bucket"] = _infer_lifetime_bucket(record)
    candidate["lifetime_value_s"] = None
    candidate["lifetime_confidence"] = "low" if candidate["lifetime_bucket"] == "qualitative_only" else ("medium" if candidate["lifetime_bucket"] != "unknown" else "unknown")
    return candidate


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    candidates = []
    tier_counter = Counter()
    stability_counter = Counter()
    bucket_counter = Counter()
    intermediate_counter = Counter()
    source_counter = Counter()

    for fp in sorted(glob.glob(os.path.join(FINAL_DIR, "*_normalized.json"))):
        bn = os.path.basename(fp)
        if bn in IGNORE_FILES or bn.startswith("._"):
            continue
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            candidate = _record_to_candidate(record)
            candidates.append(candidate)
            tier_counter[candidate["ml_tier"] or "missing"] += 1
            stability_counter[candidate["stability_class"]] += 1
            bucket_counter[candidate["lifetime_bucket"]] += 1
            intermediate_counter[candidate["intermediate_class"] or "unknown"] += 1
            source_counter[candidate["batch_or_flow"]] += 1

    fieldnames = sorted({k for row in candidates for k in row.keys()})
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    summary = {
        "total_candidates": len(candidates),
        "ml_tier_distribution": dict(tier_counter),
        "stability_class_distribution": dict(stability_counter),
        "lifetime_bucket_distribution": dict(bucket_counter),
        "intermediate_class_distribution": dict(intermediate_counter),
        "batch_or_flow_distribution": dict(source_counter),
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(OUT_JSONL)
    print(OUT_CSV)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
