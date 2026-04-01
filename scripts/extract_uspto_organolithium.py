#!/usr/bin/env python3
"""Extract rich organolithium reaction records from the local USPTO-LLM datasets."""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "other dataset"

SINGLE_STEP_CSV = DATASET_ROOT / "single_step" / "transformer" / "uspto_llm.csv"
MULTI_STEP_CSV = DATASET_ROOT / "multi_step" / "uspto_multiple_step.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "input" / "uspto_organolithium_extracted_rich.csv"
SUMMARY_JSON = (
    PROJECT_ROOT / "data" / "input" / "uspto_organolithium_extracted_rich_summary.json"
)

NAME_KEYWORD_PATTERN = re.compile(
    r"(?ix)"
    r"("
    r"\b(?:n|sec|s|tert|t)[-\s]?butyllithium\b|"
    r"\bmethyllithium\b|"
    r"\bethyllithium\b|"
    r"\bphenyllithium\b|"
    r"\bvinyllithium\b|"
    r"\baryllithium\b|"
    r"\balkyllithium\b|"
    r"\borganolithium\b|"
    r"\bdilithio[\w-]*\b|"
    r"\b[\w-]*lithio[\w-]*\b"
    r")"
)

OUTPUT_COLUMNS = [
    "reaction_id",
    "data_source",
    "source_file",
    "reaction_smiles_raw",
    "reactants_smiles",
    "agents_smiles",
    "products_smiles",
    "reactant_count",
    "product_count",
    "reactant_fragments_json",
    "product_fragments_json",
    "agent_tokens_json",
    "reactant1_smiles",
    "reactant2_smiles",
    "reactant3_smiles",
    "reactant4_smiles",
    "reactant5_smiles",
    "product1_smiles",
    "product2_smiles",
    "product3_smiles",
    "solvent",
    "solvent_tokens_json",
    "catalyst",
    "catalyst_tokens_json",
    "temperature_raw",
    "temperature_tokens_json",
    "temperature_C",
    "reaction_time_raw",
    "reaction_time_tokens_json",
    "reaction_time_s",
    "reaction_class",
    "raw_class",
    "raw_solvents",
    "raw_catalyst",
    "raw_temperature",
    "raw_time",
    "source_row_json",
    "is_organolithium",
    "match_reason",
    "matched_fragment",
]


def parse_list_field(raw: str) -> list[str]:
    if not raw or raw.strip() in {"", "[]"}:
        return []
    try:
        value = ast.literal_eval(raw.strip())
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
    except Exception:
        return [raw.strip()]


def parse_temperature(values: list[str]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    raw = "; ".join(values)
    for value in values:
        text = value.strip().lower()
        if text in {"room temperature", "rt", "ambient"}:
            return raw, 25.0
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            number = float(match.group())
            if number > 200:
                number -= 273.15
            return raw, round(number, 1)
    return raw, None


def parse_time(values: list[str]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    raw = "; ".join(values)
    for value in values:
        text = value.strip().lower()
        if text in {"overnight", "overnight (12h)", "o/n"}:
            return raw, 43200.0
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match:
            return raw, float(match.group())
    return raw, None


def split_smiles_components(smiles: str) -> list[str]:
    if not smiles:
        return []
    parts = []
    bracket_depth = 0
    paren_depth = 0
    start = 0
    for index, char in enumerate(smiles):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "." and bracket_depth == 0 and paren_depth == 0:
            parts.append(smiles[start:index])
            start = index + 1
    parts.append(smiles[start:])
    return [part for part in parts if part]


def is_organolithium_smiles(smiles: str) -> bool:
    if "Li" not in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    lithium_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 3]
    if not lithium_atoms:
        return False
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if not carbon_atoms:
        return False
    for atom in lithium_atoms:
        if any(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()):
            return True
    has_positive_lithium = any(atom.GetFormalCharge() > 0 for atom in lithium_atoms)
    has_negative_carbanion = any(atom.GetFormalCharge() < 0 for atom in carbon_atoms)
    return has_positive_lithium and has_negative_carbanion


def detect_organolithium(reactants_smiles: str, extra_texts: list[str]) -> tuple[bool, str, str]:
    for fragment in split_smiles_components(reactants_smiles):
        if is_organolithium_smiles(fragment):
            return True, "smiles_structure", fragment
    for text in extra_texts:
        if not text:
            continue
        match = NAME_KEYWORD_PATTERN.search(text)
        if match:
            return True, "name_keyword", match.group(0)
    return False, "", ""


def parse_multi_step_agents(agents_smiles: str) -> dict:
    tokens = [token.strip() for token in agents_smiles.split(".") if token.strip()] if agents_smiles else []
    solvents = []
    temperature_tokens = []
    time_tokens = []
    numeric_tokens = []

    for token in tokens:
        lowered = token.lower()
        if lowered in {"room temperature", "rt", "ambient"}:
            temperature_tokens.append(token)
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            numeric_tokens.append(token)
        else:
            solvents.append(token)

    if len(numeric_tokens) >= 2:
        temperature_tokens.extend(numeric_tokens[:-1])
        time_tokens.extend(numeric_tokens[-1:])
    elif len(numeric_tokens) == 1:
        value = float(numeric_tokens[0])
        if temperature_tokens:
            time_tokens.extend(numeric_tokens)
        elif abs(value) <= 150 and not solvents:
            temperature_tokens.extend(numeric_tokens)
        elif abs(value) <= 150 and solvents and value < 0:
            temperature_tokens.extend(numeric_tokens)
        else:
            time_tokens.extend(numeric_tokens)

    temp_raw, temp_c = parse_temperature(temperature_tokens)
    time_raw, time_s = parse_time(time_tokens)

    return {
        "agent_tokens": tokens,
        "solvents": solvents,
        "temperature_tokens": temperature_tokens,
        "time_tokens": time_tokens,
        "temperature_raw": temp_raw,
        "temperature_C": temp_c,
        "reaction_time_raw": time_raw,
        "reaction_time_s": time_s,
    }


def base_record() -> dict:
    return {column: None for column in OUTPUT_COLUMNS}


def populate_component_slots(
    record: dict,
    count_column: str,
    json_column: str,
    slot_prefix: str,
    values: list[str],
    max_slots: int,
) -> None:
    record[count_column] = len(values)
    record[json_column] = json.dumps(values, ensure_ascii=False)
    for index in range(max_slots):
        column = f"{slot_prefix}{index + 1}_smiles"
        record[column] = values[index] if index < len(values) else None


def build_single_step_record(row: dict) -> dict | None:
    reaction_smiles = row.get("rs>>ps", "")
    parts = reaction_smiles.split(">>")
    reactants_smiles = parts[0] if len(parts) > 0 else ""
    products_smiles = parts[1] if len(parts) > 1 else ""

    solvents = parse_list_field(row.get("solvents", ""))
    catalysts = parse_list_field(row.get("catalyst", ""))
    temperatures = parse_list_field(row.get("temperature", ""))
    times = parse_list_field(row.get("time", ""))

    matched, match_reason, matched_fragment = detect_organolithium(
        reactants_smiles,
        solvents + catalysts + temperatures + times,
    )
    if not matched:
        return None

    reactant_list = split_smiles_components(reactants_smiles)
    product_list = split_smiles_components(products_smiles)
    temperature_raw, temperature_c = parse_temperature(temperatures)
    reaction_time_raw, reaction_time_s = parse_time(times)

    record = base_record()
    record.update(
        {
            "reaction_id": row.get("id", ""),
            "data_source": "USPTO-LLM",
            "source_file": "single_step",
            "reaction_smiles_raw": reaction_smiles,
            "reactants_smiles": reactants_smiles,
            "agents_smiles": None,
            "products_smiles": products_smiles,
            "product_count": len(product_list),
            "product_fragments_json": json.dumps(product_list, ensure_ascii=False),
            "agent_tokens_json": None,
            "product1_smiles": product_list[0] if len(product_list) > 0 else None,
            "product2_smiles": product_list[1] if len(product_list) > 1 else None,
            "product3_smiles": product_list[2] if len(product_list) > 2 else None,
            "solvent": "; ".join(solvents) if solvents else None,
            "solvent_tokens_json": json.dumps(solvents, ensure_ascii=False),
            "catalyst": "; ".join(catalysts) if catalysts else None,
            "catalyst_tokens_json": json.dumps(catalysts, ensure_ascii=False),
            "temperature_raw": temperature_raw,
            "temperature_tokens_json": json.dumps(temperatures, ensure_ascii=False),
            "temperature_C": temperature_c,
            "reaction_time_raw": reaction_time_raw,
            "reaction_time_tokens_json": json.dumps(times, ensure_ascii=False),
            "reaction_time_s": reaction_time_s,
            "reaction_class": row.get("class", ""),
            "raw_class": row.get("class", ""),
            "raw_solvents": row.get("solvents", ""),
            "raw_catalyst": row.get("catalyst", ""),
            "raw_temperature": row.get("temperature", ""),
            "raw_time": row.get("time", ""),
            "source_row_json": json.dumps(row, ensure_ascii=False),
            "is_organolithium": True,
            "match_reason": match_reason,
            "matched_fragment": matched_fragment,
        }
    )
    populate_component_slots(
        record,
        "reactant_count",
        "reactant_fragments_json",
        "reactant",
        reactant_list,
        5,
    )
    return record


def build_multi_step_record(row: dict) -> dict | None:
    reaction_smiles = row.get("reactants>>products", "")
    parts = reaction_smiles.split(">")
    reactants_smiles = parts[0] if len(parts) > 0 else ""
    agents_smiles = parts[1] if len(parts) > 2 else ""
    products_smiles = parts[2] if len(parts) > 2 else (parts[1] if len(parts) == 2 else "")

    matched, match_reason, matched_fragment = detect_organolithium(
        reactants_smiles,
        [agents_smiles],
    )
    if not matched:
        return None

    reactant_list = split_smiles_components(reactants_smiles)
    product_list = split_smiles_components(products_smiles)
    agent_info = parse_multi_step_agents(agents_smiles)

    record = base_record()
    record.update(
        {
            "reaction_id": row.get("reaction_id", ""),
            "data_source": "USPTO-LLM",
            "source_file": "multi_step",
            "reaction_smiles_raw": reaction_smiles,
            "reactants_smiles": reactants_smiles,
            "agents_smiles": agents_smiles,
            "products_smiles": products_smiles,
            "product_count": len(product_list),
            "product_fragments_json": json.dumps(product_list, ensure_ascii=False),
            "agent_tokens_json": json.dumps(agent_info["agent_tokens"], ensure_ascii=False),
            "product1_smiles": product_list[0] if len(product_list) > 0 else None,
            "product2_smiles": product_list[1] if len(product_list) > 1 else None,
            "product3_smiles": product_list[2] if len(product_list) > 2 else None,
            "solvent": "; ".join(agent_info["solvents"]) if agent_info["solvents"] else None,
            "solvent_tokens_json": json.dumps(agent_info["solvents"], ensure_ascii=False),
            "catalyst": None,
            "catalyst_tokens_json": json.dumps([], ensure_ascii=False),
            "temperature_raw": agent_info["temperature_raw"],
            "temperature_tokens_json": json.dumps(agent_info["temperature_tokens"], ensure_ascii=False),
            "temperature_C": agent_info["temperature_C"],
            "reaction_time_raw": agent_info["reaction_time_raw"],
            "reaction_time_tokens_json": json.dumps(agent_info["time_tokens"], ensure_ascii=False),
            "reaction_time_s": agent_info["reaction_time_s"],
            "reaction_class": None,
            "raw_class": None,
            "raw_solvents": None,
            "raw_catalyst": None,
            "raw_temperature": None,
            "raw_time": None,
            "source_row_json": json.dumps(row, ensure_ascii=False),
            "is_organolithium": True,
            "match_reason": match_reason,
            "matched_fragment": matched_fragment,
        }
    )
    populate_component_slots(
        record,
        "reactant_count",
        "reactant_fragments_json",
        "reactant",
        reactant_list,
        5,
    )
    return record


def extract_rows() -> tuple[list[dict], dict, dict]:
    single_rows = []
    multi_rows = []
    single_scanned = 0
    multi_scanned = 0

    with SINGLE_STEP_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            single_scanned += 1
            record = build_single_step_record(row)
            if record is not None:
                single_rows.append(record)

    with MULTI_STEP_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            multi_scanned += 1
            record = build_multi_step_record(row)
            if record is not None:
                multi_rows.append(record)

    return (
        single_rows + multi_rows,
        {"rows_scanned": single_scanned, "rows_matched": len(single_rows)},
        {"rows_scanned": multi_scanned, "rows_matched": len(multi_rows)},
    )


def build_summary(rows: list[dict], single_stats: dict, multi_stats: dict) -> dict:
    by_source = Counter(row["source_file"] for row in rows)
    by_reason = Counter(row["match_reason"] for row in rows)
    top_fragments = Counter(row["matched_fragment"] for row in rows if row["matched_fragment"])
    duplicate_ids = len(rows) - len({row["reaction_id"] for row in rows})
    return {
        "inputs": {
            "single_step_csv": str(SINGLE_STEP_CSV),
            "multi_step_csv": str(MULTI_STEP_CSV),
        },
        "outputs": {
            "csv": str(OUTPUT_CSV),
            "summary_json": str(SUMMARY_JSON),
        },
        "single_step": single_stats,
        "multi_step": multi_stats,
        "combined": {
            "rows_matched": len(rows),
            "unique_reaction_ids": len({row["reaction_id"] for row in rows}),
            "duplicate_id_rows": duplicate_ids,
            "by_source": dict(by_source),
            "by_match_reason": dict(by_reason),
            "top_matched_fragments": dict(top_fragments.most_common(20)),
            "has_temperature": sum(row["temperature_C"] is not None for row in rows),
            "has_time": sum(row["reaction_time_s"] is not None for row in rows),
            "has_solvent": sum(bool(row["solvent"]) for row in rows),
            "has_catalyst": sum(bool(row["catalyst"]) for row in rows),
        },
    }


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    print("Extracting rich organolithium records from USPTO-LLM...\n")

    rows, single_stats, multi_stats = extract_rows()
    summary = build_summary(rows, single_stats, multi_stats)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"single_step matched: {single_stats['rows_matched']} / {single_stats['rows_scanned']}")
    print(f"multi_step matched : {multi_stats['rows_matched']} / {multi_stats['rows_scanned']}")
    print(f"total matched      : {len(rows)}")
    print(f"saved csv          : {OUTPUT_CSV}")
    print(f"saved summary      : {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
