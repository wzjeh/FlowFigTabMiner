#!/usr/bin/env python3
"""Extract rich organolithium reaction records from the local ORD clone."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2, reaction_pb2
from rdkit import Chem
from rdkit import RDLogger


DATA_DIR = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner/other dataset/ord/ord-data/data")
OUTPUT_DIR = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner/other dataset/ord/organolithium")
DETAIL_CSV = OUTPUT_DIR / "ord_organolithium_reactions_rich.csv"
SUMMARY_JSON = OUTPUT_DIR / "ord_organolithium_reactions_rich_summary.json"

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
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

COMPOUND_IDENTIFIER_ENUM = (
    reaction_pb2.CompoundIdentifier.DESCRIPTOR.enum_types_by_name["IdentifierType"]
)
REACTION_IDENTIFIER_ENUM = (
    reaction_pb2.ReactionIdentifier.DESCRIPTOR.enum_types_by_name["IdentifierType"]
)
REACTION_ROLE_ENUM = (
    reaction_pb2.Compound.DESCRIPTOR.fields_by_name["reaction_role"].enum_type
)

OUTPUT_COLUMNS = [
    "doi",
    "patent",
    "publication_url",
    "dataset_id",
    "dataset_name",
    "dataset_description",
    "dataset_file",
    "reaction_id",
    "reaction_smiles",
    "reaction_identifiers_json",
    "match_count",
    "matched_components_json",
    "has_setup",
    "has_conditions",
    "has_notes",
    "has_observations",
    "has_workups",
    "has_outcomes",
    "conditions_json",
    "setup_json",
    "notes_json",
    "observations_json",
    "workups_json",
    "outcomes_json",
    "provenance_json",
    "inputs_json",
    "reaction_json",
]


def enum_name(enum_descriptor, value: int) -> str:
    item = enum_descriptor.values_by_number.get(value)
    return item.name if item is not None else f"UNKNOWN_{value}"


def extract_doi(*texts: str) -> str:
    for text in texts:
        match = DOI_PATTERN.search(text or "")
        if match:
            return match.group(0)
    return ""


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


def message_to_json(message) -> str:
    return json.dumps(MessageToDict(message, preserving_proto_field_name=True), ensure_ascii=False)


def extract_reaction_smiles(reaction) -> tuple[str, str]:
    identifiers = []
    reaction_smiles = ""
    for identifier in reaction.identifiers:
        identifier_type = enum_name(REACTION_IDENTIFIER_ENUM, identifier.type)
        entry = {
            "type": identifier_type,
            "value": identifier.value,
            "details": identifier.details,
        }
        identifiers.append(entry)
        if not reaction_smiles and identifier_type in {"REACTION_SMILES", "REACTION_CXSMILES"}:
            reaction_smiles = identifier.value
    return reaction_smiles, json.dumps(identifiers, ensure_ascii=False)


def scan_component_identifiers(identifiers, scope: str, container_name: str, component_index: int, reaction_role: str) -> list[dict]:
    matches = []
    for identifier in identifiers:
        value = identifier.value.strip()
        if not value:
            continue
        identifier_type = enum_name(COMPOUND_IDENTIFIER_ENUM, identifier.type)
        if identifier_type in {"SMILES", "CXSMILES"} and is_organolithium_smiles(value):
            matches.append(
                {
                    "scope": scope,
                    "container_name": container_name,
                    "component_index": component_index,
                    "reaction_role": reaction_role,
                    "match_reason": "smiles_structure",
                    "identifier_type": identifier_type,
                    "matched_value": value,
                }
            )
        elif identifier_type in {"NAME", "IUPAC_NAME"} and NAME_KEYWORD_PATTERN.search(value):
            matches.append(
                {
                    "scope": scope,
                    "container_name": container_name,
                    "component_index": component_index,
                    "reaction_role": reaction_role,
                    "match_reason": "name_keyword",
                    "identifier_type": identifier_type,
                    "matched_value": value,
                }
            )
    return matches


def collect_matches(reaction) -> list[dict]:
    matches = []
    for container_name, reaction_input in reaction.inputs.items():
        for index, component in enumerate(reaction_input.components):
            matches.extend(
                scan_component_identifiers(
                    component.identifiers,
                    "input",
                    container_name,
                    index,
                    enum_name(REACTION_ROLE_ENUM, component.reaction_role),
                )
            )

    for outcome_index, outcome in enumerate(reaction.outcomes):
        for product_index, product in enumerate(outcome.products):
            matches.extend(
                scan_component_identifiers(
                    product.identifiers,
                    f"outcome_{outcome_index}",
                    f"product_{product_index}",
                    product_index,
                    "PRODUCT",
                )
            )
    return matches


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_reactions = 0
    matched_reactions = 0
    by_doi = Counter()
    field_coverage = defaultdict(int)

    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for dataset_file in sorted(DATA_DIR.rglob("*.pb.gz")):
            total_files += 1
            dataset = load_message(str(dataset_file), dataset_pb2.Dataset)
            dataset_doi = extract_doi(dataset.name, dataset.description)

            for reaction in dataset.reactions:
                total_reactions += 1
                matches = collect_matches(reaction)
                if not matches:
                    continue

                matched_reactions += 1
                reaction_doi = reaction.provenance.doi or dataset_doi
                by_doi[reaction_doi or "NO_DOI"] += 1

                reaction_smiles, reaction_identifiers_json = extract_reaction_smiles(reaction)
                record = {
                    "doi": reaction_doi,
                    "patent": reaction.provenance.patent,
                    "publication_url": reaction.provenance.publication_url,
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.name,
                    "dataset_description": dataset.description,
                    "dataset_file": str(dataset_file),
                    "reaction_id": reaction.reaction_id,
                    "reaction_smiles": reaction_smiles,
                    "reaction_identifiers_json": reaction_identifiers_json,
                    "match_count": len(matches),
                    "matched_components_json": json.dumps(matches, ensure_ascii=False),
                    "has_setup": reaction.HasField("setup"),
                    "has_conditions": reaction.HasField("conditions"),
                    "has_notes": reaction.HasField("notes"),
                    "has_observations": len(reaction.observations) > 0,
                    "has_workups": len(reaction.workups) > 0,
                    "has_outcomes": len(reaction.outcomes) > 0,
                    "conditions_json": message_to_json(reaction.conditions) if reaction.HasField("conditions") else "",
                    "setup_json": message_to_json(reaction.setup) if reaction.HasField("setup") else "",
                    "notes_json": message_to_json(reaction.notes) if reaction.HasField("notes") else "",
                    "observations_json": json.dumps(
                        [MessageToDict(item, preserving_proto_field_name=True) for item in reaction.observations],
                        ensure_ascii=False,
                    ),
                    "workups_json": json.dumps(
                        [MessageToDict(item, preserving_proto_field_name=True) for item in reaction.workups],
                        ensure_ascii=False,
                    ),
                    "outcomes_json": json.dumps(
                        [MessageToDict(item, preserving_proto_field_name=True) for item in reaction.outcomes],
                        ensure_ascii=False,
                    ),
                    "provenance_json": message_to_json(reaction.provenance) if reaction.HasField("provenance") else "",
                    "inputs_json": json.dumps(
                        {
                            name: MessageToDict(value, preserving_proto_field_name=True)
                            for name, value in reaction.inputs.items()
                        },
                        ensure_ascii=False,
                    ),
                    "reaction_json": message_to_json(reaction),
                }
                writer.writerow(record)

                for field in ["has_setup", "has_conditions", "has_notes", "has_observations", "has_workups", "has_outcomes"]:
                    if record[field]:
                        field_coverage[field] += 1

    summary = {
        "input_data_dir": str(DATA_DIR),
        "detail_csv": str(DETAIL_CSV),
        "total_files_scanned": total_files,
        "total_reactions_scanned": total_reactions,
        "matched_reactions": matched_reactions,
        "by_doi": dict(by_doi),
        "field_coverage": dict(field_coverage),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
