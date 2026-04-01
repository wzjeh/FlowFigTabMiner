#!/usr/bin/env python3
"""Build cleaned v5 dataset with reagent-level organolithium role labels."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

V4_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v4.xlsx"
V5_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v5.xlsx"
V5_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v5_modeling.xlsx"
V5_SUMMARY_JSON = COMPARE_DIR / "my_dataset_cleaning_v5_summary.json"
V5_ROLE_AUDIT_CSV = COMPARE_DIR / "organolithium_role_v5_audit.csv"


ROLE_ORDER = [
    "Base-mediated deprotonation",
    "Halogen-lithium exchange",
    "Nucleophilic organolithium",
    "C-C bond-forming organolithium",
    "SET reductant / reductive lithiation",
    "Anionic polymerization initiator",
    "Ambiguous / other",
]


TITLE_PATTERNS = [
    ("Anionic polymerization initiator", [r"\bpolymeri[sz]ation\b", r"\btelechelic\b", r"\bliving\b"]),
    ("SET reductant / reductive lithiation", [r"\breductive\b", r"\breduction\b", r"\bradical anion\b", r"\bdimerization\b", r"\bynolate\b"]),
    (
        "Halogen-lithium exchange",
        [
            r"\bhalogen[\s-]lithium exchange\b",
            r"\blithium[\s-]halogen exchange\b",
            r"\biodine[\s-]lithium exchange\b",
            r"\bbromine[\s-]lithium exchange\b",
            r"\bbr\s*[-/]\s*li exchange\b",
            r"\bi\s*[-/]\s*li exchange\b",
            r"\bhalogen dance\b",
        ],
    ),
    (
        "Base-mediated deprotonation",
        [
            r"\bdeprotolithiation\b",
            r"\bdeprotonation\b",
            r"\bdirected metalation\b",
            r"\bdirected lithiation\b",
            r"\bortholithiation\b",
            r"\bortho[- ]?lithiation\b",
            r"\bmetalation\b",
            r"\blithiation-substitution\b",
        ],
    ),
    (
        "C-C bond-forming organolithium",
        [
            r"\bcross-coupling\b",
            r"\bc-c coupling\b",
            r"\bcoupling\b",
            r"\bhomocoupling\b",
            r"\bmurahashi\b",
            r"\bsuzuki\b",
            r"\bcyanation\b",
            r"\bcarbolithiation\b",
            r"\bc-glycosylation\b",
            r"\bglycosylation\b",
        ],
    ),
    (
        "Nucleophilic organolithium",
        [
            r"\bnucleophilic addition\b",
            r"\baddition to\b",
            r"\bpropargylation\b",
            r"\bketone(s)?\b",
            r"\baldehyde(s)?\b",
            r"\bimine(s)?\b",
            r"\bacid chloride(s)?\b",
            r"\bketo ester(s)?\b",
            r"\bketoamide(s)?\b",
            r"\bacylation\b",
            r"\bamidation\b",
            r"\bamination\b",
        ],
    ),
]


FAMILY_DEFAULTS = {
    "LDA": "Base-mediated deprotonation",
    "LiHMDS": "Base-mediated deprotonation",
    "Lithium naphthalenide": "SET reductant / reductive lithiation",
}


def infer_role_from_title(text: str) -> tuple[str | None, str | None]:
    haystack = str(text or "").lower()
    for role, patterns in TITLE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, haystack):
                return role, pattern
    return None, None


def choose_role(row: pd.Series) -> tuple[str, str, str | None, bool]:
    paper = str(row.get("paper_basename") or "")
    reagent_family = str(row.get("reagent_family") or "")
    reaction_class = str(row.get("reaction_class") or "")
    assignment = None
    evidence = None
    needs_review = False

    title_role, title_pattern = infer_role_from_title(paper)

    if reagent_family in {"LDA", "LiHMDS"}:
        if title_role in {
            "Base-mediated deprotonation",
            "Halogen-lithium exchange",
            "Anionic polymerization initiator",
            "SET reductant / reductive lithiation",
        }:
            return title_role, "family_override_title", title_pattern, False
        return "Base-mediated deprotonation", "family_override_default", None, False

    if reagent_family == "Lithium naphthalenide":
        if title_role in {"Anionic polymerization initiator", "SET reductant / reductive lithiation"}:
            return title_role, "family_override_title", title_pattern, False
        if "carbamoyl anion" in paper.lower() or "generation and reaction" in paper.lower():
            return "SET reductant / reductive lithiation", "family_override_context", "carbamoyl/generation context", False
        return "SET reductant / reductive lithiation", "family_override_default", None, False

    if title_role:
        return title_role, "title_keyword", title_pattern, False

    rc_map = {
        "directed metalation": "Base-mediated deprotonation",
        "halogen-metal exchange": "Halogen-lithium exchange",
        "nucleophilic addition": "Nucleophilic organolithium",
        "acylation": "Nucleophilic organolithium",
        "amidation": "Nucleophilic organolithium",
        "alkylation": "Nucleophilic organolithium",
        "C-C coupling": "C-C bond-forming organolithium",
        "anionic cyclization": "C-C bond-forming organolithium",
        "C-N coupling": "Nucleophilic organolithium",
        "C-O coupling": "Nucleophilic organolithium",
        "polymerization": "Anionic polymerization initiator",
        "reduction": "SET reductant / reductive lithiation",
        "oxidation": "Ambiguous / other",
        "halogenation": "Ambiguous / other",
        "photocatalysis": "Ambiguous / other",
        "other": "Ambiguous / other",
        "Unspecified": "Ambiguous / other",
        "nan": "Ambiguous / other",
    }
    mapped = rc_map.get(reaction_class, "Ambiguous / other")
    assignment = "reaction_class_bridge"
    evidence = reaction_class if reaction_class else None
    if mapped == "Ambiguous / other":
        needs_review = True
    return mapped, assignment, evidence, needs_review


def main() -> None:
    df = pd.read_excel(V4_XLSX)

    roles = []
    for _, row in df.iterrows():
        role, assignment, evidence, needs_review = choose_role(row)
        roles.append((role, assignment, evidence, needs_review))

    df["organolithium_role_v5"] = [x[0] for x in roles]
    df["organolithium_role_assignment_v5"] = [x[1] for x in roles]
    df["organolithium_role_evidence_v5"] = [x[2] for x in roles]
    df["organolithium_role_needs_review_v5"] = [x[3] for x in roles]

    with pd.ExcelWriter(V5_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)

    modeling_df = df.loc[~df["exclude_candidate_v3"].fillna(False)].copy()
    with pd.ExcelWriter(V5_MODELING_XLSX, engine="openpyxl") as writer:
        modeling_df.to_excel(writer, sheet_name="All Records", index=False)

    audit = (
        df.groupby(["reagent_family", "organolithium_role_v5", "organolithium_role_assignment_v5"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["reagent_family", "rows"], ascending=[True, False])
    )
    audit.to_csv(V5_ROLE_AUDIT_CSV, index=False)

    summary = {
        "input_file": str(V4_XLSX),
        "output_file": str(V5_XLSX),
        "modeling_output_file": str(V5_MODELING_XLSX),
        "audit_csv": str(V5_ROLE_AUDIT_CSV),
        "rows": int(len(df)),
        "rows_modeling": int(len(modeling_df)),
        "artifact_excluded_modeling": int(df["exclude_candidate_v3"].fillna(False).sum()),
        "role_counts_all_v5": dict(Counter(df["organolithium_role_v5"])),
        "role_counts_modeling_v5": dict(Counter(modeling_df["organolithium_role_v5"])),
        "assignment_counts_all_v5": dict(Counter(df["organolithium_role_assignment_v5"])),
        "assignment_counts_modeling_v5": dict(Counter(modeling_df["organolithium_role_assignment_v5"])),
        "needs_review_rows_v5": int(df["organolithium_role_needs_review_v5"].sum()),
    }
    V5_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v5 workbook: {V5_XLSX}")
    print(f"Saved cleaned v5 modeling workbook: {V5_MODELING_XLSX}")
    print(f"Saved v5 role audit CSV: {V5_ROLE_AUDIT_CSV}")
    print(f"Saved v5 summary: {V5_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
