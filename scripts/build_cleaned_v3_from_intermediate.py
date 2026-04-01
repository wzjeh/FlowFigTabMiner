#!/usr/bin/env python3
"""Build cleaned v3 dataset with explicit v2/v3 rescue markers using intermediate files."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"

V2_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v2.xlsx"
V3_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
V3_JSON = COMPARE_DIR / "my_dataset_cleaning_v3_summary.json"
V3_AUDIT_CSV = COMPARE_DIR / "my_dataset_v3_intermediate_rescue_audit.csv"


def classify_reagent_family_from_text(text: str) -> str | None:
    if not text:
        return None
    t = str(text).lower().strip()
    if not t or t == "nan":
        return None

    patterns = [
        ("n-BuLi", [r"\bn-?butyl ?lithium\b", r"\bnbuli\b", r"\bn-buli\b"]),
        ("s-BuLi", [r"\bsec-?butyl ?lithium\b", r"\bs-?butyllithium\b", r"\bsbuli\b"]),
        ("t-BuLi", [r"\btert-?butyl ?lithium\b", r"\bt-?butyllithium\b", r"\btbuli\b", r"\btertbutyllithium\b"]),
        ("MeLi", [r"\bmethyl ?lithium\b", r"\bmeli\b"]),
        ("PhLi", [r"\bphenyl ?lithium\b", r"\bphli\b"]),
        ("LDA", [r"\blithium diisopropylamide\b", r"\blda\b"]),
        ("LiHMDS", [r"\blihmds\b", r"lithium hexamethyldisilazide"]),
        ("Lithium naphthalenide", [r"lithium naphthalenide", r"\bnaphthalenide\b"]),
        ("HexylLi", [r"\bhexyl ?lithium\b", r"\bn-?hexyllithium\b"]),
        ("EthylLi", [r"\bethyl ?lithium\b"]),
        ("MesLi", [r"\bmesli\b", r"\bmesityllithium\b"]),
        ("Vinyl/alkynylLi", [r"vinyllithium", r"lithio.*yne", r"lithiophenylacetylene", r"ynolate"]),
        ("Aryl/heteroarylLi", [r"aryllithium", r"lithiothiophene", r"lithiopyrid", r"lithiofuran", r"lithiotoluene", r"phenyllithium"]),
        ("Other organolithium", [r"organolithium", r"alkyllithium", r"dilithio", r"\blithio\b"]),
    ]
    hits = []
    for label, pats in patterns:
        if any(re.search(p, t) for p in pats):
            hits.append(label)
    if not hits:
        return None
    return hits[0] if len(set(hits)) == 1 else None


def artifact_candidate(paper_basename: str) -> bool:
    return bool(re.search(r"^(example|example1|zz_codex_smoke)", str(paper_basename or ""), flags=re.I))


def parse_source_table_or_figure(label: str) -> list[str]:
    if not isinstance(label, str) or not label.strip():
        return []
    text = label.strip()
    candidates: list[str] = []

    m = re.search(r"(?i)\b(table|figure)\s*(\d+)([a-z])?\s*(?:\(p\.?\s*(\d+)\))?", text)
    if not m:
        return candidates

    kind = m.group(1).lower()
    number = int(m.group(2)) - 1
    page = m.group(4)
    base = f"{kind}_{number}"
    if page:
        if kind == "figure":
            candidates.append(f"page_{page}_{base}_t0")
            candidates.append(f"page_{page}_{base}")
        else:
            candidates.append(f"page_{page}_{base}")
    if kind == "figure":
        candidates.append(f"{base}_t0")
    candidates.append(base)
    return candidates


def source_id_from_path(path: Path) -> str | None:
    name = path.name
    for suffix in ["_local_vars.json", "_extracted.csv", "_evidence.json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name == "scheme_conditions.txt":
        return "scheme_conditions"
    if name == "compound_pool.json":
        return "compound_pool"
    return None


def build_intermediate_index(paper_basename: str) -> tuple[dict[str, str], str | None, list[dict]]:
    paper_dir = INTERMEDIATE_DIR / str(paper_basename)
    source_family_hits: dict[str, list[str]] = defaultdict(list)
    audit_rows: list[dict] = []
    paper_level_hits: list[str] = []

    if not paper_dir.exists():
        return {}, None, audit_rows

    files = list(paper_dir.rglob("*.json")) + list(paper_dir.rglob("*.txt")) + list(paper_dir.rglob("*.csv"))
    for file in files:
        source_id = source_id_from_path(file)
        if source_id is None:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        fam = classify_reagent_family_from_text(text)
        audit_rows.append(
            {
                "paper_basename": paper_basename,
                "file": str(file),
                "source_id": source_id,
                "matched_family": fam,
            }
        )
        if fam:
            if source_id in {"scheme_conditions", "compound_pool"}:
                paper_level_hits.append(fam)
            else:
                source_family_hits[source_id].append(fam)

    resolved_source_map: dict[str, str] = {}
    for source_id, hits in source_family_hits.items():
        uniq = sorted(set(hits))
        if len(uniq) == 1:
            resolved_source_map[source_id] = uniq[0]

    paper_family = None
    paper_candidates = set(paper_level_hits) | set(resolved_source_map.values())
    if len(paper_candidates) == 1:
        paper_family = next(iter(paper_candidates))

    return resolved_source_map, paper_family, audit_rows


def main() -> None:
    df = pd.read_excel(V2_XLSX)

    df["reagent_family_v2"] = df["reagent_family"]
    df["reagent_family_assignment_v2"] = df["reagent_family_assignment"]
    df["reagent_family_rescued_v2"] = df["reagent_family_assignment_v2"].ne("direct")
    df["artifact_candidate_v3"] = df["paper_basename"].map(artifact_candidate)
    df["exclude_candidate_v3"] = df["artifact_candidate_v3"]
    df["reagent_family_assignment_v3"] = df["reagent_family_assignment_v2"]
    df["reagent_family_rescued_v3"] = False
    df["reagent_family_v3_source_id"] = pd.NA

    unresolved_mask = df["reagent_family"].eq("Unclassified") & ~df["artifact_candidate_v3"]
    paper_basenames = sorted(df.loc[unresolved_mask, "paper_basename"].dropna().astype(str).unique())

    all_audit_rows: list[dict] = []
    per_paper_source_map: dict[str, dict[str, str]] = {}
    per_paper_family: dict[str, str | None] = {}
    for paper in paper_basenames:
        source_map, paper_family, audit_rows = build_intermediate_index(paper)
        per_paper_source_map[paper] = source_map
        per_paper_family[paper] = paper_family
        all_audit_rows.extend(audit_rows)

    rescued_source = 0
    rescued_paper = 0
    for idx, row in df.loc[unresolved_mask].iterrows():
        paper = str(row.get("paper_basename") or "")
        source_label = row.get("source_table_or_figure")
        source_map = per_paper_source_map.get(paper, {})
        paper_family = per_paper_family.get(paper)

        assigned = None
        assigned_source = None
        candidates = parse_source_table_or_figure(source_label)
        for cand in candidates:
            if cand in source_map:
                assigned = source_map[cand]
                assigned_source = cand
                break

        if assigned is None and candidates:
            # Allow page-unspecified fallback when a table/figure index maps uniquely within the paper.
            reduced = [c for c in candidates if c.startswith("table_") or c.startswith("figure_") or re.match(r"^(table|figure)_\d+(_t0)?$", c)]
            for cand in reduced:
                suffix = cand.replace("_t0", "")
                matches = {sid: fam for sid, fam in source_map.items() if sid.endswith(suffix) or suffix in sid}
                if len(set(matches.values())) == 1 and matches:
                    assigned_source = sorted(matches)[0]
                    assigned = next(iter(set(matches.values())))
                    break

        if assigned is not None:
            df.at[idx, "reagent_family"] = assigned
            df.at[idx, "reagent_family_assignment"] = "v3_intermediate_source"
            df.at[idx, "reagent_family_assignment_v3"] = "v3_intermediate_source"
            df.at[idx, "reagent_family_rescued_v3"] = True
            df.at[idx, "reagent_family_v3_source_id"] = assigned_source
            rescued_source += 1
            continue

        if paper_family is not None:
            df.at[idx, "reagent_family"] = paper_family
            df.at[idx, "reagent_family_assignment"] = "v3_intermediate_paper_unique"
            df.at[idx, "reagent_family_assignment_v3"] = "v3_intermediate_paper_unique"
            df.at[idx, "reagent_family_rescued_v3"] = True
            df.at[idx, "reagent_family_v3_source_id"] = "paper_unique"
            rescued_paper += 1

    audit_df = pd.DataFrame(all_audit_rows)
    if not audit_df.empty:
        audit_df.to_csv(V3_AUDIT_CSV, index=False)

    summary = {
        "input_file": str(V2_XLSX),
        "output_file": str(V3_XLSX),
        "artifact_candidates": int(df["artifact_candidate_v3"].sum()),
        "exclude_candidates": int(df["exclude_candidate_v3"].sum()),
        "v2_unclassified": int((df["reagent_family_v2"] == "Unclassified").sum()),
        "v3_unclassified": int((df["reagent_family"] == "Unclassified").sum()),
        "v3_rescued_total": int(df["reagent_family_rescued_v3"].sum()),
        "v3_rescued_from_intermediate_source": rescued_source,
        "v3_rescued_from_intermediate_paper_unique": rescued_paper,
        "v2_assignment_counts": dict(Counter(df["reagent_family_assignment_v2"])),
        "v3_assignment_counts": dict(Counter(df["reagent_family_assignment"])),
    }

    with pd.ExcelWriter(V3_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)
    V3_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v3 workbook: {V3_XLSX}")
    print(f"Saved cleaned v3 summary: {V3_JSON}")
    print(f"Saved v3 audit CSV: {V3_AUDIT_CSV}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
