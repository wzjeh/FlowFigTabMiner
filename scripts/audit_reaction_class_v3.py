#!/usr/bin/env python3
"""Audit reaction_class consistency in cleaned v3 using intermediate context files."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"

V3_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
AUDIT_CSV = COMPARE_DIR / "reaction_class_v3_audit.csv"
AUDIT_SUMMARY_JSON = COMPARE_DIR / "reaction_class_v3_audit_summary.json"


def normalize_paper_name(text: str) -> str:
    text = str(text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.lower()


def collect_intermediate_context(paper_basename: str) -> str:
    paper_dir = INTERMEDIATE_DIR / str(paper_basename)
    if not paper_dir.exists():
        return ""
    snippets = []
    for path in list(paper_dir.rglob("*local_vars.json"))[:20]:
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for key in ["reaction_context", "data_interpretation_notes"]:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                snippets.append(val.strip())
        fixed = obj.get("fixed_conditions") or {}
        if isinstance(fixed, dict):
            for key in ["notes", "catalyst", "reactor_type"]:
                val = fixed.get(key)
                if isinstance(val, str) and val.strip():
                    snippets.append(val.strip())
    scheme = paper_dir / "scheme_conditions.txt"
    if scheme.exists():
        try:
            txt = scheme.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                snippets.append(txt)
        except Exception:
            pass
    return " | ".join(snippets[:12])


def main() -> None:
    df = pd.read_excel(V3_XLSX)
    df["reaction_class_clean"] = df["reaction_class"].fillna("Unspecified")
    df["paper_key_norm"] = df["paper_basename"].map(normalize_paper_name)

    rows = []
    for key, grp in df.groupby("paper_key_norm", dropna=False):
        class_counts = grp["reaction_class_clean"].value_counts()
        n_classes = int(class_counts.size)
        top_class = class_counts.index[0]
        top_class_pct = float(class_counts.iloc[0] / len(grp) * 100)
        paper_names = sorted(set(grp["paper_basename"].fillna("NA").astype(str)))
        dois = sorted(set(grp["paper_doi"].fillna("NA").astype(str)))
        context = collect_intermediate_context(paper_names[0]) if paper_names else ""
        rows.append(
            {
                "paper_key_norm": key,
                "rows": int(len(grp)),
                "n_classes": n_classes,
                "top_class": top_class,
                "top_class_pct": top_class_pct,
                "class_counts_json": json.dumps(class_counts.to_dict(), ensure_ascii=False),
                "paper_names_json": json.dumps(paper_names, ensure_ascii=False),
                "doi_list_json": json.dumps(dois, ensure_ascii=False),
                "has_name_variant": len(paper_names) > 1,
                "intermediate_context_excerpt": context,
            }
        )

    audit_df = pd.DataFrame(rows).sort_values(["n_classes", "rows", "top_class_pct"], ascending=[False, False, True])
    audit_df.to_csv(AUDIT_CSV, index=False)

    summary = {
        "input_file": str(V3_XLSX),
        "output_csv": str(AUDIT_CSV),
        "paper_groups": int(len(audit_df)),
        "inconsistent_papers": int((audit_df["n_classes"] > 1).sum()),
        "papers_with_name_variants": int(audit_df["has_name_variant"].sum()),
        "top_inconsistent_examples": audit_df.loc[audit_df["n_classes"] > 1, ["paper_names_json", "class_counts_json"]].head(10).to_dict(orient="records"),
    }
    AUDIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved audit CSV: {AUDIT_CSV}")
    print(f"Saved summary JSON: {AUDIT_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
