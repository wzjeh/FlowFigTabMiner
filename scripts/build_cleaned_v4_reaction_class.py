#!/usr/bin/env python3
"""Build cleaned v4 dataset with paper-level reaction_class normalization."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"

V3_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
V4_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v4.xlsx"
V4_SUMMARY_JSON = COMPARE_DIR / "my_dataset_cleaning_v4_summary.json"
V4_AUDIT_CSV = COMPARE_DIR / "reaction_class_v4_audit.csv"
V4_AUDIT_SUMMARY_JSON = COMPARE_DIR / "reaction_class_v4_audit_summary.json"


STOPWORDS = {
    "a", "an", "and", "application", "approach", "batch", "by", "chemistry",
    "chem", "compounds", "compound", "continuous", "control", "development",
    "enables", "flow", "for", "from", "generation", "highly", "in", "integrated",
    "microreactor", "microreactors", "microflow", "mode", "of", "on", "one",
    "preparation", "practical", "process", "processes", "production", "rapid",
    "reaction", "reactions", "selective", "synthesis", "system", "systems",
    "the", "their", "through", "to", "towards", "ultrafast", "using", "via",
    "with", "without",
}


TITLE_RULES: list[tuple[str, list[str]]] = [
    ("polymerization", [r"\bpolymeri[sz]ation\b", r"\bpolymer(s|ic)?\b", r"\btelechelic\b"]),
    ("amidation", [r"\bamidation\b", r"\bamide\b"]),
    ("C-N coupling", [r"\bamination\b", r"\bc-n coupling\b"]),
    ("C-O coupling", [r"\bc-o coupling\b", r"\betherification\b"]),
    ("photocatalysis", [r"\bphotocatal", r"\bphotoredox\b"]),
    ("oxidation", [r"\boxidation\b", r"\boxidative\b"]),
    ("reduction", [r"\breduction\b", r"\breductive\b", r"\bdibal\b"]),
    ("hydrogenation", [r"\bhydrogenation\b"]),
    ("halogenation", [r"\bhalogenation\b", r"\biodination\b", r"\bbromination\b", r"\bchlorination\b", r"\bfluorination\b"]),
    ("acylation", [r"\bacylation\b"]),
    ("alkylation", [r"\balkylation\b"]),
    ("hydrolysis", [r"\bhydrolysis\b"]),
    ("anionic cyclization", [r"\bcycli[sz]ation\b", r"\bring closure\b", r"\banionic cyclization\b"]),
    (
        "halogen-metal exchange",
        [
            r"\bhalogen[\s-]metal exchange\b",
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
        "C-C coupling",
        [
            r"\bc-c coupling\b",
            r"\bcross-coupling\b",
            r"\bcoupling\b",
            r"\bhomocoupling\b",
            r"\bmurahashi\b",
            r"\bsuzuki\b",
            r"\bcyanation\b",
            r"\barylation\b",
            r"\bcarbolithiation\b",
            r"\bglycosylation\b",
        ],
    ),
    (
        "directed metalation",
        [
            r"\bdirected metalation\b",
            r"\bdirected lithiation\b",
            r"\bdeprotolithiation\b",
            r"\bdeprotonation\b",
            r"\bmetalation\b",
            r"\bmetalation-substitution\b",
            r"\bortho[- ]?lithiation\b",
            r"\blateral metalation\b",
            r"\blaterally lithiated\b",
            r"\blithiation\b",
        ],
    ),
    (
        "nucleophilic addition",
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
        ],
    ),
]


CONTEXT_RULES: list[tuple[str, list[str]]] = [
    ("polymerization", [r"\bmw/mn\b", r"\bmn\b", r"\bdispersity\b", r"\bliving polymerization\b"]),
    ("amidation", [r"\bamidation\b", r"\bamide\b"]),
    ("C-N coupling", [r"\bamination\b", r"\bc-n coupling\b"]),
    ("C-O coupling", [r"\bc-o coupling\b", r"\betherification\b"]),
    ("oxidation", [r"\boxidation\b", r"\boxidative\b"]),
    ("reduction", [r"\breduction\b", r"\breductive\b", r"\bsingle-electron transfer\b"]),
    ("halogenation", [r"\biodination\b", r"\bbromination\b", r"\bfluorination\b"]),
    ("acylation", [r"\bacylation\b"]),
    ("alkylation", [r"\balkylation\b", r"\bnucleophilic substitution\b"]),
    ("anionic cyclization", [r"\bcycli[sz]ation\b", r"\bintramolecular\b"]),
    ("halogen-metal exchange", [r"\bhalogen[\s-]lithium exchange\b", r"\blithium[\s-]halogen exchange\b", r"\bhalogen dance\b", r"\bbr\s*[-/]\s*li exchange\b"]),
    ("directed metalation", [r"\bdirected metalation\b", r"\bdeprotolithiation\b", r"\bdeprotonation\b", r"\bmetalation\b"]),
    ("C-C coupling", [r"\bcross-coupling\b", r"\bc-c coupling\b", r"\bhomocoupling\b", r"\bmurahashi\b", r"\bsuzuki\b", r"\bcarbolithiation\b"]),
    ("nucleophilic addition", [r"\bnucleophilic addition\b", r"\baddition to\b", r"\btrapping\b", r"\bcarbonyl\b", r"\bimine\b"]),
]


OVERRIDE_MAP = {
    "carbolithiation": "C-C coupling",
    "nucleophilic substitution": "alkylation",
    "unspecified": None,
    "": None,
    "nan": None,
}


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.lower()


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def normalize_existing_class(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    low = normalize_text(text)
    if low in OVERRIDE_MAP:
        return OVERRIDE_MAP[low]
    return text


def infer_from_rules(text: str, rules: list[tuple[str, list[str]]]) -> tuple[str | None, str | None]:
    haystack = normalize_text(text)
    if not haystack:
        return None, None
    for label, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, haystack):
                return label, pattern
    return None, None


def build_intermediate_index() -> list[dict]:
    rows: list[dict] = []
    for paper_dir in sorted(p for p in INTERMEDIATE_DIR.iterdir() if p.is_dir()):
        snippets: list[str] = []
        for path in list(paper_dir.rglob("*local_vars.json"))[:20]:
            try:
                obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            for key in ("reaction_context", "data_interpretation_notes"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    snippets.append(val.strip())
            fixed = obj.get("fixed_conditions") or {}
            if isinstance(fixed, dict):
                note = fixed.get("notes")
                if isinstance(note, str) and note.strip():
                    snippets.append(note.strip())
        scheme = paper_dir / "scheme_conditions.txt"
        if scheme.exists():
            try:
                txt = scheme.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                txt = ""
            if txt:
                snippets.append(txt)
        joined = " | ".join(snippets[:12])
        rows.append(
            {
                "dir_name": paper_dir.name,
                "tokens": tokenize(paper_dir.name),
                "context": joined,
            }
        )
    return rows


def best_intermediate_context(paper_name: str, index: list[dict]) -> tuple[str, str | None]:
    paper_tokens = tokenize(paper_name)
    if not paper_tokens:
        return "", None

    best_score = 0.0
    best_context = ""
    best_dir = None
    for row in index:
        dir_tokens = row["tokens"]
        if not dir_tokens:
            continue
        overlap = paper_tokens & dir_tokens
        if len(overlap) < 2:
            continue
        score = len(overlap) / max(1, min(len(paper_tokens), len(dir_tokens)))
        if score > best_score:
            best_score = score
            best_context = row["context"]
            best_dir = row["dir_name"]
    if best_score >= 0.45:
        return best_context, best_dir
    return "", None


def choose_paper_class(
    paper_name: str,
    counts: dict[str, int],
    intermediate_context: str,
) -> dict:
    title_class, title_pattern = infer_from_rules(paper_name, TITLE_RULES)
    context_class, context_pattern = infer_from_rules(intermediate_context, CONTEXT_RULES)

    filtered_counts = {k: v for k, v in counts.items() if k}
    total = sum(filtered_counts.values())
    top_class = None
    top_share = 0.0
    if filtered_counts:
        top_class, top_n = Counter(filtered_counts).most_common(1)[0]
        top_share = top_n / total if total else 0.0

    chosen = None
    assignment = None
    evidence = None
    needs_review = False

    if title_class:
        chosen = title_class
        assignment = "title_keyword"
        evidence = title_pattern
        if top_class and top_class != chosen and top_share >= 0.35:
            assignment = "title_keyword_over_majority"
            needs_review = True
    elif context_class:
        chosen = context_class
        assignment = "intermediate_context_keyword"
        evidence = context_pattern
        if top_class and top_class != chosen and top_share >= 0.35:
            assignment = "intermediate_context_over_majority"
            needs_review = True
    elif top_class and top_share >= 0.75:
        chosen = top_class
        assignment = "majority_vote_high_conf"
        evidence = f"top_share={top_share:.3f}"
    elif top_class and top_share >= 0.5:
        chosen = top_class
        assignment = "majority_vote_low_conf"
        evidence = f"top_share={top_share:.3f}"
        needs_review = True
    elif top_class:
        chosen = top_class
        assignment = "plurality_vote"
        evidence = f"top_share={top_share:.3f}"
        needs_review = True
    else:
        chosen = "other"
        assignment = "fallback_other"
        evidence = None
        needs_review = True

    return {
        "reaction_class_paper_level_v4": chosen,
        "reaction_class_assignment_v4": assignment,
        "reaction_class_evidence_v4": evidence,
        "reaction_class_needs_review_v4": bool(needs_review),
        "reaction_class_title_match_v4": title_class,
        "reaction_class_context_match_v4": context_class,
        "reaction_class_top_share_v4": round(top_share, 4),
    }


def main() -> None:
    df = pd.read_excel(V3_XLSX)
    df["reaction_class_record_level_v3"] = df["reaction_class"]
    df["paper_key_norm_v4"] = df["paper_basename"].map(normalize_text)

    intermediate_index = build_intermediate_index()

    paper_rows: list[dict] = []
    for paper_key, grp in df.groupby("paper_key_norm_v4", dropna=False):
        paper_names = sorted(set(grp["paper_basename"].fillna("NA").astype(str)))
        paper_name = paper_names[0] if paper_names else str(paper_key)
        counts = Counter()
        for val in grp["reaction_class_record_level_v3"]:
            norm = normalize_existing_class(val)
            if norm:
                counts[norm] += 1

        intermediate_context, intermediate_dir = best_intermediate_context(paper_name, intermediate_index)
        decision = choose_paper_class(paper_name, dict(counts), intermediate_context)
        decision.update(
            {
                "paper_key_norm_v4": paper_key,
                "paper_names_json": json.dumps(paper_names, ensure_ascii=False),
                "doi_list_json": json.dumps(sorted(set(grp["paper_doi"].fillna("NA").astype(str))), ensure_ascii=False),
                "rows": int(len(grp)),
                "record_level_counts_json": json.dumps(dict(counts), ensure_ascii=False),
                "matched_intermediate_dir_v4": intermediate_dir,
                "intermediate_context_excerpt_v4": intermediate_context[:1000] if intermediate_context else "",
            }
        )
        paper_rows.append(decision)

    audit_df = pd.DataFrame(paper_rows).sort_values(
        ["reaction_class_needs_review_v4", "rows"],
        ascending=[False, False],
    )
    audit_df.to_csv(V4_AUDIT_CSV, index=False)

    merge_cols = [
        "paper_key_norm_v4",
        "reaction_class_paper_level_v4",
        "reaction_class_assignment_v4",
        "reaction_class_evidence_v4",
        "reaction_class_needs_review_v4",
        "reaction_class_title_match_v4",
        "reaction_class_context_match_v4",
        "reaction_class_top_share_v4",
        "matched_intermediate_dir_v4",
        "intermediate_context_excerpt_v4",
    ]
    df = df.merge(audit_df[merge_cols], on="paper_key_norm_v4", how="left")
    df["reaction_class"] = df["reaction_class_paper_level_v4"]

    summary = {
        "input_file": str(V3_XLSX),
        "output_file": str(V4_XLSX),
        "audit_csv": str(V4_AUDIT_CSV),
        "rows": int(len(df)),
        "paper_groups": int(df["paper_key_norm_v4"].nunique(dropna=True)),
        "rows_changed_from_v3": int(
            (
                df["reaction_class_record_level_v3"].fillna("Unspecified").astype(str)
                != df["reaction_class"].fillna("Unspecified").astype(str)
            ).sum()
        ),
        "papers_needing_review_v4": int(audit_df["reaction_class_needs_review_v4"].sum()),
        "assignment_counts_v4": dict(Counter(df["reaction_class_assignment_v4"])),
        "paper_level_class_counts_v4": dict(Counter(df["reaction_class"])),
    }

    audit_summary = {
        "paper_groups": int(len(audit_df)),
        "papers_needing_review_v4": int(audit_df["reaction_class_needs_review_v4"].sum()),
        "assignment_counts_v4": dict(Counter(audit_df["reaction_class_assignment_v4"])),
        "top_review_examples": audit_df.loc[
            audit_df["reaction_class_needs_review_v4"],
            [
                "paper_names_json",
                "rows",
                "reaction_class_paper_level_v4",
                "reaction_class_assignment_v4",
                "record_level_counts_json",
                "matched_intermediate_dir_v4",
            ],
        ].head(20).to_dict(orient="records"),
    }

    with pd.ExcelWriter(V4_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)
    V4_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    V4_AUDIT_SUMMARY_JSON.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v4 workbook: {V4_XLSX}")
    print(f"Saved cleaned v4 summary: {V4_SUMMARY_JSON}")
    print(f"Saved v4 audit CSV: {V4_AUDIT_CSV}")
    print(f"Saved v4 audit summary JSON: {V4_AUDIT_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
