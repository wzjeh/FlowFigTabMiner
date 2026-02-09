#!/usr/bin/env python3
"""Build metadata.jsonl and metadata.csv for downloaded PDFs."""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE_DIR = "/Users/zhaowenyuan/Projects/FlowFigTabMiner/data/papers"
UA = "FlowFigTabMiner/1.0 (mailto:local)"
def parse_year_from_filename(filename: str) -> str:
    match = re.search(r"_(\\d{4})", filename)
    return match.group(1) if match else ""


def main() -> None:
    rows = []

    # PMC
    pmc_dir = os.path.join(BASE_DIR, "pmc")
    if os.path.isdir(pmc_dir):
        for fn in sorted(os.listdir(pmc_dir)):
            if not fn.endswith(".pdf"):
                continue
            pmcid = fn.split("_", 1)[0]
            year = parse_year_from_filename(fn)
            rows.append({
                "source": "PMC",
                "id": pmcid,
                "title": "",
                "year": year,
                "pdf_url": "",
                "keyword": "",
                "path": os.path.join(pmc_dir, fn),
            })

    # arXiv
    arxiv_dir = os.path.join(BASE_DIR, "arxiv")
    arxiv_files = []
    if os.path.isdir(arxiv_dir):
        for fn in sorted(os.listdir(arxiv_dir)):
            if not fn.endswith(".pdf"):
                continue
            arxiv_id = fn.split("_", 1)[0]
            arxiv_files.append((arxiv_id, fn))
    for arxiv_id, fn in arxiv_files:
        year = parse_year_from_filename(fn)
        rows.append({
            "source": "arXiv",
            "id": arxiv_id,
            "title": "",
            "year": year,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "keyword": "",
            "path": os.path.join(arxiv_dir, fn),
        })

    jsonl_path = os.path.join(BASE_DIR, "metadata.jsonl")
    csv_path = os.path.join(BASE_DIR, "metadata.csv")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source", "id", "title", "year", "pdf_url", "keyword", "path"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote metadata for {len(rows)} PDFs")


if __name__ == "__main__":
    main()
