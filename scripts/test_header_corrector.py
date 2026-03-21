"""
Quick test for HeaderCorrector on real extracted tables.

Usage (from project root, inside venv):
    python scripts/test_header_corrector.py
"""

import sys
import os
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.table.header_corrector import HeaderCorrector

# ------------------------------------------------------------------
# Test cases: (csv_path, image_path, description)
# ------------------------------------------------------------------
BASE = "data/intermediate"
CASES = [
    (
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_13_table_1_extracted.csv",
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_13_table_1.png",
        "Double-header: 'ionic liquid' row + Ered/Eox sub-headers",
    ),
    (
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_4_table_2_extracted.csv",
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_4_table_2.png",
        "Sparse row-0: mostly empty cells (spanning header likely)",
    ),
    (
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_5_table_1_extracted.csv",
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_5_table_1.png",
        "row-0 looks like header but has odd OCR artefact '(W)3'",
    ),
    (
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_6_table_1_extracted.csv",
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_6_table_1.png",
        "Standard 2-column compound/E(V) table — should NOT trigger VLM",
    ),
    (
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_17_table_2_extracted.csv",
        f"{BASE}/modern-strategies-in-electroorganic-synthesis/tables/page_17_table_2.png",
        "Clean substrate/product/yield table — should NOT trigger VLM",
    ),
]


def load_csv_as_grid(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def print_grid(grid: list, title: str = ""):
    if title:
        print(f"\n  {title}")
    for i, row in enumerate(grid[:4]):          # show first 4 rows only
        print(f"    row{i}: {row}")
    if len(grid) > 4:
        print(f"    ... ({len(grid) - 4} more data rows)")


def main():
    # Minimal fake structure_data (no TATR header_regions → forces heuristic path)
    fake_structure = {"header_regions": [], "rows": []}

    # Use VLM-enabled config (reads QWEN_API_KEY from .env)
    config = {
        "enabled": True,
        "provider": "dashscope",
        "model_name": "qwen-vl-plus",
        "trigger_threshold": 0.5,
        "send_header_crop_only": False,
    }

    hc = HeaderCorrector(config)

    print("\n" + "=" * 70)
    print("HeaderCorrector test — checking trigger heuristics + VLM correction")
    print("=" * 70)

    for csv_path, img_path, desc in CASES:
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {desc}\n  File not found: {csv_path}")
            continue

        grid = load_csv_as_grid(csv_path)
        if not grid:
            print(f"\n[SKIP] {desc} — empty CSV")
            continue

        print(f"\n{'─' * 70}")
        print(f"Case: {desc}")
        print(f"  CSV: {csv_path}")
        print_grid(grid, "BEFORE:")

        triggers = hc.needs_correction(grid, fake_structure)
        print(f"\n  needs_correction() → {triggers}")

        if triggers:
            if not os.path.exists(img_path):
                print(f"  [SKIP VLM] image not found: {img_path}")
                continue
            print("  Calling VLM (qwen-vl-plus)...")
            corrected = hc.correct(img_path, grid, fake_structure)
            print_grid(corrected, "AFTER:")
        else:
            print("  → Heuristic passed, no VLM call needed ✓")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
