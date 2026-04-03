"""
Prepare VLM input batch for tR sub-dataset extraction.

Collects:
  - 49 heatmap sub-images (from v10 raw figures, excluding zz_ test dirs)
  - 10 tR-containing table images (from 5 papers)

Outputs a flat directory structure under data/vlm_input_tr/ with:
  - heatmaps/        ← PNG files + heatmap_prompt.txt
  - tables/          ← PNG files + table_prompt.txt + mol_meta.json (for 80 only, already exist)
  - manifest.csv     ← master list of all images with metadata

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/prepare_vlm_tr_batch.py
"""

import os
import sys
import csv
import shutil
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

HEATMAP_SRC = os.path.join(
    PROJECT_ROOT,
    "data/final_output/dataset_comparison/organolithium_tr_points_raw_figures_only_v10",
)

INTERMEDIATE = os.path.join(PROJECT_ROOT, "data/intermediate")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/vlm_input_tr")

HEATMAP_PROMPT = os.path.join(PROJECT_ROOT, "eval/vlm_prompts/heatmap_prompt.txt")
TABLE_PROMPT = os.path.join(PROJECT_ROOT, "eval/vlm_prompts/table_prompt.txt")

# tR tables: (paper_dir_name, table_id, short_label)
TR_TABLES = [
    ("80", "page_5_table_1", "80"),
    ("80", "page_6_table_0", "80"),
    ("80", "page_8_table_1", "80"),
    ("80", "page_9_table_1", "80"),
    (
        "Degennaro et al. 2016 - A direct and sustainable synthesis of tertiary butyl esters enabled by flow microreactors",
        "page_3_table_0",
        "Degennaro2016",
    ),
    (
        "Fukuyama et al. 2014 - Flow update for the carbonylation of 1-silyl-substituted organolithiums under CO pressure",
        "page_2_table_0",
        "Fukuyama2014",
    ),
    (
        "Sun et al. 2020 - Practical and rapid construction of 2-pyridyl ketone library in continuous flow",
        "page_3_table_0",
        "Sun2020",
    ),
    (
        "Sun et al. 2020 - Practical and rapid construction of 2-pyridyl ketone library in continuous flow",
        "page_3_table_1",
        "Sun2020",
    ),
    (
        "Wong et al. 2021 - Flash chemistry enables high productivity metalation-substitution of 5-alkyltetrazoles",
        "page_5_table_0",
        "Wong2021",
    ),
    (
        "Wong et al. 2021 - Flash chemistry enables high productivity metalation-substitution of 5-alkyltetrazoles",
        "page_7_table_2",
        "Wong2021",
    ),
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _extract_index(dirname):
    """Extract leading numeric index from directory name like '03_...' → 3."""
    m = re.match(r"^(\d+)_", dirname)
    return int(m.group(1)) if m else 999


def collect_heatmaps(out_dir):
    """Collect heatmap sub-images into out_dir/heatmaps/."""
    hm_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(hm_dir, exist_ok=True)

    # Copy prompt
    if os.path.exists(HEATMAP_PROMPT):
        shutil.copy2(HEATMAP_PROMPT, os.path.join(hm_dir, "heatmap_prompt.txt"))

    manifest_rows = []

    if not os.path.isdir(HEATMAP_SRC):
        print(f"[WARN] Heatmap source not found: {HEATMAP_SRC}")
        return manifest_rows

    for fig_dir_name in sorted(os.listdir(HEATMAP_SRC)):
        # Skip zz_ test directories
        if "zz_" in fig_dir_name:
            continue

        fig_dir = os.path.join(HEATMAP_SRC, fig_dir_name)
        if not os.path.isdir(fig_dir):
            continue

        idx = _extract_index(fig_dir_name)
        # Extract paper name (everything after the index prefix)
        paper_name = re.sub(r"^\d+_", "", fig_dir_name)

        # Walk into sub-figure directories to find *_raw.png
        for root, dirs, files in os.walk(fig_dir):
            for f in sorted(files):
                if f.endswith("_raw.png"):
                    src = os.path.join(root, f)
                    # Create flat filename: {idx:02d}_{figure}_{tN}.png
                    # e.g., 03_page_2_figure_1_t0.png
                    base = f.replace("_raw.png", "")
                    dst_name = f"{idx:02d}_{base}.png"
                    dst = os.path.join(hm_dir, dst_name)
                    shutil.copy2(src, dst)

                    manifest_rows.append({
                        "type": "heatmap",
                        "file": f"heatmaps/{dst_name}",
                        "paper": paper_name,
                        "v10_index": idx,
                        "source_path": src,
                    })

    print(f"[Heatmap] Collected {len(manifest_rows)} images → {hm_dir}")
    return manifest_rows


def collect_tables(out_dir):
    """Collect tR table images + existing mol_meta into out_dir/tables/."""
    tbl_dir = os.path.join(out_dir, "tables")
    os.makedirs(tbl_dir, exist_ok=True)

    # Copy prompt
    if os.path.exists(TABLE_PROMPT):
        shutil.copy2(TABLE_PROMPT, os.path.join(tbl_dir, "table_prompt.txt"))

    manifest_rows = []

    for paper_dir_name, table_id, short_label in TR_TABLES:
        paper_tables = os.path.join(INTERMEDIATE, paper_dir_name, "tables")
        img_src = os.path.join(paper_tables, f"{table_id}.png")

        if not os.path.exists(img_src):
            print(f"[WARN] Table image not found: {img_src}")
            continue

        # Flat filename: {short_label}_{table_id}.png
        dst_name = f"{short_label}_{table_id}.png"
        dst = os.path.join(tbl_dir, dst_name)
        shutil.copy2(img_src, dst)

        # Copy existing mol_meta if available (from vlm_input)
        vlm_input = os.path.join(INTERMEDIATE, paper_dir_name, "vlm_input")
        mol_meta_src = os.path.join(vlm_input, f"{table_id}_mol_meta.json")
        if os.path.exists(mol_meta_src):
            mol_meta_dst = os.path.join(tbl_dir, f"{short_label}_{table_id}_mol_meta.json")
            shutil.copy2(mol_meta_src, mol_meta_dst)
            mol_meta_status = "copied"
        else:
            mol_meta_status = "missing"

        manifest_rows.append({
            "type": "table",
            "file": f"tables/{dst_name}",
            "paper": paper_dir_name,
            "table_id": table_id,
            "short_label": short_label,
            "mol_meta": mol_meta_status,
        })

    print(f"[Table] Collected {len(manifest_rows)} images → {tbl_dir}")
    return manifest_rows


def write_manifest(out_dir, hm_rows, tbl_rows):
    """Write manifest.csv listing all collected images."""
    manifest_path = os.path.join(out_dir, "manifest.csv")

    all_rows = []
    for r in hm_rows:
        all_rows.append({
            "type": r["type"],
            "file": r["file"],
            "paper": r["paper"],
            "v10_index": r.get("v10_index", ""),
            "table_id": "",
            "mol_meta": "",
            "vlm_done": "",
        })
    for r in tbl_rows:
        all_rows.append({
            "type": r["type"],
            "file": r["file"],
            "paper": r["paper"],
            "v10_index": "",
            "table_id": r.get("table_id", ""),
            "mol_meta": r.get("mol_meta", ""),
            "vlm_done": "",
        })

    fieldnames = ["type", "file", "paper", "v10_index", "table_id", "mol_meta", "vlm_done"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[Manifest] Written {len(all_rows)} rows → {manifest_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hm_rows = collect_heatmaps(OUTPUT_DIR)
    tbl_rows = collect_tables(OUTPUT_DIR)
    write_manifest(OUTPUT_DIR, hm_rows, tbl_rows)

    # Summary
    print("\n" + "=" * 60)
    print(f"Total heatmaps: {len(hm_rows)}")
    print(f"Total tables:   {len(tbl_rows)}")
    tbl_missing = [r for r in tbl_rows if r.get("mol_meta") == "missing"]
    if tbl_missing:
        print(f"\nTables missing mol_meta ({len(tbl_missing)}):")
        for r in tbl_missing:
            print(f"  - {r['file']}  ({r['paper']})")
        print("\nTo generate mol_meta for these, run:")
        print("  python -c \"from src.extraction.table.vlm_table_pipeline import prepare_vlm_batch; prepare_vlm_batch('data/intermediate/<paper>')\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
