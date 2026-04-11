#!/usr/bin/env python3
"""Batch extract tables from SI PDFs using FlowFigTabMiner pipeline.

Only runs:
  Step 1: TF-ID (Florence-2) — detect & crop tables from SI PDFs
  Step B: Table Pipeline (YOLO filter + TATR + PaddleOCR) — extract table data to CSV

Skips: figure pipeline, scheme parsing, local vars, global assembly, post-processing.
"""

import os, sys, glob, gc, re, argparse
import pypdfium2 as pdfium

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.parsing.active_area_detector import ActiveAreaDetector
from src.extraction.table.pipeline import TablePipeline
from src.extraction.common.content_recognizer import ContentRecognizer


SI_DIR = "data/ml_lifetime/papers/SI"
INTERMEDIATE_BASE = "data/intermediate"

# Keywords for filtering relevant tables (tR / organolithium kinetics)
TR_KEYWORDS = re.compile(
    r"residence\s*time|t\s*[_/]?\s*R\b|tR\b|flow\s*rate.*length|"
    r"inner\s*diameter.*length|温度.*停留",
    re.IGNORECASE,
)
LITHIUM_KEYWORDS = re.compile(
    r"lithium|lithiation|BuLi|PhLi|MeLi|halogen.*metal\s*exchange|"
    r"Br[\-–]Li|quench|methanol|electrophile|"
    r"aryllithium|organolithium|carbenoid|benzyne",
    re.IGNORECASE,
)


def get_page_context(pdf_doc, page_idx, n_context=1):
    """Get text from a page and its neighbors for table title matching."""
    texts = []
    for i in range(max(0, page_idx - n_context), min(len(pdf_doc), page_idx + n_context + 1)):
        texts.append(pdf_doc[i].get_textpage().get_text_bounded())
    return "\n".join(texts)


def is_relevant_table_page(pdf_doc, page_idx):
    """Check if a table page is about tR / organolithium kinetics."""
    context = get_page_context(pdf_doc, page_idx)
    has_tr = bool(TR_KEYWORDS.search(context))
    has_li = bool(LITHIUM_KEYWORDS.search(context))
    return has_tr and has_li


def process_one_si(pdf_path, detector, tab_pipeline, skip_tfid=False):
    """Run TF-ID + table pipeline on a single SI PDF."""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join(INTERMEDIATE_BASE, basename)
    tables_dir = os.path.join(intermediate_dir, "tables")

    # Step 1: TF-ID
    tables_exist = len(glob.glob(os.path.join(tables_dir, "*.png"))) > 0
    if skip_tfid and tables_exist:
        print(f"  [TF-ID] SKIP — {len(glob.glob(os.path.join(tables_dir, '*.png')))} crops already exist")
    else:
        print(f"  [TF-ID] Detecting tables...")
        detections = detector.process_pdf(pdf_path)
        saved = detector.save_crops(pdf_path, detections, intermediate_dir)
        n_tables = len([p for p in saved if "/tables/" in p])
        n_figures = len([p for p in saved if "/figures/" in p])
        print(f"  [TF-ID] {n_tables} tables, {n_figures} figures (figures ignored)")

    # Step B: Table Pipeline (only process tables/ directory)
    if not os.path.exists(tables_dir):
        print(f"  [Table] No tables detected, skip")
        return 0, []

    table_imgs = glob.glob(os.path.join(tables_dir, "*.png"))
    table_imgs = [f for f in table_imgs if "_body" not in f and "_crop" not in f]

    if not table_imgs:
        print(f"  [Table] No table images, skip")
        return 0, []

    # Filter: only keep tables on pages about tR + organolithium
    pdf_doc = pdfium.PdfDocument(pdf_path)
    relevant_imgs = []
    for t_img in table_imgs:
        # Extract page number from filename: page_5_table_0.png -> 5
        m = re.search(r"page_(\d+)_table", os.path.basename(t_img))
        if not m:
            relevant_imgs.append(t_img)
            continue
        page_num = int(m.group(1))
        page_idx = page_num - 1  # 0-indexed
        if is_relevant_table_page(pdf_doc, page_idx):
            relevant_imgs.append(t_img)
        else:
            print(f"    SKIP {os.path.basename(t_img)} (no tR/lithium keywords on page)")
    pdf_doc.close()

    if not relevant_imgs:
        print(f"  [Table] No relevant tables after filtering, skip")
        return 0, []

    print(f"  [Table] Processing {len(relevant_imgs)}/{len(table_imgs)} relevant table crops...")
    n_ok = 0
    for t_img in relevant_imgs:
        try:
            tab_pipeline.process_table(t_img, output_dir=tables_dir)
            n_ok += 1
        except Exception as e:
            print(f"    Error {os.path.basename(t_img)}: {e}")

    # Quality check on extracted CSVs — flag badly OCR'd tables
    csvs = glob.glob(os.path.join(tables_dir, "*_extracted.csv"))
    bad_tables = []
    for csv_path in csvs:
        try:
            with open(csv_path) as f:
                content = f.read()
            # Heuristic: count garbled cells (uppercase letters where digits expected)
            # e.g. "JUU", "C.0N", "U.2Z" — uppercase letter clusters in data rows
            lines = content.strip().split("\n")
            if len(lines) < 2:
                continue
            data_lines = lines[1:]  # skip header
            total_cells = 0
            bad_cells = 0
            for line in data_lines:
                cells = line.split(",")
                for cell in cells:
                    cell = cell.strip()
                    if not cell:
                        continue
                    total_cells += 1
                    # Flag: cell has mixed uppercase + digits in a way that suggests garbled OCR
                    n_upper = sum(1 for c in cell if c.isupper() and c not in "ETRCML")  # allow common header chars
                    n_digit = sum(1 for c in cell if c.isdigit() or c in ".-")
                    if n_upper >= 2 and n_digit >= 1:
                        bad_cells += 1
            if total_cells > 0:
                bad_ratio = bad_cells / total_cells
                if bad_ratio > 0.15:
                    tname = os.path.basename(csv_path)
                    bad_tables.append((tname, f"{bad_ratio:.0%} garbled cells ({bad_cells}/{total_cells})"))
                    print(f"    ⚠ BAD OCR: {tname} — {bad_ratio:.0%} garbled")
        except Exception:
            pass

    print(f"  [Table] {n_ok}/{len(relevant_imgs)} succeeded, {len(csvs)} CSVs generated")
    return len(csvs), bad_tables


def main():
    parser = argparse.ArgumentParser(description="Extract tables from SI PDFs")
    parser.add_argument("--skip-tfid", action="store_true",
                        help="Skip TF-ID if intermediate crops already exist")
    parser.add_argument("--only", type=str, default=None,
                        help="Process only SIs matching this substring (e.g. 'Nagaki_2009')")
    args = parser.parse_args()

    # Collect SI PDFs
    si_pdfs = sorted(glob.glob(os.path.join(SI_DIR, "*.pdf")))
    if args.only:
        si_pdfs = [p for p in si_pdfs if args.only in os.path.basename(p)]

    print(f"Found {len(si_pdfs)} SI PDFs to process\n")

    # Initialize models once (shared across all SIs)
    print("Loading models...")
    detector = ActiveAreaDetector()
    shared_content_rec = ContentRecognizer()
    tab_pipeline = TablePipeline(sequential_mode=True, content_recognizer=shared_content_rec)
    print("Models loaded.\n")

    total_csvs = 0
    all_bad_tables = {}
    results = {}

    for i, pdf_path in enumerate(si_pdfs):
        fname = os.path.basename(pdf_path)
        print(f"[{i+1}/{len(si_pdfs)}] {fname}")
        try:
            n, bad = process_one_si(pdf_path, detector, tab_pipeline, skip_tfid=args.skip_tfid)
            results[fname] = {"status": "ok", "csvs": n}
            total_csvs += n
            if bad:
                all_bad_tables[fname] = bad
        except Exception as e:
            print(f"  FAILED: {e}")
            results[fname] = {"status": "error", "error": str(e)}
        print()

    # Summary
    print("=" * 60)
    print(f"Processed: {len(si_pdfs)} SI PDFs")
    print(f"Total CSVs: {total_csvs}")
    ok = sum(1 for v in results.values() if v["status"] == "ok")
    print(f"Success: {ok}, Failed: {len(si_pdfs) - ok}")

    if any(v["status"] == "error" for v in results.values()):
        print("\nFailed papers:")
        for fname, v in results.items():
            if v["status"] == "error":
                print(f"  - {fname}: {v['error']}")

    # Write bad-OCR report for manual review
    if all_bad_tables:
        report_path = os.path.join(SI_DIR, "BAD_OCR_REPORT.md")
        with open(report_path, "w") as f:
            f.write("# Tables Needing Manual Review\n\n")
            f.write("These tables had poor OCR quality (>15% garbled cells),\n")
            f.write("likely due to cross-page splits or folded layouts.\n\n")
            for si_name, tables in sorted(all_bad_tables.items()):
                f.write(f"## {si_name}\n")
                for tname, reason in tables:
                    # Convert table name to page info
                    f.write(f"- `{tname}` — {reason}\n")
                f.write("\n")
        n_bad = sum(len(t) for t in all_bad_tables.values())
        print(f"\n⚠ {n_bad} tables flagged for manual review → {report_path}")


if __name__ == "__main__":
    main()
