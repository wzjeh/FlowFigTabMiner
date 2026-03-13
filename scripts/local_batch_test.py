"""
Local robustness batch tester.
Runs src/pipeline/main.py on N PDFs from data/input/rebust/,
records metrics to data/robustness/batch_results.csv.
"""
import os
import sys
import glob
import json
import time
import re
import subprocess
import argparse
import csv

# Run from project root
sys.path.insert(0, os.getcwd())

REBUST_DIR = "data/input/rebust"
OUTPUT_DIR = "data/robustness"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "batch_results.csv")
PYTHON = "flowfigtabminer/bin/python"


def count_records(basename):
    """Count records in final output JSON."""
    path = os.path.join("data/final_output", f"{basename}_final.json")
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            content = f.read().strip()
        # Try to parse as JSON
        data = json.loads(content)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and "dataset" in data:
            return len(data["dataset"])
        return 1  # non-empty but not a list
    except Exception:
        return -1  # parse error


def parse_metrics(stdout, basename):
    """Extract n_figures and n_tables from intermediate dir (most reliable)."""
    figures_dir = os.path.join("data/intermediate", basename, "figures")
    tables_dir = os.path.join("data/intermediate", basename, "tables")
    n_figures = len(glob.glob(os.path.join(figures_dir, "*.png"))) if os.path.exists(figures_dir) else 0
    n_tables = len(glob.glob(os.path.join(tables_dir, "*.png"))) if os.path.exists(tables_dir) else 0
    return n_figures, n_tables


def run_pdf(pdf_path):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"{'='*60}")

    start = time.time()
    cmd = [PYTHON, "src/pipeline/main.py", pdf_path, "--skip-tfid"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900
        )
        elapsed = time.time() - start
        stdout = result.stdout + result.stderr

        n_figures, n_tables = parse_metrics(stdout, basename)
        n_records = count_records(basename)

        if result.returncode != 0:
            status = "error"
            error = (result.stderr or result.stdout)[-500:].replace("\n", " ")
        elif n_records == 0:
            status = "empty"
            error = ""
        elif n_records == -1:
            status = "json_error"
            error = "Final JSON parse failed"
        else:
            status = "success"
            error = ""

        print(f"  -> status={status}, figures={n_figures}, tables={n_tables}, records={n_records}, time={elapsed:.0f}s")
        return {
            "pdf": basename,
            "n_figures": n_figures,
            "n_tables": n_tables,
            "n_records": n_records,
            "runtime_s": round(elapsed, 1),
            "status": status,
            "error": error,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  -> TIMEOUT after {elapsed:.0f}s")
        return {
            "pdf": basename,
            "n_figures": 0,
            "n_tables": 0,
            "n_records": 0,
            "runtime_s": round(elapsed, 1),
            "status": "timeout",
            "error": "Process timed out after 600s",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "pdf": basename,
            "n_figures": 0,
            "n_tables": 0,
            "n_records": 0,
            "runtime_s": round(elapsed, 1),
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Local batch robustness test")
    parser.add_argument("--n", type=int, default=10, help="Number of PDFs to test")
    parser.add_argument("--dir", default=REBUST_DIR, help="Directory containing PDFs")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect PDFs, sorted by name
    pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))[:args.n]
    if not pdfs:
        print(f"No PDFs found in {args.dir}")
        return

    print(f"Found {len(pdfs)} PDFs to test (up to {args.n}):")
    for p in pdfs:
        print(f"  {os.path.basename(p)}")

    results = []
    for pdf_path in pdfs:
        row = run_pdf(pdf_path)
        results.append(row)

        # Save incrementally after each PDF
        fieldnames = ["pdf", "n_figures", "n_tables", "n_records", "runtime_s", "status", "error"]
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — {len(results)} PDFs processed")
    success = [r for r in results if r["status"] == "success"]
    print(f"  Success: {len(success)}/{len(results)}")
    print(f"  Total records extracted: {sum(r['n_records'] for r in success)}")
    print(f"  Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
