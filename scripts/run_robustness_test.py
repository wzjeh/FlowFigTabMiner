"""
Robustness test: run 128 PDFs through the Cloud Run microservices pipeline.

Usage:
    python scripts/run_robustness_test.py \
        --input-dir data/input/rebust \
        --tfid-url https://tfid-service-xxx-uc.a.run.app \
        --figure-url https://figure-service-xxx-uc.a.run.app \
        --table-url https://table-service-xxx-uc.a.run.app \
        --output data/output/robustness_report.csv

The script:
  1. Uploads each PDF to GCS (gs://flowfigtabminer-data/uploads/{job_id}/input.pdf)
  2. Calls tfid-service /detect
  3. Concurrently calls figure-service /extract and table-service /extract for each crop
  4. Writes a per-PDF summary row to robustness_report.csv
"""
import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import uuid

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services._gcs import upload_file

GCS_BUCKET = "flowfigtabminer-data"
MAX_WORKERS = 8  # Concurrent figure/table calls per PDF


def upload_pdf(pdf_path: str, job_id: str) -> str:
    blob_name = f"uploads/{job_id}/input.pdf"
    upload_file(pdf_path, GCS_BUCKET, blob_name)
    return f"gs://{GCS_BUCKET}/{blob_name}"


def call_tfid(url: str, pdf_uri: str, job_id: str) -> dict:
    resp = requests.post(
        f"{url}/detect",
        json={"pdf_gcs_uri": pdf_uri, "job_id": job_id},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def call_figure(url: str, image_uri: str, job_id: str) -> dict:
    resp = requests.post(
        f"{url}/extract",
        json={"image_gcs_uri": image_uri, "job_id": job_id},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def call_table(url: str, image_uri: str, job_id: str) -> dict:
    resp = requests.post(
        f"{url}/extract",
        json={"image_gcs_uri": image_uri, "job_id": job_id},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def process_pdf(pdf_path, tfid_url, figure_url, table_url) -> dict:
    job_id = str(uuid.uuid4())[:8]
    pdf_name = os.path.basename(pdf_path)
    row = {
        "pdf": pdf_name,
        "job_id": job_id,
        "status": "error",
        "filter_status": "",
        "n_figures_detected": 0,
        "n_tables_detected": 0,
        "n_figures_success": 0,
        "n_tables_success": 0,
        "n_figures_no_data": 0,
        "n_tables_filtered_kw": 0,
        "n_errors": 0,
        "elapsed_s": 0,
        "error_detail": "",
    }

    t0 = time.time()
    try:
        # Step 1: Upload PDF
        pdf_uri = upload_pdf(pdf_path, job_id)

        # Step 2: TF-ID
        tfid_result = call_tfid(tfid_url, pdf_uri, job_id)
        tfid_status = tfid_result.get("status", "error")

        if tfid_status == "filtered":
            row["status"] = "filtered"
            row["filter_status"] = tfid_result.get("filter_reason", "")
            row["elapsed_s"] = round(time.time() - t0, 1)
            return row

        if tfid_status == "error":
            row["error_detail"] = "tfid-service error"
            row["elapsed_s"] = round(time.time() - t0, 1)
            return row

        figures = tfid_result.get("figures", [])
        tables = tfid_result.get("tables", [])
        row["n_figures_detected"] = len(figures)
        row["n_tables_detected"] = len(tables)

        # Step 3: Concurrent figure + table extraction
        futures = []
        labels = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for fig_uri in figures:
                futures.append(ex.submit(call_figure, figure_url, fig_uri, job_id))
                labels.append("figure")
            for tab_uri in tables:
                futures.append(ex.submit(call_table, table_url, tab_uri, job_id))
                labels.append("table")

            for future, label in zip(futures, labels):
                try:
                    res = future.result()
                    status = res.get("status", "error")
                    if label == "figure":
                        if status == "success":
                            row["n_figures_success"] += 1
                        elif status in ("no_data_points", "no_scatter_points"):
                            row["n_figures_no_data"] += 1
                        elif status == "error":
                            row["n_errors"] += 1
                    else:
                        if status == "success":
                            row["n_tables_success"] += 1
                        elif status == "filtered_keywords":
                            row["n_tables_filtered_kw"] += 1
                        elif status == "error":
                            row["n_errors"] += 1
                except Exception as e:
                    row["n_errors"] += 1

        row["status"] = "success"
    except Exception as e:
        row["status"] = "error"
        row["error_detail"] = str(e)

    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def main():
    parser = argparse.ArgumentParser(description="Robustness test for FlowFigTabMiner Cloud Run services")
    parser.add_argument("--input-dir", default="data/input/rebust", help="Directory with 128 test PDFs")
    parser.add_argument("--tfid-url", required=True, help="tfid-service Cloud Run URL")
    parser.add_argument("--figure-url", required=True, help="figure-service Cloud Run URL")
    parser.add_argument("--table-url", required=True, help="table-service Cloud Run URL")
    parser.add_argument("--output", default="data/output/robustness_report.csv", help="Output CSV path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel PDFs processed simultaneously")
    args = parser.parse_args()

    pdf_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDFs in {args.input_dir}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fieldnames = [
        "pdf", "job_id", "status", "filter_status",
        "n_figures_detected", "n_tables_detected",
        "n_figures_success", "n_tables_success",
        "n_figures_no_data", "n_tables_filtered_kw",
        "n_errors", "elapsed_s", "error_detail",
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = {
            ex.submit(
                process_pdf,
                os.path.join(args.input_dir, pdf),
                args.tfid_url,
                args.figure_url,
                args.table_url,
            ): pdf
            for pdf in pdf_files
        }
        for i, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            pdf = future_map[future]
            try:
                row = future.result()
            except Exception as e:
                row = {"pdf": pdf, "status": "error", "error_detail": str(e)}
            results.append(row)
            print(f"[{i}/{len(pdf_files)}] {pdf} -> {row.get('status')} ({row.get('elapsed_s', '?')}s)")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    filtered = sum(1 for r in results if r.get("status") == "filtered")
    errors = sum(1 for r in results if r.get("status") == "error")
    print(f"\n=== Robustness Test Summary ===")
    print(f"  Total PDFs : {total}")
    print(f"  Success    : {success}")
    print(f"  Filtered   : {filtered} (not flow chemistry)")
    print(f"  Errors     : {errors}")
    print(f"  Report     : {args.output}")


if __name__ == "__main__":
    main()
