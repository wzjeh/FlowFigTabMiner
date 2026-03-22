"""
Full-pipeline batch runner for 200 organolithium PDFs.

Each PDF is run as an isolated subprocess — if a PDF causes a Python crash
(segfault from YOLO/OpenCV/PaddleOCR), it only kills that subprocess,
not the entire batch.

Progress is checkpointed to data/batch_progress.json after each PDF,
so if the batch is interrupted you can resume from where it stopped.

Usage:
    # Full pipeline (Steps 1–6):
    flowfigtabminer/bin/python scripts/batch_all.py

    # Skip TF-ID (crops already extracted):
    flowfigtabminer/bin/python scripts/batch_all.py --skip-tfid

    # Force re-run LLM even if _final.json exists:
    flowfigtabminer/bin/python scripts/batch_all.py --force-assembly

    # Resume interrupted batch (skips already-done PDFs):
    flowfigtabminer/bin/python scripts/batch_all.py --resume

    # After all PDFs, rebuild combined Excel from existing results:
    flowfigtabminer/bin/python scripts/batch_step5_6.py --combine-only
"""

import os
import sys
import json
import argparse
import subprocess
import traceback
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
PDF_DIR        = "data/input/organolithium"
CHECKPOINT     = "data/batch_progress.json"
VENV_PYTHON    = "flowfigtabminer/bin/python"
PIPELINE       = "src/pipeline/main.py"
PER_PDF_TIMEOUT = 900   # seconds — kill subprocess if it exceeds this

# ─────────────────────────────────────────────────────────────────────────────


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": [], "failed": [], "started_at": None, "last_updated": None}


def _save_checkpoint(cp: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    cp["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)


def get_all_pdfs() -> list[str]:
    if not os.path.isdir(PDF_DIR):
        print(f"[ERROR] PDF directory not found: {PDF_DIR}")
        print(f"        Create it and place your PDFs there, then re-run.")
        sys.exit(1)
    pdfs = sorted([
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ])
    if not pdfs:
        print(f"[WARN] No PDFs found in {PDF_DIR}")
    return pdfs


def run_one_pdf(pdf_path: str, skip_tfid: bool, force_assembly: bool) -> bool:
    """
    Run the full pipeline for one PDF in an isolated subprocess.
    Returns True on success (exit code 0), False on failure or timeout.
    """
    cmd = [VENV_PYTHON, PIPELINE, pdf_path]
    if skip_tfid:
        cmd.append("--skip-tfid")
    if force_assembly:
        cmd.append("--force-assembly")

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    log_path = os.path.join("data/intermediate", basename, "pipeline.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=PER_PDF_TIMEOUT,
                cwd=os.getcwd(),
            )
        if proc.returncode == 0:
            return True
        else:
            print(f"  [FAIL] Exit code {proc.returncode} — see {log_path}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Exceeded {PER_PDF_TIMEOUT}s — killed. See {log_path}")
        return False
    except Exception as e:
        print(f"  [ERROR] Subprocess launch failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Full-pipeline batch runner with subprocess isolation + checkpoint."
    )
    parser.add_argument("--skip-tfid",      action="store_true",
                        help="Pass --skip-tfid to each PDF's pipeline run")
    parser.add_argument("--force-assembly", action="store_true",
                        help="Force LLM re-run even if _final.json exists")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip PDFs already marked done in checkpoint")
    parser.add_argument("--reset",          action="store_true",
                        help="Clear checkpoint and start from scratch")
    args = parser.parse_args()

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    cp = _load_checkpoint()
    if args.reset:
        cp = {"done": [], "failed": [], "started_at": None, "last_updated": None}
        _save_checkpoint(cp)
        print("[Batch] Checkpoint cleared.")

    if cp["started_at"] is None:
        cp["started_at"] = datetime.now().isoformat()

    # ── PDF list ─────────────────────────────────────────────────────────────
    all_pdfs = get_all_pdfs()
    if args.resume:
        done_set = set(cp.get("done", []))
        pdfs = [p for p in all_pdfs if os.path.basename(p) not in done_set]
        print(f"[Batch] Resume mode: {len(done_set)} already done, {len(pdfs)} remaining.")
    else:
        pdfs = all_pdfs
        print(f"[Batch] {len(pdfs)} PDFs to process.")

    if not pdfs:
        print("[Batch] Nothing to do.")
    else:
        print(f"[Batch] Timeout per PDF: {PER_PDF_TIMEOUT}s  |  Log: data/intermediate/<name>/pipeline.log")
        print(f"[Batch] Checkpoint: {CHECKPOINT}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, pdf_path in enumerate(pdfs, 1):
        name = os.path.basename(pdf_path)
        ts   = datetime.now().strftime("%H:%M:%S")
        print(f"[{i:>3}/{len(pdfs)}] {ts}  {name}")

        ok = run_one_pdf(pdf_path, args.skip_tfid, args.force_assembly)

        if ok:
            print(f"         ✓ done")
            if name not in cp["done"]:
                cp["done"].append(name)
            if name in cp.get("failed", []):
                cp["failed"].remove(name)
        else:
            print(f"         ✗ failed")
            if name not in cp.get("failed", []):
                cp.setdefault("failed", []).append(name)

        _save_checkpoint(cp)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Batch complete.")
    print(f"  Done   : {len(cp['done'])}")
    print(f"  Failed : {len(cp.get('failed', []))}")
    if cp.get("failed"):
        print(f"\nFailed PDFs:")
        for f in cp["failed"]:
            print(f"  - {f}")
        print(f"\nTo retry failed PDFs:")
        print(f"  flowfigtabminer/bin/python scripts/batch_all.py --resume")
    print(f"\nTo build combined Excel from results:")
    print(f"  flowfigtabminer/bin/python scripts/batch_step5_6.py --combine-only")


if __name__ == "__main__":
    main()
