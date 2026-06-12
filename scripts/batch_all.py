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
import time
import argparse
import subprocess
import traceback
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
PDF_DIR        = "data/input/organolithium"
CHECKPOINT     = "data/batch_progress.json"
VENV_PYTHON    = "flowfigtabminer/bin/python"
PIPELINE       = "src/pipeline/main.py"
PER_PDF_TIMEOUT = 2400  # seconds (40 min) — kill subprocess if it exceeds this
                        # (guards against malformed / huge-SI PDFs hanging the batch)
                        # Raised 1200->2400 (issue #14): table-stage MolNexTR runs
                        # ~22s per molecule box serially, so structure-dense tables
                        # (Nagaki 84 boxes = 50 min) legitimately exceed 20 min.
                        # SHORT-TERM guard so research runs don't get killed mid-paper;
                        # LONG-TERM = dynamic budget scaled by detected molecule_boxes.
EXIT_SKIPPED    = 3     # main.py exit code for a pre-filter skip (review/non-flow)

# ─────────────────────────────────────────────────────────────────────────────


def _new_checkpoint() -> dict:
    return {
        "done": [],
        "skipped": [],          # pre-filter skips (review / non-flow), exit 3
        "failed": [],           # crashes / non-zero exit
        "timeout_failed": [],   # [{"pdf": name, "elapsed": <sec>}] — killed by watchdog
        "started_at": None,
        "last_updated": None,
    }


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT) as f:
                cp = json.load(f)
            # Fail-safe: backfill any missing bucket so an old checkpoint
            # (pre-skipped/timeout_failed) loads without KeyError.
            for k, v in _new_checkpoint().items():
                cp.setdefault(k, v)
            return cp
        except Exception:
            pass
    return _new_checkpoint()


def _save_checkpoint(cp: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    cp["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)


def _forget(cp: dict, name: str) -> None:
    """Remove ``name`` from every status bucket so re-runs are reclassified
    cleanly (buckets stay mutually exclusive)."""
    for k in ("done", "skipped", "failed"):
        if name in cp.get(k, []):
            cp[k].remove(name)
    cp["timeout_failed"] = [e for e in cp.get("timeout_failed", []) if e.get("pdf") != name]


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


def run_one_pdf(pdf_path: str, skip_tfid: bool, force_assembly: bool):
    """
    Run the full pipeline for one PDF in an isolated subprocess.

    Returns ``(status, elapsed)`` where status is one of
    ``"ok" | "skipped" | "fail" | "timeout"`` and elapsed is wall-clock
    seconds.  On timeout the watchdog kills the subprocess and elapsed is
    ≈PER_PDF_TIMEOUT (recorded so "hung 1200s" is distinguishable from a
    fast-failing case at review time — timing.json is never written when
    the child is killed mid-run).
    """
    cmd = [VENV_PYTHON, PIPELINE, pdf_path]
    if skip_tfid:
        cmd.append("--skip-tfid")
    if force_assembly:
        cmd.append("--force-assembly")

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    log_path = os.path.join("data/intermediate", basename, "pipeline.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t0 = time.perf_counter()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=PER_PDF_TIMEOUT,
                cwd=os.getcwd(),
            )
        elapsed = round(time.perf_counter() - t0, 1)
        if proc.returncode == 0:
            return "ok", elapsed
        elif proc.returncode == EXIT_SKIPPED:
            print(f"  [SKIP] Pre-filter rejected (review/non-flow) — see {log_path}")
            return "skipped", elapsed
        else:
            print(f"  [FAIL] Exit code {proc.returncode} — see {log_path}")
            return "fail", elapsed

    except subprocess.TimeoutExpired:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [TIMEOUT] Exceeded {PER_PDF_TIMEOUT}s — killed. See {log_path}")
        return "timeout", elapsed
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [ERROR] Subprocess launch failed: {e}")
        traceback.print_exc()
        return "fail", elapsed


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
    parser.add_argument("--pdf-dir",        default=None,
                        help="Override the PDF directory (default: data/input/organolithium)")
    args = parser.parse_args()

    if args.pdf_dir:
        global PDF_DIR
        PDF_DIR = args.pdf_dir

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    cp = _load_checkpoint()
    if args.reset:
        cp = _new_checkpoint()
        _save_checkpoint(cp)
        print("[Batch] Checkpoint cleared.")

    if cp["started_at"] is None:
        cp["started_at"] = datetime.now().isoformat()

    # ── PDF list ─────────────────────────────────────────────────────────────
    all_pdfs = get_all_pdfs()
    if args.resume:
        # Skip papers already resolved as done OR deliberately skipped by the
        # pre-filter (re-running them would just skip again).  failed /
        # timeout_failed are NOT in this set, so they get retried.
        handled = set(cp.get("done", [])) | set(cp.get("skipped", []))
        pdfs = [p for p in all_pdfs if os.path.basename(p) not in handled]
        print(f"[Batch] Resume mode: {len(handled)} already handled "
              f"({len(cp.get('done', []))} done, {len(cp.get('skipped', []))} skipped), "
              f"{len(pdfs)} remaining.")
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

        status, elapsed = run_one_pdf(pdf_path, args.skip_tfid, args.force_assembly)

        _forget(cp, name)  # reclassify cleanly on re-run (mutually exclusive buckets)
        if status == "ok":
            print(f"         ✓ done ({elapsed}s)")
            cp["done"].append(name)
        elif status == "skipped":
            print(f"         ⊘ skipped ({elapsed}s)")
            cp["skipped"].append(name)
        elif status == "timeout":
            print(f"         ⏱ timeout ({elapsed}s)")
            cp["timeout_failed"].append({"pdf": name, "elapsed": elapsed})
        else:  # fail
            print(f"         ✗ failed ({elapsed}s)")
            cp["failed"].append(name)

        _save_checkpoint(cp)

    # ── Summary ───────────────────────────────────────────────────────────────
    timeouts = cp.get("timeout_failed", [])
    print(f"\n{'='*55}")
    print(f"Batch complete.")
    print(f"  Done    : {len(cp['done'])}")
    print(f"  Skipped : {len(cp.get('skipped', []))}  (pre-filter: review/non-flow)")
    print(f"  Failed  : {len(cp.get('failed', []))}")
    print(f"  Timeout : {len(timeouts)}  (killed after {PER_PDF_TIMEOUT}s — malformed/huge-SI)")
    if cp.get("failed"):
        print(f"\nFailed PDFs:")
        for f in cp["failed"]:
            print(f"  - {f}")
    if timeouts:
        print(f"\nTimed-out PDFs (hung — inspect SI size / table count):")
        for e in timeouts:
            print(f"  - {e.get('pdf')}  ({e.get('elapsed')}s)")
    if cp.get("failed") or timeouts:
        print(f"\nTo retry failed/timed-out PDFs:")
        print(f"  flowfigtabminer/bin/python scripts/batch_all.py --resume")
    print(f"\nTo build combined Excel from results:")
    print(f"  flowfigtabminer/bin/python scripts/batch_step5_6.py --combine-only")


if __name__ == "__main__":
    main()
