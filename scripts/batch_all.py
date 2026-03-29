"""
Full-pipeline batch runner for organolithium PDFs.

Each PDF is run as an isolated subprocess so a crash in one paper does not
kill the whole batch. Progress is checkpointed after each PDF.
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime


PDF_DIR = "data/input/organolithium"
CHECKPOINT = "data/batch_progress.json"
VENV_PYTHON = sys.executable
PIPELINE = "src/pipeline/main.py"
PER_PDF_TIMEOUT = 900


def _safe_text(value) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _safe_print(*parts) -> None:
    print(" ".join(_safe_text(part) for part in parts))


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": [], "failed": [], "started_at": None, "last_updated": None}


def _save_checkpoint(cp: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    cp["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)


def get_all_pdfs() -> list[str]:
    if not os.path.isdir(PDF_DIR):
        print(f"[ERROR] PDF directory not found: {PDF_DIR}")
        sys.exit(1)
    pdfs = sorted(
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf") and not f.startswith("._")
    )
    if not pdfs:
        print(f"[WARN] No PDFs found in {PDF_DIR}")
    return pdfs


def run_one_pdf(pdf_path: str, skip_tfid: bool, force_assembly: bool, stop_before_llm: bool) -> bool:
    cmd = [VENV_PYTHON, PIPELINE, pdf_path]
    if skip_tfid:
        cmd.append("--skip-tfid")
    if force_assembly:
        cmd.append("--force-assembly")
    if stop_before_llm:
        cmd.append("--stop-before-llm")

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    log_path = os.path.join("data/intermediate", basename, "pipeline.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=PER_PDF_TIMEOUT,
                cwd=os.getcwd(),
            )
        if proc.returncode == 0:
            return True
        print(f"  [FAIL] Exit code {proc.returncode} - see {log_path}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Exceeded {PER_PDF_TIMEOUT}s - killed. See {log_path}")
        return False
    except Exception as e:
        print(f"  [ERROR] Subprocess launch failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner with subprocess isolation and checkpointing."
    )
    parser.add_argument("--skip-tfid", action="store_true", help="Pass --skip-tfid to each PDF run")
    parser.add_argument("--force-assembly", action="store_true", help="Force re-run LLM assembly")
    parser.add_argument("--resume", action="store_true", help="Skip PDFs already marked done")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start from scratch")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N PDFs")
    parser.add_argument(
        "--stop-before-llm",
        action="store_true",
        help="Run only Steps 1-3.5 and stop before LLM-dependent stages",
    )
    args = parser.parse_args()

    cp = _load_checkpoint()
    if args.reset:
        cp = {"done": [], "failed": [], "started_at": None, "last_updated": None}
        _save_checkpoint(cp)
        print("[Batch] Checkpoint cleared.")

    if cp["started_at"] is None:
        cp["started_at"] = datetime.now().isoformat()

    all_pdfs = get_all_pdfs()
    if args.resume:
        done_set = set(cp.get("done", []))
        pdfs = [p for p in all_pdfs if os.path.basename(p) not in done_set]
        print(f"[Batch] Resume mode: {len(done_set)} already done, {len(pdfs)} remaining.")
    else:
        pdfs = all_pdfs
        print(f"[Batch] {len(pdfs)} PDFs to process.")

    if args.limit is not None:
        pdfs = pdfs[: args.limit]
        print(f"[Batch] Limit enabled: processing {len(pdfs)} PDFs.")

    if not pdfs:
        print("[Batch] Nothing to do.")
    else:
        _safe_print(f"[Batch] Timeout per PDF: {PER_PDF_TIMEOUT}s")
        _safe_print("[Batch] Log per PDF: data/intermediate/<name>/pipeline.log")
        _safe_print(f"[Batch] Checkpoint: {CHECKPOINT}")

    for i, pdf_path in enumerate(pdfs, 1):
        name = os.path.basename(pdf_path)
        ts = datetime.now().strftime("%H:%M:%S")
        _safe_print(f"[{i:>3}/{len(pdfs)}] {ts}  {name}")

        ok = run_one_pdf(pdf_path, args.skip_tfid, args.force_assembly, args.stop_before_llm)
        if ok:
            _safe_print("         [OK] done")
            if name not in cp["done"]:
                cp["done"].append(name)
            if name in cp.get("failed", []):
                cp["failed"].remove(name)
        else:
            _safe_print("         [FAIL] failed")
            if name not in cp.get("failed", []):
                cp.setdefault("failed", []).append(name)

        _save_checkpoint(cp)

    _safe_print("\n" + "=" * 55)
    _safe_print("Batch complete.")
    _safe_print(f"  Done   : {len(cp['done'])}")
    _safe_print(f"  Failed : {len(cp.get('failed', []))}")
    if cp.get("failed"):
        _safe_print("\nFailed PDFs:")
        for failed_name in cp["failed"]:
            _safe_print(f"  - {failed_name}")
        _safe_print("\nTo retry failed PDFs:")
        _safe_print(f"  {VENV_PYTHON} scripts/batch_all.py --resume")
    _safe_print("\nTo build combined Excel from results:")
    _safe_print(f"  {VENV_PYTHON} scripts/batch_step5_6.py --combine-only")


if __name__ == "__main__":
    main()
