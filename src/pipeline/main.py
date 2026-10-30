"""Command-line entry point.  The pipeline itself lives in ``src.pipeline.api``
(one function per step + ``extract``); this file parses arguments, takes the
single-instance lock and maps the result to an exit code.

Run::

    flowfigtabminer paper.pdf                                  # whole pipeline
    flowfigtabminer paper.pdf --steps prefilter,detect         # only these steps
    flowfigtabminer paper.pdf --steps assemble,normalize --redo
    flowfigtabminer paper.pdf --provider gemini                # or claude / openai
    flowfigtabminer --dir data/input/corpus                    # one subprocess per PDF
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys

# Ensure src is importable from project root
sys.path.insert(0, os.getcwd())

logger = logging.getLogger(__name__)

# Exit code emitted when a PDF is skipped by the Step 0 pre-filter (review /
# non-flow paper).  batch_all.py maps this code to the ``skipped`` bucket so
# a deliberately-skipped paper isn't counted as a success (0) or a crash.
EXIT_SKIPPED = 3
# A step failed, or a requested step could not run because an earlier one never did.
EXIT_FAILED = 1


def _run_single(args) -> str:
    """Process ONE PDF under the single-instance lock.

    The lock lives here (not in ``main``) so the batch driver can spawn
    subprocesses that each lock themselves and serialise cleanly.
    Returns the ``extract`` status (ok / skipped / failed / blocked).
    """
    from src.pipeline.api import STEPS, extract
    from src.pipeline.single_instance import single_instance_lock

    steps = tuple(s.strip() for s in args.steps.split(",")) if args.steps else STEPS
    with single_instance_lock("flowfigtabminer-pipeline"):
        summary = extract(
            args.pdf_path, work_root=args.work_root, out_dir=args.out_dir,
            provider=args.provider, model=args.model, steps=steps, redo=args.redo,
            use_prefilter=not args.no_prefilter, smiles_lookup=not args.no_smiles_lookup, no_vlm=args.no_vlm,
        )
    if summary["status"] == "ok":
        print("\n=== Pipeline Complete ===")
    return summary["status"]


def _run_batch(args) -> None:
    """Process every PDF in ``args.dir`` in a SEPARATE subprocess, serially.

    Subprocess isolation is the durable fix for cross-PDF thread/memory
    accumulation: the in-process loop leaked C++ thread pools (paddle /
    torch / ultralytics) that ``del`` + gc cannot reclaim, pushing the
    load average from ~10 on the first PDF to 44 on the second.  Running
    each PDF as its own ``python -m src.pipeline.main <pdf>`` process lets
    the OS reclaim *everything* on exit — every PDF starts from a clean
    slate.

    The parent (this driver) does NOT take the lock; each child does, so
    the children serialise on the lock and two batches can't run heavy
    work concurrently either.
    """
    pdfs = sorted(glob.glob(os.path.join(glob.escape(args.dir), "*.pdf")))
    if not pdfs:
        print(f"Error: no *.pdf found in {args.dir}")
        return

    passthrough = ["--work-root", args.work_root, "--out-dir", args.out_dir]
    for flag in ("steps", "provider", "model"):
        if getattr(args, flag):
            passthrough += [f"--{flag}", getattr(args, flag)]
    if args.redo:
        passthrough.append("--redo")
    if args.no_smiles_lookup:
        passthrough.append("--no-smiles-lookup")
    if args.no_prefilter:
        passthrough.append("--no-prefilter")
    if args.no_vlm:
        passthrough.append("--no-vlm")

    for idx, pdf in enumerate(pdfs, 1):
        print(f"\n########## [{idx}/{len(pdfs)}] {os.path.basename(pdf)} (subprocess) ##########")
        cmd = [sys.executable, "-m", "src.pipeline.main", pdf] + passthrough
        result = subprocess.run(cmd)
        if result.returncode == EXIT_SKIPPED:
            print(f"[batch] {os.path.basename(pdf)} skipped by pre-filter (review/non-flow)")
        elif result.returncode != 0:
            print(f"[batch] {os.path.basename(pdf)} exited with code {result.returncode}")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="FlowFigTabMiner Unified Pipeline")
    parser.add_argument("pdf_path", nargs="?", help="Path to a single input PDF")
    parser.add_argument("--dir", help="Process every *.pdf in this directory, "
                                      "each in its own subprocess (serial)")
    parser.add_argument("--steps", help="Comma-separated subset of: prefilter, detect, figures, tables, "
                                         "schemes, context, assemble, normalize (default: all)")
    parser.add_argument("--redo", action="store_true",
                        help="Re-run steps already marked ok in steps.json (and force the assembly LLM call)")
    parser.add_argument("--work-root", default="data/intermediate",
                        help="Intermediate files go to WORK_ROOT/<pdf name> (default: data/intermediate)")
    parser.add_argument("--out-dir", default="data/final_output",
                        help="Final JSON / xlsx directory (default: data/final_output)")
    parser.add_argument("--provider", choices=("gemini", "claude", "openai"),
                        help="LLM vendor for every call (default: llm.provider in config.yaml)")
    parser.add_argument("--model", help="Model id for every call (default: llm.models[provider])")
    parser.add_argument("--no-smiles-lookup", action="store_true",
                        help="Disable PubChem name→SMILES lookup in Step 6 (default: enabled)")
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Disable Step 0 pre-filter (review/non-flow skip; default: enabled)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM inspection hooks (paper modules 6 & 12). "
                             "Adjudication still uses the LLM.")
    args = parser.parse_args()

    if args.dir:
        # Batch driver: one subprocess per PDF, no lock here (children lock).
        _run_batch(args)
    elif args.pdf_path:
        if not os.path.exists(args.pdf_path):
            print(f"Error: PDF not found at {args.pdf_path}")
            return
        status = _run_single(args)
        if status == "skipped":
            sys.exit(EXIT_SKIPPED)
        if status != "ok":
            sys.exit(EXIT_FAILED)
    else:
        print("Error: provide a PDF path or --dir <directory>")


if __name__ == "__main__":
    main()
