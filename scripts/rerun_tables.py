"""Re-run ONLY the table stage (VLM transcription + MolNexTR) for papers whose
TF-ID crops already exist, e.g. after a change to the structure reader.
Follow with scripts/reassemble.py --rebuild-vars (table local_vars come from
the CSV head).

  set -a; source .env; set +a
  flowfigtabminer/bin/python scripts/rerun_tables.py "data/input/x/paper.pdf" [...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.providers.gemini import GeminiProvider
from src.pipeline.main import run_step_tables


def main(pdf_paths):
    provider = GeminiProvider()
    for pdf in pdf_paths:
        basename = os.path.splitext(os.path.basename(pdf))[0]
        idir = os.path.join("data", "intermediate", basename)
        if not os.path.isdir(idir):
            print(f"[tables] SKIP {basename}: no intermediate dir")
            continue
        print(f"\n[tables] === {basename} ===")
        run_step_tables(idir, provider)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
