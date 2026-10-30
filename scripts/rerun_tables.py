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

from src.pipeline.api import Paper, build_llm, tables


def main(pdf_paths):
    llm = build_llm(no_vlm=True)
    for pdf in pdf_paths:
        paper = Paper(pdf)
        if not os.path.isdir(paper.work_dir):
            print(f"[tables] SKIP {paper.paper_id}: no intermediate dir")
            continue
        print(f"\n[tables] === {paper.paper_id} ===")
        tables(paper, llm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
