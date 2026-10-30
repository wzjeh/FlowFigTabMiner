"""Re-run ONLY Step 5 (Global Assembly) + Step 6 (Post-processing) for one or
more papers, reusing the existing figure/table evidence in data/intermediate/.

This skips the slow extraction stages (TF-ID, YOLO, TATR, MolNexTR, OCR) so an
assembly/prompt change can be validated in seconds-per-paper instead of the
~10 min/paper a full --force-assembly run costs (table OCR dominates).

Usage:
  set -a; source .env; set +a
  flowfigtabminer/bin/python scripts/reassemble.py [--rebuild-vars] \
      "data/input/test_10/Bohara ... .pdf" ["data/input/.../Other.pdf" ...]

--rebuild-vars also re-runs Step 4.5 (LocalVarsBuilder) after clearing the
cached local_vars/*.json, so prompt/context changes in that stage are
validated too (adds ~1 LLM call per source).

Uses llm.provider from config.yaml (its API key from the environment). Writes the same
data/final_output/{basename}_{final,normalized}.{json,xlsx} as the full pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.api import Paper, assemble, build_llm, context, normalize


def main(pdf_paths, smiles_lookup=True, rebuild_vars=False):
    llm = build_llm(no_vlm=True)

    for pdf_path in pdf_paths:
        paper = Paper(pdf_path)
        if not os.path.isdir(paper.work_dir):
            print(f"[reassemble] SKIP {paper.paper_id}: no intermediate dir")
            continue
        print(f"\n[reassemble] === {paper.paper_id} ===")
        if rebuild_vars:
            context(paper, llm, rebuild=True)
        assemble(paper, llm, force=True)
        normalize(paper, smiles_lookup=smiles_lookup)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    argv = sys.argv[1:]
    rebuild = "--rebuild-vars" in argv
    main([a for a in argv if a != "--rebuild-vars"], rebuild_vars=rebuild)
