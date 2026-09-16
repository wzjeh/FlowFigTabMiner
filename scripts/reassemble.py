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

Reads GEMINI_API_KEY from the environment. Writes the same
data/final_output/{basename}_{final,normalized}.{json,xlsx} as the full pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.llm.concurrency import configure_concurrency
from src.llm.config import load_llm_config
from src.llm.providers.gemini import GeminiProvider
from src.adjudication.global_assembly import GlobalAssembly
from src.adjudication.post_processor import PostProcessor
from src.pipeline.main import run_step44_global_vars, run_step45_local_vars

CONFIG_PATH = "config.yaml"


def main(pdf_paths, smiles_lookup=True, rebuild_vars=False):
    raw_cfg = yaml.safe_load(open(CONFIG_PATH))
    configure_concurrency(int(raw_cfg.get("llm", {}).get("max_concurrent", 5)))
    llm_cfg = load_llm_config(CONFIG_PATH)
    provider = GeminiProvider()

    for pdf_path in pdf_paths:
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        intermediate_dir = os.path.join("data", "intermediate", basename)
        if not os.path.isdir(intermediate_dir):
            print(f"[reassemble] SKIP {basename}: no intermediate dir")
            continue
        print(f"\n[reassemble] === {basename} ===")
        if rebuild_vars:
            gv = run_step44_global_vars(pdf_path, intermediate_dir, provider, llm_cfg, rebuild=True)
            run_step45_local_vars(pdf_path, intermediate_dir, provider, llm_cfg, rebuild=True, global_vars=gv)
        GlobalAssembly(llm=provider, llm_cfg=llm_cfg).run(
            pdf_path, intermediate_dir, force=True
        )
        PostProcessor().run(pdf_path, intermediate_dir, smiles_lookup=smiles_lookup)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    argv = sys.argv[1:]
    rebuild = "--rebuild-vars" in argv
    main([a for a in argv if a != "--rebuild-vars"], rebuild_vars=rebuild)
