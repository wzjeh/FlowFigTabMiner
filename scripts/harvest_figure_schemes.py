"""Run only Step 3.5b (figure-scheme harvest) for papers whose figure / table
stages already ran, then merge the labels into compound_pool.json.  Follow
with scripts/reassemble.py so the pool reaches the assembly prompts and the
post-processor.

  set -a; source .env; set +a
  flowfigtabminer/bin/python scripts/harvest_figure_schemes.py "data/input/x/paper.pdf" [...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.config import load_vlm_config
from src.llm.providers.gemini import GeminiProvider
from src.pipeline.main import run_step35_figure_schemes

CONFIG_PATH = "config.yaml"


def main(pdf_paths):
    provider, vlm_cfg = GeminiProvider(), load_vlm_config(CONFIG_PATH)
    for pdf in pdf_paths:
        basename = os.path.splitext(os.path.basename(pdf))[0]
        idir = os.path.join("data", "intermediate", basename)
        if not os.path.isdir(idir):
            print(f"[harvest] SKIP {basename}: no intermediate dir")
            continue
        print(f"\n[harvest] === {basename} ===")
        fig_pool = run_step35_figure_schemes(idir, provider, vlm_cfg)
        pool_path = os.path.join(idir, "compound_pool.json")
        pools = json.load(open(pool_path)) if os.path.exists(pool_path) else {}
        if "reactant_pool" not in pools:
            pools = {"reactant_pool": {}, "product_pool": {}, "compound_pool": dict(pools)}
        taken = set(pools["reactant_pool"]) | set(pools["product_pool"])
        for k, v in fig_pool.items():
            if k not in taken:
                pools["compound_pool"].setdefault(k, v)
        if any(pools.values()):
            json.dump(pools, open(pool_path, "w"), indent=2)
            print(f"[harvest] compound_pool.json: {len(pools['compound_pool'])} compound labels")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
