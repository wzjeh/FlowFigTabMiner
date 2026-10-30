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

from src.pipeline.api import Paper, build_llm, figure_schemes


def main(pdf_paths):
    llm = build_llm(no_vlm=True)
    for pdf in pdf_paths:
        paper = Paper(pdf)
        idir = paper.work_dir
        if not os.path.isdir(idir):
            print(f"[harvest] SKIP {paper.paper_id}: no intermediate dir")
            continue
        print(f"\n[harvest] === {paper.paper_id} ===")
        fig_pool = figure_schemes(paper, llm)
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
