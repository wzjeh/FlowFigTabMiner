"""Precompute the three examples shown on the demo page (docker/webapp).

  set -a; source .env; set +a
  flowfigtabminer/bin/python scripts/build_web_examples.py

Figure: Figure 2 of the same paper (a yield heatmap over temperature and
residence time) and, as figure2, a two-axis scatter / line plot (the Cloud Run
demo's example figure); table: the Cloud Run demo's example table.  Both run once
through the same functions the page uses for a visitor's upload.  Paper: the current
pipeline output for Nagaki et al. 2007 (Chem. Asian J.), packaged as is; the PDF
itself is not copied.  Output: docker/webapp/examples/{figure,table,paper}/.
"""
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docker", "webapp"))

import jobs  # noqa: E402

OUT = os.path.join(ROOT, "docker", "webapp", "examples")
PAPER_BASENAME = ("Nagaki et al. 2007 - Integrated micro flow synthesis based on sequential Br-Li exchange "
                  "reactions of p-, m-, and o-dibromobenzenes")
PAPER_TITLE = ("Integrated Micro Flow Synthesis Based on Sequential Br–Li Exchange Reactions of p-, m-, "
               "and o-Dibromobenzenes")
PAPER_CITATION = ("A. Nagaki, Y. Tomida, H. Usutani, H. Kim, N. Takabayashi, T. Nokami, H. Okamoto, J. Yoshida · "
                  "Chem. Asian J. 2007")   # as printed on the paper's first page


def _fresh(kind):
    d = os.path.join(OUT, kind)
    if os.path.isdir(d):
        shutil.move(d, d + ".previous")        # kept aside, not deleted
    os.makedirs(d)
    return d


def _save(kind, result, out_dir):
    json.dump(jobs.clean(result), open(os.path.join(out_dir, "result.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{kind}: saved {out_dir}/result.json")


def main(which):
    keys = jobs.server_keys()
    if ("figure" in which or "table" in which) and not keys:
        sys.exit("no working GEMINI_API_KEY in the environment")
    examples = os.path.join(ROOT, "src", "services", "examples")
    if "figure" in which:
        d = _fresh("figure")
        crop = os.path.join(ROOT, "data", "intermediate", PAPER_BASENAME, "figures", "page_3_figure_0.png")
        result = jobs.extract_figure(crop, keys, d)
        for p in result.get("panels", []):
            p["label"] = "Figure 2 · Nagaki et al. 2007"
            p["axes"] = {"x": "residence time (s)", "y": "temperature (°C)", "value": "yield (%)"}
        _save("figure", result, d)
    if "figure2" in which:
        # a scatter / line plot with two y axes (the Cloud Run demo's example figure)
        d = _fresh("figure2")
        result = jobs.extract_figure(os.path.join(examples, "example_figure.png"), keys, d)
        for p in result.get("panels", []):
            p["label"] = "Scatter / line plot"
            p["axes"] = {"x": "liquid flow rate (mL/min)", "y": "selectivity (%)"}
            # each point is read against both y axes; the legend puts the main
            # product on the left axis and the by-products on the right one
            p["series_axis"] = {s["name"]: ("left" if s["name"].startswith("3,4-dichloroaniline") else "right")
                                for s in p["series"]}
        _save("figure2", result, d)
    if "table" in which:
        d = _fresh("table")
        _save("table", jobs.extract_table(os.path.join(examples, "example_table.png"), keys, d), d)
    if "paper" in which:
        d = _fresh("paper")
        records = json.load(open(os.path.join(ROOT, "data", "final_output", PAPER_BASENAME + "_normalized.json")))
        result = jobs.paper_result(records, os.path.join(ROOT, "data", "intermediate", PAPER_BASENAME), d,
                                   title=PAPER_TITLE)
        result["citation"] = PAPER_CITATION
        _save("paper", result, d)
    for p in glob.glob(os.path.join(OUT, "*.previous")):
        print("previous version kept at", p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["figure", "figure2", "table", "paper"])
