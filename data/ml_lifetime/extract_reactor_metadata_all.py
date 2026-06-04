"""
Extract flow-reactor engineering params (mixer ID, tube ID, flow rate) for ALL source papers
of the 45 global_arrhenius substrates, via Gemini + Google Search grounding. Used to compute a
per-paper τ_eff so the Damköhler regime map scatters each substrate by its REAL apparatus
(option 3). Reuses the Gemini call/retry/prompt from extract_reactor_metadata.py.

Run: flowfigtabminer/bin/python extract_reactor_metadata_all.py
"""
import json, time
from pathlib import Path
import pandas as pd
from google import genai
from google.genai import types
from extract_reactor_metadata import PROMPT, parse_json, call_with_retry, load_key

BASE = Path(__file__).parent

# ---- map 45 substrates → source paper (paper_id, doi) + substrate names ----
g = pd.read_csv(BASE / "global_arrhenius.csv")
raw = pd.read_csv(BASE / "clean_organolithium_unified.csv")
papers = {}   # pid -> dict(doi, names=set)
for _, r in g.iterrows():
    sub = raw[raw["intermediate_smiles_canonical"] == r["smi"]]
    for _, s in sub.dropna(subset=["paper_id"]).iterrows():
        pid = str(s["paper_id"]); doi = str(s.get("paper_doi", "") or "")
        papers.setdefault(pid, dict(doi=doi, names=set()))["names"].add(r["intermediate"])
PAPERS = [dict(pid=pid, doi=d["doi"],
               title=pid.replace("_", " "),
               substrates="; ".join(sorted(d["names"])[:6]) + " (aryl/alkyl-lithium by Br/Li or direct lithiation in a flow microreactor)")
          for pid, d in sorted(papers.items())]
print(f"{len(PAPERS)} papers to query\n")

def main():
    key = load_key()
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.1)
    rows = []
    for p in PAPERS:
        print(f"[Gemini] {p['pid']} ({p['doi']}) ...", flush=True)
        try:
            r = call_with_retry(client, "gemini-2.5-flash", PROMPT.format(**p), cfg)
            txt = r.text if getattr(r, "text", None) else (
                "".join(part.text for part in r.candidates[0].content.parts if getattr(part, "text", None))
                if getattr(r, "candidates", None) else "")
            d = parse_json(txt) if txt else {"_api_error": "empty response"}
        except Exception as e:
            d = {"_api_error": str(e)}
        d.update(paper_id=p["pid"], doi=p["doi"])
        rows.append(d)
        print("   ->", json.dumps({k: d.get(k) for k in
              ["mixer_inner_diameter_mm", "reactor_tube_inner_diameter_mm",
               "flow_rate_total_mL_min", "confidence"]}, ensure_ascii=False), flush=True)
        time.sleep(5)
    out = BASE / "reactor_metadata_all.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
