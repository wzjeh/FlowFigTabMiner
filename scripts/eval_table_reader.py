"""Evaluate the VLM table transcriber (table stage rerun + reassembly + metrics).

Re-runs ONLY the table stage on existing TF-ID crops (``data/intermediate/{paper}/tables/*.png``)
with the reader set by ``--reader gemini|claude`` (via FFTM_TABLE_READER), then
``scripts/reassemble.py --rebuild-vars`` (table local_vars depend on the CSV), then
collects per-table transcription facts and table-record coverage.

Usage:
  python scripts/eval_table_reader.py --reader gemini --papers-from eval/paper_sets/eval30.txt
  python scripts/eval_table_reader.py --reader gemini --papers "data/input/Clean organolithium/Nagaki*.pdf"
  python scripts/eval_table_reader.py --reader gemini --papers-from eval/paper_sets/eval30.txt --report-only
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PRICE_PER_M = {"gemini-2.5-flash": (0.30, 2.50), "claude-sonnet-5": (2.00, 10.00)}
INPUT_ROOTS = ["data/input"]


def _resolve_pdf(name: str) -> str | None:
    if os.path.exists(name):
        return name
    base = os.path.basename(name)
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    # Intermediate dir names are shortened for long titles; layout.json keeps the real PDF path.
    layout = os.path.join("data/intermediate", base, "layout.json")
    if os.path.exists(layout):
        try:
            pdf = json.load(open(layout)).get("pdf")
            if pdf and os.path.exists(pdf):
                return pdf
        except Exception:
            pass
    for root in INPUT_ROOTS:
        hits = glob.glob(os.path.join(glob.escape(root), "**", glob.escape(base) + ".pdf"), recursive=True)
        if hits:
            return hits[0]
    return None


def _papers_from(path: str) -> list[str]:
    if path.endswith((".xlsx", ".csv")):
        import pandas as pd
        df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
        names = df["pdf"].dropna().astype(str)
    else:
        names = [l.strip() for l in open(path) if l.strip()]
    out = []
    for n in names:
        p = _resolve_pdf(n)
        if p:
            out.append(p)
        else:
            print(f"[eval] paper not found: {n}")
    return out


def run_stage(paths: list[str], reader: str) -> dict:
    os.environ["FFTM_TABLE_READER"] = reader
    import src.pipeline._threadcaps  # noqa: F401
    from src.extraction.common.content_recognizer import ContentRecognizer
    from src.extraction.table.pipeline import TablePipeline
    from src.llm.providers.gemini import GeminiProvider
    from src.pipeline.main import _build_table_transcriber
    from scripts.reassemble import main as reassemble_main
    transcriber, cfg = _build_table_transcriber(GeminiProvider())
    pipe = TablePipeline(transcriber=transcriber, content_recognizer=ContentRecognizer(),
                         min_text_agreement=cfg.min_text_agreement)
    timing = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        idir = os.path.join("data/intermediate", base)
        tdir = os.path.join(idir, "tables")
        imgs = [f for f in sorted(glob.glob(os.path.join(tdir, "*.png"))) if "_body" not in f and "_crop" not in f]
        print(f"[eval] === {base} ({len(imgs)} tables, reader={reader})", flush=True)
        t0 = time.time()
        for img in imgs:
            try:
                pipe.process_table(img, output_dir=tdir)
            except Exception as exc:
                print(f"[eval] table stage error {img}: {exc}", flush=True)
        timing[base] = round(time.time() - t0, 1)
    reassemble_main(paths, rebuild_vars=True)
    return timing


def _rdkit_valid(s: str) -> bool:
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return bool(s) and len(s) > 2 and Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


def collect(paths: list[str]) -> tuple[dict, list[dict]]:
    m = collections.Counter(); tables = []
    tok_in = tok_out = calls = 0; models = collections.Counter(); agreements = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        idir = os.path.join("data/intermediate", base)
        for evp in sorted(glob.glob(os.path.join(idir, "tables", "*", "*_evidence.json"))):
            ev = json.load(open(evp)); sid = os.path.basename(os.path.dirname(evp))
            m["tables"] += 1
            ps = str(ev.get("parse_status", "ok"))
            m[f"status_{ps.split(':')[0]}"] += 1
            al = ev.get("structure_alignment") or {}
            if al:
                m[f"align_{al.get('status')}"] += 1
                m["structure_tokens"] += int(al.get("n_tokens") or 0); m["structure_assigned"] += int(al.get("assigned") or 0)
            ag = ev.get("grid_text_agreement")
            if ag is not None:
                agreements.append(ag)
            else:
                m["no_text_layer_check"] += 1
            tr = ev.get("transcriber") or {}
            if tr.get("model") and not tr.get("cache_hit"):
                calls += 1; models[tr["model"]] += 1
                tok_in += tr.get("tokens_in") or 0; tok_out += tr.get("tokens_out") or 0
            # SMILES cells that did not come from the aligner (hallucination / OCR-fallback detector)
            valid_cells = 0; csv_preview = ""
            csvp = ev.get("csv_path")
            if csvp and os.path.exists(csvp):
                lines = open(csvp).read().splitlines()
                csv_preview = "\n".join(lines[:6])
                import csv as _csv
                for row in _csv.reader(lines):
                    valid_cells += sum(1 for c in row if _rdkit_valid(c.strip()))
            extra = max(0, valid_cells - int(al.get("assigned") or 0)) if al else 0
            m["smiles_cells_not_from_aligner"] += extra
            tables.append({"paper": base, "source_id": sid, "parse_status": ps, "n_rows": ev.get("n_rows"), "n_cols": ev.get("n_cols"),
                           "header_row_count": ev.get("header_row_count"), "grid_text_agreement": ag,
                           "structure_alignment": al.get("status"), "structures": f"{al.get('assigned')}/{al.get('n_tokens')}" if al else "",
                           "n_molecules": ev.get("n_molecules"), "smiles_cells_not_from_aligner": extra,
                           "tokens_out": tr.get("tokens_out"), "vlm_caption": ev.get("caption_text", "")[:200],
                           "png": os.path.join(idir, "tables", f"{sid}.png"), "csv_preview": csv_preview})
        fin = f"data/final_output/{base}_normalized.json"
        if os.path.exists(fin):
            for r in json.load(open(fin)):
                if r.get("__synthesized"):
                    continue
                c = r.get("conditions") or {}
                m["records"] += 1
                m["rec_T"] += c.get("temperature_C") is not None; m["rec_tR"] += c.get("residence_time_s") is not None
                m["rec_yield"] += r.get("yield_pct") is not None; m["rec_solvent"] += bool(c.get("solvent"))
                m["rec_product_name"] += bool(r.get("product_name")); m["rec_product_smiles"] += bool(r.get("product_smiles"))
                m["rec_reactant_smiles"] += bool(r.get("reactant1_smiles")); m["rec_has_outcome"] += bool(r.get("has_outcome"))
    n = m["records"] or 1
    cov = {k.replace("rec_", "") + "_pct": round(100 * m[k] / n, 1) for k in list(m) if k.startswith("rec_")}
    model = models.most_common(1)[0][0] if models else None
    pin, pout = PRICE_PER_M.get(model or "", (0, 0))
    metrics = {**m, **cov,
               "text_agreement_mean": round(sum(agreements) / len(agreements), 3) if agreements else None,
               "vlm_calls": calls, "vlm_model": model, "tokens_in": tok_in, "tokens_out": tok_out,
               "est_cost_usd": round(tok_in / 1e6 * pin + tok_out / 1e6 * pout, 4)}
    return metrics, tables


def write_spotcheck(tables: list[dict], n: int, out_xlsx: str, seed: int = 0) -> None:
    import pandas as pd
    pool = [t for t in tables if not t["parse_status"].startswith("failed")]
    random.Random(seed).shuffle(pool)
    rows = []
    for t in pool[:n]:
        ctx_p = os.path.join("data/intermediate", t["paper"], "context", f"{t['source_id']}_context.json")
        ctx = json.load(open(ctx_p)) if os.path.exists(ctx_p) else {}
        rows.append({"paper": t["paper"], "source_id": t["source_id"], "table_png": t["png"],
                     "context_caption": (ctx.get("caption") or "")[:200], "vlm_caption": t["vlm_caption"],
                     "header_row_count": t["header_row_count"], "n_rows": t["n_rows"], "n_cols": t["n_cols"],
                     "structure_alignment": t["structure_alignment"], "structures": t["structures"],
                     "grid_text_agreement": t["grid_text_agreement"], "parse_status": t["parse_status"],
                     "csv_preview": t["csv_preview"], "correct? (Y/N)": "", "note": ""})
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    print("-> spot-check sheet:", out_xlsx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--papers", nargs="*", default=[])
    ap.add_argument("--papers-from", help="txt of paper basenames/paths, or audit xlsx/csv with a 'pdf' column")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--spot-check", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paths = [p for pat in a.papers for p in (glob.glob(pat) or [pat])]
    if a.papers_from:
        paths += _papers_from(a.papers_from)
    paths = sorted(set(paths))
    if not paths:
        ap.error("no papers")
    t0 = time.time(); timing = {}
    if not a.report_only:
        timing = run_stage(paths, a.reader)
    metrics, tables = collect(paths)
    metrics.update({"papers": len(paths), "wall_s": round(time.time() - t0, 1), "reader": a.reader,
                    "table_stage_s": round(sum(timing.values()), 1) if timing else None})
    tag = a.out or f"eval/table_reader/{a.reader}_{time.strftime('%Y-%m-%d')}.json"
    os.makedirs(os.path.dirname(tag), exist_ok=True)
    json.dump({"metrics": metrics, "tables": tables, "timing": timing}, open(tag, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2)); print("->", tag)
    if a.spot_check:
        write_spotcheck(tables, a.spot_check, tag.replace(".json", "_spotcheck.xlsx"))


if __name__ == "__main__":
    main()
