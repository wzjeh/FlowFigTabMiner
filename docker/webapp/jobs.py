"""Extraction jobs for the demo page: one figure image, one table image, or a PDF.

Every job runs on the server's Gemini key (GEMINI_API_KEY, then
GEMINI_API_KEY_n / GOOGLE_API_KEY_n when a key's quota is exhausted).  Visitors
get one free try per kind (by IP) and the month has a total cap per kind
(FFTM_TRIAL_MONTHLY, default "figure=1000,table=1000,pdf=100"), counted in
data/web_usage.json.  A job that fails on our side is refunded.

Each job returns a display dict (no files to download): the same shape the
precomputed examples use, see ``figure_result`` / ``table_result`` /
``paper_result``.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import pandas as pd

APP_DIR = "/app" if os.path.isdir("/app/src") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)            # the pipeline reads config.yaml and models/ by relative path

INTER_DIR = os.path.join(APP_DIR, "data", "intermediate")
FINAL_DIR = os.path.join(APP_DIR, "data", "final_output")
JOBS_DIR = os.path.join(APP_DIR, "data", "web_jobs")          # per-job display images, served under /files/
USAGE_PATH = os.path.join(APP_DIR, "data", "web_usage.json")
EXIT_SKIPPED = 3
KINDS = ("figure", "table", "pdf")


class TrialRefused(Exception):
    """The visitor's free try is used up, or the month's cap is reached."""


# ── keys ──────────────────────────────────────────────────────────────────

_KEY_OK = {}


def _quota_exhausted(text) -> bool:
    t = str(text)
    return "429" in t or "RESOURCE_EXHAUSTED" in t or "quota" in t.lower()


def _key_works(key: str) -> bool:
    """One cheap call per key per process: an invalid key does not raise inside
    the pipeline (each VLM step logs and carries on), it only empties the result."""
    if key not in _KEY_OK:
        try:
            from google import genai
            client = genai.Client(api_key=key)          # keep a reference: a collected client closes itself
            next(iter(client.models.list(config={"page_size": 1})))
            _KEY_OK[key] = True
        except Exception as exc:  # noqa: BLE001
            _KEY_OK[key] = _quota_exhausted(exc)     # over quota is still a valid key
    return _KEY_OK[key]


def server_keys():
    """Working server keys, primary first."""
    keys = [os.environ.get("GEMINI_API_KEY", "")]
    for i in range(1, 5):
        keys += [os.environ.get(f"GEMINI_API_KEY_{i}", ""), os.environ.get(f"GOOGLE_API_KEY_{i}", "")]
    out = []
    for k in keys:
        if k and k not in out and _key_works(k):
            out.append(k)
    return out


def _with_fallback(keys, job_fn):
    last = None
    for i, key in enumerate(keys):
        try:
            return job_fn(key)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i + 1 < len(keys) and _quota_exhausted(exc):
                continue
            raise
    raise last


class _WithKey:
    """GEMINI_API_KEY in the process environment for one job (jobs run one at a time)."""

    def __init__(self, key):
        self.key, self.old = key, None

    def __enter__(self):
        self.old = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = self.key

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.old


# ── free-trial accounting ─────────────────────────────────────────────────

_USAGE_LOCK = threading.Lock()


def _monthly_caps():
    raw = os.environ.get("FFTM_TRIAL_MONTHLY", "figure=1000,table=1000,pdf=100")
    return {k.strip(): int(v) for k, v in (kv.split("=") for kv in raw.split(",") if "=" in kv)}


def _load_usage():
    month = time.strftime("%Y-%m")
    try:
        u = json.load(open(USAGE_PATH))
    except Exception:  # noqa: BLE001 - first run or unreadable: start the month fresh
        u = {}
    if u.get("month") != month:
        u = {"month": month, "counts": {}, "visitors": {}}
    return u


def take_trial(kind: str, visitor: str):
    with _USAGE_LOCK:
        u = _load_usage()
        used = u["visitors"].setdefault(kind, [])
        if visitor in used:
            raise TrialRefused(f"The free {kind} try has already been used from this address.")
        cap = _monthly_caps().get(kind, 0)
        if u["counts"].get(kind, 0) >= cap:
            raise TrialRefused(f"This month's free {kind} tries are used up. Please come back next month.")
        used.append(visitor)
        u["counts"][kind] = u["counts"].get(kind, 0) + 1
        os.makedirs(os.path.dirname(USAGE_PATH), exist_ok=True)
        json.dump(u, open(USAGE_PATH, "w"), indent=1)


def refund_trial(kind: str, visitor: str):
    with _USAGE_LOCK:
        u = _load_usage()
        if visitor in u["visitors"].get(kind, []):
            u["visitors"][kind].remove(visitor)
            u["counts"][kind] = max(0, u["counts"].get(kind, 0) - 1)
            json.dump(u, open(USAGE_PATH, "w"), indent=1)


# ── display helpers ───────────────────────────────────────────────────────


def clean(o):
    """JSON-safe: NaN / inf become None (browsers reject NaN in JSON)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def looks_like_smiles(text) -> bool:
    """A table cell MolNexTR filled with a structure (not '95', 'THF' or 'Me3SiCl')."""
    s = str(text or "").strip()
    if len(s) < 3 or " " in s or re.fullmatch(r"[\d.,%\-–()]+", s):
        return False
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(s)
    except Exception:  # noqa: BLE001
        return False
    return mol is not None and mol.GetNumHeavyAtoms() >= 3


def _publish(src, out_dir, name, max_side=1400):
    """Copy an image into the job's display dir (downscaled); return its file name."""
    import cv2
    img = cv2.imread(src)
    if img is None:
        return None
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name), img)
    return name


def figure_result(evidence_paths, input_png, out_dir, seconds=None):
    panels = []
    for i, ev_path in enumerate(evidence_paths):
        ev = json.load(open(ev_path))
        meta = ev.get("meta") or {}
        facts = meta.get("facts") or {}
        series = {}
        for row in ev.get("raw_data") or []:
            x, y = row.get("X"), row.get("Y_Left")
            if x is None or y is None:
                continue
            series.setdefault(str(row.get("Series") or "data"), []).append(
                [x, y, row.get("Y_Right/Data_Value")])
        cleaned = ev_path.replace("_evidence.json", "_cleaned.png")
        panels.append({
            "label": meta.get("label") or f"Panel {i + 1}",
            "caption": meta.get("caption_pdf") or meta.get("caption") or "",
            "chart_type": meta.get("figure_type") or facts.get("chart_type"),
            "x_log": str(facts.get("x_scale") or "").startswith("log"),
            "cleaned": _publish(cleaned, out_dir, f"cleaned_{i}.png") if os.path.exists(cleaned) else None,
            "series": [{"name": k, "points": v} for k, v in series.items()],
        })
    return {"kind": "figure", "input": _publish(input_png, out_dir, "input.png"), "panels": panels,
            "n_points": sum(len(s["points"]) for p in panels for s in p["series"]), "seconds": seconds}


def _table_cell(v):
    """A structure cell may carry its compound label and yield after the SMILES ('<smiles> 3a 24')."""
    head, _, rest = str(v).strip().partition(" ")
    if looks_like_smiles(head):
        return {"smiles": head, "text": rest} if rest else {"smiles": head}
    return {"text": str(v)}


def table_result(result, input_png, out_dir, seconds=None):
    if result.get("csv_path") and os.path.exists(result["csv_path"]):
        df = pd.read_csv(result["csv_path"], dtype=str, keep_default_na=False)
    else:
        df = pd.DataFrame(result.get("dataframe") if result.get("dataframe") is not None else [])
    columns = [("" if str(c).startswith("Unnamed") else str(c)) for c in df.columns]
    rows = [[_table_cell(v) for v in r] for r in df.astype(str).values.tolist()]
    return {"kind": "table", "input": _publish(input_png, out_dir, "input.png"), "caption": result.get("caption_text") or "",
            "notes": result.get("table_note_text") or "", "columns": columns, "rows": rows, "seconds": seconds}


def _record_row(r):
    c = r.get("conditions") or {}
    return {"r1": r.get("reactant1_smiles"), "r1_name": r.get("reactant1_name"),
            "r2": r.get("reactant2_smiles"), "r2_name": r.get("reactant2_name"),
            "p": r.get("product_smiles"), "p_name": r.get("product_name"), "p_label": r.get("product_label"),
            "T": c.get("temperature_C"), "tR": c.get("residence_time_s"), "tR2": c.get("residence_time_2_s"),
            "yield": r.get("yield_pct"), "source": (r.get("source_table_or_figure") or "").split(" — ")[0]}


def paper_result(records, intermediate_dir, out_dir, title, doi=None, seconds=None, max_rows=None):
    """Records with an outcome, plus one thumbnail per source figure/table."""
    shown = [r for r in records if r.get("has_outcome")] or list(records)
    sources, seen = [], set()
    for r in shown:
        sid = r.get("__source_id") or ""
        if not sid or sid in seen:
            continue
        seen.add(sid)
        base = re.sub(r"_t\d+$", "", sid)
        crop = next((p for p in (os.path.join(intermediate_dir, "figures", base + ".png"),
                                 os.path.join(intermediate_dir, "tables", sid + ".png")) if os.path.exists(p)), None)
        label = (r.get("source_table_or_figure") or sid).split(" — ")[0]
        sources.append({"label": label, "thumb": _publish(crop, out_dir, f"src_{len(sources)}.png", 900) if crop else None,
                        "n": sum(1 for x in shown if x.get("__source_id") == sid)})
    def _order(s):
        m = re.match(r"(\D*?)\s*(\d+)", s["label"])
        return (not s["label"].lower().startswith("fig"), int(m.group(2)) if m else 999, s["label"])
    sources.sort(key=_order)
    rows = [_record_row(r) for r in shown]
    return {"kind": "paper", "title": title, "doi": doi or next((r.get("paper_doi") for r in records if r.get("paper_doi")), None),
            "n_records": len(shown), "n_with_structure": sum(1 for r in shown if r.get("product_smiles")),
            "n_figures": sum(1 for s in sources if s["label"].lower().startswith("fig")),
            "n_tables": sum(1 for s in sources if s["label"].lower().startswith("table")),
            "sources": sources, "records": rows[:max_rows] if max_rows else rows, "seconds": seconds}


# ── the three extractions ─────────────────────────────────────────────────


def _build_figure_pipeline(provider):
    from src.extraction.figure.metadata_vlm import FigureMetadataExtractor
    from src.llm.config import load_vlm_config
    from src.pipeline.figure_pipeline import FigurePipeline
    from src.pipeline.main import _build_label_reader
    vlm_cfg = load_vlm_config("config.yaml")
    label_reader = _build_label_reader(provider)
    return FigurePipeline(metadata_extractor=FigureMetadataExtractor(vlm=provider, cfg=vlm_cfg),
                          label_reader=label_reader,
                          value_conflict_policy=getattr(label_reader, "value_conflict_policy", "vlm"))


def extract_figure(image_path, keys, out_dir, status=lambda s: None):
    work = os.path.join(INTER_DIR, "web_figure_" + uuid.uuid4().hex[:8])
    os.makedirs(os.path.join(work, "figures"))
    src = os.path.join(work, "figures", "figure.png")
    shutil.copy(image_path, src)
    t0 = time.time()
    status("Reading the plot")

    def _job(key):
        with _WithKey(key):
            from src.llm.providers.gemini import GeminiProvider
            return _build_figure_pipeline(GeminiProvider()).process_images([src], work)

    try:
        evidence = _with_fallback(keys, _job)
        if not evidence:
            return {"kind": "figure", "input": _publish(src, out_dir, "input.png"), "panels": [], "n_points": 0,
                    "message": "No scatter plot or heatmap was found in this image."}
        return figure_result(evidence, src, out_dir, round(time.time() - t0))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def extract_table(image_path, keys, out_dir, status=lambda s: None):
    work = os.path.join(INTER_DIR, "web_table_" + uuid.uuid4().hex[:8])
    tables_dir = os.path.join(work, "tables")
    os.makedirs(tables_dir)
    src = os.path.join(tables_dir, "table.png")
    shutil.copy(image_path, src)
    t0 = time.time()
    status("Reading the table and its structures")

    def _job(key):
        with _WithKey(key):
            from src.extraction.common.content_recognizer import ContentRecognizer
            from src.extraction.table.pipeline import TablePipeline
            from src.llm.providers.gemini import GeminiProvider
            from src.pipeline.main import _build_table_transcriber
            transcriber, table_cfg = _build_table_transcriber(GeminiProvider())
            pipeline = TablePipeline(transcriber=transcriber, content_recognizer=ContentRecognizer(),
                                     min_text_agreement=table_cfg.min_text_agreement)
            return pipeline.process_table(src, output_dir=tables_dir)

    try:
        result = _with_fallback(keys, _job)
        if not result.get("is_valid"):
            return {"kind": "table", "input": _publish(src, out_dir, "input.png"), "columns": [], "rows": [],
                    "message": "No reaction table was recognised in this image."}
        return table_result(result, src, out_dir, round(time.time() - t0))
    finally:
        shutil.rmtree(work, ignore_errors=True)


_STEP_NAMES = {"0": "Checking the paper", "1": "Finding figures and tables", "2-4": "Reading the figures",
               "Table": "Reading the tables", "3.5": "Reading reaction schemes", "4.4": "Reading the text",
               "4.5": "Reading the text", "5": "Assembling reaction records", "6": "Normalising structures"}


def extract_pdf(pdf_path, keys, out_dir, status=lambda s: None):
    job = uuid.uuid4().hex[:8]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(pdf_path))[0])[:60]
    basename = f"web_pdf_{job}_{stem}"
    input_dir = os.path.join(APP_DIR, "data", "input", "web")
    os.makedirs(input_dir, exist_ok=True)
    local_pdf = os.path.join(input_dir, basename + ".pdf")
    shutil.copy(pdf_path, local_pdf)
    t0 = time.time()
    tail = []
    try:
        proc = None
        for attempt, key in enumerate(keys):
            env = dict(os.environ, GEMINI_API_KEY=key, PYTHONUNBUFFERED="1")
            args = [sys.executable, "-m", "src.pipeline.main", local_pdf] + (["--skip-tfid", "--force-assembly"] if attempt else [])
            proc = subprocess.Popen(args, cwd=APP_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            quota_hit = False
            for line in proc.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-30:]
                quota_hit = quota_hit or _quota_exhausted(line)
                m = re.search(r"=== Step ([\w.\-]+)", line)
                if m:
                    status(_STEP_NAMES.get(m.group(1), f"Step {m.group(1)}"))
            proc.wait()
            if proc.returncode == 0 or not quota_hit or attempt + 1 >= len(keys):
                break
        js = os.path.join(FINAL_DIR, basename + "_normalized.json")
        if proc.returncode == EXIT_SKIPPED:
            return {"kind": "paper", "records": [], "sources": [],
                    "message": "This does not look like a flow-chemistry research paper, so it was skipped."}
        if proc.returncode != 0 or not os.path.exists(js):
            raise RuntimeError("the pipeline did not finish: " + " | ".join(tail[-5:]))
        records = json.load(open(js))
        if isinstance(records, dict):
            records = records.get("records", [])
        return paper_result(records, os.path.join(INTER_DIR, basename), out_dir,
                            title=os.path.splitext(os.path.basename(pdf_path))[0], seconds=round(time.time() - t0))
    finally:
        # the uploaded paper and its intermediates are not kept on the server
        shutil.rmtree(os.path.join(INTER_DIR, basename), ignore_errors=True)
        if os.path.exists(local_pdf):
            os.remove(local_pdf)
        for p in glob.glob(os.path.join(glob.escape(FINAL_DIR), glob.escape(basename) + "*")):
            os.remove(p)


EXTRACTORS = {"figure": extract_figure, "table": extract_table, "pdf": extract_pdf}
