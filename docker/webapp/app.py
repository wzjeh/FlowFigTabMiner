"""Gradio demo for the FlowFigTabMiner image: figure, table, or whole PDF.

Started by the image entrypoint as ``web`` (port 7860).  Two ways to pay for
the Gemini calls:
  * bring your own key (used for this job only, never written anywhere);
  * demo mode: the demo password unlocks the key in the server environment.

Server environment (never in this file): GEMINI_API_KEY, optional GEMINI_API_KEY_1../GOOGLE_API_KEY_1..,
DEMO_PASSWORD.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import gradio as gr
import pandas as pd

APP_DIR = "/app" if os.path.isdir("/app/src") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

WEB_ROOT = os.path.join(APP_DIR, "data", "intermediate")
FINAL_DIR = os.path.join(APP_DIR, "data", "final_output")
EXAMPLES = os.path.join(APP_DIR, "src", "services", "examples")
EXIT_SKIPPED = 3


# ── keys ──────────────────────────────────────────────────────────────────


def _demo_keys():
    """GEMINI_API_KEY first, then GEMINI_API_KEY_1.. / GOOGLE_API_KEY_1.. as quota fallbacks."""
    keys = [os.environ.get("GEMINI_API_KEY", "")]
    for i in range(1, 5):
        keys += [os.environ.get(f"GEMINI_API_KEY_{i}", ""), os.environ.get(f"GOOGLE_API_KEY_{i}", "")]
    out = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


_KEY_OK = {}


def _key_works(key: str) -> bool:
    """One cheap call per key per process: an invalid key would not raise in the
    pipeline (each VLM step logs and emits MISSING), it would just empty the result."""
    if key not in _KEY_OK:
        try:
            from google import genai
            client = genai.Client(api_key=key)          # keep a reference: a collected client closes itself
            next(iter(client.models.list(config={"page_size": 1})))
            _KEY_OK[key] = True
        except Exception as exc:  # noqa: BLE001
            _KEY_OK[key] = _quota_exhausted(exc)     # over quota is still a valid key
    return _KEY_OK[key]


def _resolve_key(user_key: str, demo_password: str):
    """-> (keys to try in order, mode label)."""
    expected = os.environ.get("DEMO_PASSWORD", "")
    if demo_password.strip():
        if expected and demo_password.strip() == expected:
            keys = [k for k in _demo_keys() if _key_works(k)]
            if not keys:
                raise gr.Error("Demo mode is not configured on this server (no working Gemini key).")
            return keys, "demo"
        raise gr.Error("Wrong demo password.")
    if user_key.strip():
        if not _key_works(user_key.strip()):
            raise gr.Error("This Gemini API key is not valid.")
        return [user_key.strip()], "own key"
    raise gr.Error("Enter your Gemini API key, or the demo password.")


def _quota_exhausted(text: str) -> bool:
    t = str(text)
    return "429" in t or "RESOURCE_EXHAUSTED" in t or "quota" in t.lower()


def _with_fallback(keys, job_fn):
    """Run ``job_fn(key)``; on a quota error move to the next key."""
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
    """GEMINI_API_KEY in the process environment for one job only."""

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


def _job_dir(kind):
    d = os.path.join(WEB_ROOT, f"web_{kind}_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


# ── figure ────────────────────────────────────────────────────────────────


def _build_figure_pipeline(provider):
    from src.extraction.figure.metadata_vlm import FigureMetadataExtractor
    from src.pipeline.figure_pipeline import FigurePipeline
    from src.pipeline.main import _build_label_reader
    from src.llm.config import load_vlm_config
    vlm_cfg = load_vlm_config("config.yaml")
    label_reader = _build_label_reader(provider)
    return FigurePipeline(metadata_extractor=FigureMetadataExtractor(vlm=provider, cfg=vlm_cfg),
                          label_reader=label_reader,
                          value_conflict_policy=getattr(label_reader, "value_conflict_policy", "vlm"))


def run_figure(image_path, user_key, demo_password):
    if not image_path:
        raise gr.Error("Upload a figure image first.")
    keys, mode = _resolve_key(user_key or "", demo_password or "")
    job = _job_dir("figure")
    fig_dir = os.path.join(job, "figures")
    os.makedirs(fig_dir)
    src = os.path.join(fig_dir, "figure.png")
    shutil.copy(image_path, src)
    t0 = time.time()

    def _job(key):
        with _WithKey(key):
            from src.llm.providers.gemini import GeminiProvider
            pipeline = _build_figure_pipeline(GeminiProvider())
            return pipeline.process_images([src], job)

    try:
        evidence_paths = _with_fallback(keys, _job)
    except Exception as exc:
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error(f"Figure extraction failed: {type(exc).__name__}: {exc}")
    if not evidence_paths:
        shutil.rmtree(job, ignore_errors=True)
        return None, "Nothing extracted: macro YOLO did not find a scatter plot or heatmap in this image.", None, None
    frames, cleaned, meta_lines = [], None, []
    for ev_path in evidence_paths:
        ev = json.load(open(ev_path))
        fid = ev.get("meta", {}).get("figure_id", os.path.basename(ev_path))
        for row in ev.get("raw_data") or []:
            frames.append({"panel": fid, **row})
        te = ev.get("text_evidence") or {}
        meta_lines.append(f"{fid}: type={ev.get('meta', {}).get('figure_type')} relevant={ev.get('is_relevant')} "
                          f"x={te.get('x_axis_title')} y={te.get('y_axis_title')} legend={te.get('legend_text')}")
        c = ev_path.replace("_evidence.json", "_cleaned.png")
        if cleaned is None and os.path.exists(c):
            cleaned = c
    df = pd.DataFrame(frames)
    out_json = os.path.join(job, "figure_points.json")
    json.dump(frames, open(out_json, "w"), indent=1)
    summary = f"[{mode}] {len(evidence_paths)} panel(s), {len(frames)} data points in {time.time() - t0:.0f}s\n" + "\n".join(meta_lines)
    return cleaned, summary, df, out_json


# ── table ─────────────────────────────────────────────────────────────────


def run_table(image_path, user_key, demo_password):
    if not image_path:
        raise gr.Error("Upload a table image first.")
    keys, mode = _resolve_key(user_key or "", demo_password or "")
    job = _job_dir("table")
    tables_dir = os.path.join(job, "tables")
    os.makedirs(tables_dir)
    src = os.path.join(tables_dir, "table.png")
    shutil.copy(image_path, src)
    t0 = time.time()

    def _job(key):
        with _WithKey(key):
            from src.llm.providers.gemini import GeminiProvider
            from src.pipeline.main import _build_table_transcriber
            from src.extraction.common.content_recognizer import ContentRecognizer
            from src.extraction.table.pipeline import TablePipeline
            transcriber, table_cfg = _build_table_transcriber(GeminiProvider())
            pipeline = TablePipeline(transcriber=transcriber, content_recognizer=ContentRecognizer(),
                                     min_text_agreement=table_cfg.min_text_agreement)
            return pipeline.process_table(src, output_dir=tables_dir)

    try:
        result = _with_fallback(keys, _job)
    except Exception as exc:
        shutil.rmtree(job, ignore_errors=True)
        raise gr.Error(f"Table extraction failed: {type(exc).__name__}: {exc}")
    if not result.get("is_valid"):
        return None, f"Not extracted: {result.get('reason')}", None, None
    df = result.get("dataframe")
    if df is None and result.get("csv_path"):
        df = pd.read_csv(result["csv_path"])
    debug = glob.glob(os.path.join(tables_dir, "table", "*debug_yolo.png"))
    summary = (f"[{mode}] {0 if df is None else len(df)} rows, parse={result.get('parse_status')} "
               f"relevant={result.get('is_relevant')} in {time.time() - t0:.0f}s\n"
               f"Caption: {result.get('caption_text') or '(none)'}\nNotes: {result.get('table_note_text') or '(none)'}")
    return (debug[0] if debug else src), summary, df, result.get("csv_path")


# ── whole PDF ─────────────────────────────────────────────────────────────


def _preview(records):
    rows = []
    for r in records[:50]:
        c = r.get("conditions") or {}
        rows.append([r.get("reactant1_smiles"), r.get("reactant2_smiles"), r.get("product_smiles"), r.get("product_name"),
                     r.get("yield_pct"), c.get("temperature_C"), c.get("residence_time_s"), c.get("solvent"),
                     r.get("source_table_or_figure")])
    return rows


def run_pdf(pdf, user_key, demo_password, progress=gr.Progress()):
    if pdf is None:
        raise gr.Error("Upload a PDF first.")
    keys, mode = _resolve_key(user_key or "", demo_password or "")
    job = uuid.uuid4().hex[:8]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(pdf.name))[0])[:60]
    basename = f"web_pdf_{job}_{stem}"
    input_dir = os.path.join(APP_DIR, "data", "input", "web")
    os.makedirs(input_dir, exist_ok=True)
    pdf_path = os.path.join(input_dir, basename + ".pdf")
    shutil.copy(pdf.name, pdf_path)

    log_lines = [f"[{mode}] job {job}: {os.path.basename(pdf.name)}"]
    t0 = time.time()
    for attempt, key in enumerate(keys):
        env = dict(os.environ, GEMINI_API_KEY=key, PYTHONUNBUFFERED="1")
        args = [sys.executable, "-m", "src.pipeline.main", pdf_path] + (["--skip-tfid", "--force-assembly"] if attempt else [])
        proc = subprocess.Popen(args, cwd=APP_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        quota_hit = False
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                log_lines.append(line)
                quota_hit = quota_hit or _quota_exhausted(line)
                m = re.search(r"=== Step ([\w.\-]+)", line)
                if m:
                    progress(0.1, desc=f"Step {m.group(1)}  ({time.time() - t0:.0f}s)")
                yield "\n".join(log_lines[-40:]), None, None, None
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode == 0 or not quota_hit or attempt + 1 >= len(keys):
            break
        log_lines.append(f"Gemini quota exhausted on key {attempt + 1}; retrying with the next key.")
        yield "\n".join(log_lines[-40:]), None, None, None

    xlsx = os.path.join(FINAL_DIR, basename + "_normalized.xlsx")
    js = os.path.join(FINAL_DIR, basename + "_normalized.json")
    if proc.returncode == EXIT_SKIPPED:
        log_lines.append("Skipped: the pre-filter did not recognise this as a flow-chemistry research paper.")
        yield "\n".join(log_lines[-40:]), None, None, None
        return
    if proc.returncode != 0 or not os.path.exists(js):
        log_lines.append(f"Pipeline exited with code {proc.returncode}; see log above.")
        yield "\n".join(log_lines[-60:]), None, None, None
        return
    records = json.load(open(js))
    if isinstance(records, dict):
        records = records.get("records", [])
    n_out = sum(1 for r in records if r.get("has_outcome"))
    log_lines.append(f"Done in {time.time() - t0:.0f}s: {len(records)} records, {n_out} with a measured outcome.")
    yield "\n".join(log_lines[-40:]), _preview(records), xlsx if os.path.exists(xlsx) else None, js
    shutil.rmtree(os.path.join(WEB_ROOT, basename), ignore_errors=True)   # the paper is not kept
    os.remove(pdf_path)


# ── UI ────────────────────────────────────────────────────────────────────

INTRO = ("# FlowFigTabMiner\n"
         "Reaction data from flow-chemistry papers: a **figure** (scatter plot / heatmap) gives its data points, "
         "a **table** gives its rows with structures read as SMILES, a **whole PDF** gives assembled reaction records "
         "(reactants, product, conditions, yield).\n\n"
         "Gemini does the reading of text and the assembly. Enter your own key "
         "([free key](https://aistudio.google.com/apikey); used for this job only) or the demo password.")

with gr.Blocks(title="FlowFigTabMiner") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        user_key = gr.Textbox(label="Your Gemini API key", type="password", scale=2)
        demo_password = gr.Textbox(label="or: demo password", type="password", scale=1)

    with gr.Tab("Figure"):
        with gr.Row():
            fig_in = gr.Image(label="Figure image (PNG/JPG)", type="filepath", height=320)
            with gr.Column():
                fig_summary = gr.Textbox(label="Summary", lines=6)
                fig_btn = gr.Button("Extract data points", variant="primary")
                if os.path.exists(os.path.join(EXAMPLES, "example_figure.png")):
                    gr.Examples([[os.path.join(EXAMPLES, "example_figure.png")]], inputs=[fig_in], label="Example")
        fig_cleaned = gr.Image(label="Cleaned plot area", height=320)
        fig_table = gr.Dataframe(label="Data points", wrap=True)
        fig_json = gr.File(label="JSON")
        fig_btn.click(run_figure, [fig_in, user_key, demo_password], [fig_cleaned, fig_summary, fig_table, fig_json],
                      concurrency_limit=1)

    with gr.Tab("Table"):
        with gr.Row():
            tab_in = gr.Image(label="Table image (PNG/JPG)", type="filepath", height=320)
            with gr.Column():
                tab_summary = gr.Textbox(label="Summary", lines=6)
                tab_btn = gr.Button("Extract table", variant="primary")
                if os.path.exists(os.path.join(EXAMPLES, "example_table.png")):
                    gr.Examples([[os.path.join(EXAMPLES, "example_table.png")]], inputs=[tab_in], label="Example")
        tab_debug = gr.Image(label="Detected structures", height=320)
        tab_table = gr.Dataframe(label="Rows (structures as SMILES)", wrap=True)
        tab_csv = gr.File(label="CSV")
        tab_btn.click(run_table, [tab_in, user_key, demo_password], [tab_debug, tab_summary, tab_table, tab_csv],
                      concurrency_limit=1)

    with gr.Tab("Whole PDF"):
        gr.Markdown("One paper takes 10–40 minutes on CPU. The PDF is deleted from the server when the job ends.")
        with gr.Row():
            pdf_in = gr.File(label="Paper (PDF)", file_types=[".pdf"])
            pdf_btn = gr.Button("Extract records", variant="primary")
        pdf_log = gr.Textbox(label="Log", lines=16, max_lines=16)
        pdf_table = gr.Dataframe(label="Records (first 50)",
                                 headers=["reactant1", "reactant2", "product", "product_name", "yield %", "T (°C)",
                                          "tR (s)", "solvent", "source"], wrap=True)
        with gr.Row():
            pdf_xlsx = gr.File(label="Excel")
            pdf_json = gr.File(label="JSON")
        pdf_btn.click(run_pdf, [pdf_in, user_key, demo_password], [pdf_log, pdf_table, pdf_xlsx, pdf_json],
                      concurrency_limit=1)

demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
