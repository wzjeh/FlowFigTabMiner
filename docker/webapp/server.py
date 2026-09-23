"""FlowFigTabMiner demo page (FastAPI).  Started by the image as ``web``:

    uvicorn server:app --app-dir docker/webapp --host 0.0.0.0 --port 7860

Examples are precomputed (docker/webapp/examples, built by
scripts/build_web_examples.py) and shown instantly.  A visitor's own figure,
table or paper runs live as a job: POST /api/jobs returns an id, the page polls
GET /api/jobs/{id}; every request is short, nothing depends on a long-lived
connection through a proxy.
"""
import functools
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import jobs  # noqa: E402  (uvicorn --app-dir docker/webapp)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "examples")
STATIC_DIR = os.path.join(HERE, "static")
MAX_UPLOAD = {"figure": 10, "table": 10, "pdf": 30}          # MB

app = FastAPI(title="FlowFigTabMiner demo", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")

_JOBS = {}                     # id -> {kind, state, step, submitted, started, finished, result, error}
_QUEUE = queue.Queue()
TYPICAL_SECONDS = {"figure": 90, "table": 75, "pdf": 360}      # until the server has timed a few of its own
KIND_NAME = {"figure": "a figure", "table": "a table", "pdf": "a paper"}


def _visitor(request: Request) -> str:
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _worker():
    """One job at a time: the models share one CPU box and one set of keys."""
    while True:
        job_id, kind, path, visitor = _QUEUE.get()
        job = _JOBS[job_id]
        job.update(state="running", started=time.time(), step="Starting")
        out_dir = os.path.join(jobs.JOBS_DIR, job_id)
        try:
            keys = jobs.server_keys()
            if not keys:
                raise RuntimeError("no working Gemini key on the server")
            job["result"] = jobs.clean(jobs.EXTRACTORS[kind](path, keys, out_dir, status=lambda s: job.update(step=s)))
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - the visitor gets a message, the log gets the trace
            traceback.print_exc()
            jobs.refund_trial(kind, visitor)
            job.update(state="failed", error="Extraction failed on our side; your free try was not used. "
                                             f"({type(exc).__name__})")
        finally:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            job["finished"] = time.time()


threading.Thread(target=_worker, daemon=True).start()


def _typical(kind: str) -> float:
    """Mean of the last five finished jobs of this kind on this server."""
    done = sorted((j for j in _JOBS.values() if j["kind"] == kind and j["state"] == "done" and "finished" in j),
                  key=lambda j: j["finished"])[-5:]
    return sum(j["finished"] - j["started"] for j in done) / len(done) if done else TYPICAL_SECONDS[kind]


def _queue_info(job):
    """Where a waiting job stands: jobs ahead, a rough wait, and what is running now."""
    now = time.time()
    ahead = [j for j in _JOBS.values() if j["state"] in ("queued", "running") and j["submitted"] < job["submitted"]]
    wait = 0.0
    for j in ahead:
        left = _typical(j["kind"])
        if j["state"] == "running":
            left = max(left - (now - j["started"]), 20)          # overran its typical time: a little longer
        wait += left
    running = next((j for j in ahead if j["state"] == "running"), None)
    return {"ahead": len(ahead), "wait": round(wait),
            "now": {"kind": KIND_NAME[running["kind"]], "step": running.get("step")} if running else None}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/limits")
def limits():
    return {"tries": jobs.tries_per_address()}


@app.get("/api/examples")
def examples():
    import json
    out = {}
    for kind in ("figure", "figure2", "table", "paper"):
        p = os.path.join(EXAMPLES_DIR, kind, "result.json")
        if os.path.exists(p):
            out[kind] = dict(json.load(open(p)), base=f"/examples/{kind}/")
    return out


@functools.lru_cache(maxsize=4096)
def _mol_svg(smiles: str, w: int, h: int) -> str:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.2
    opts.padding = 0.08
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


@app.get("/api/mol.svg")
def mol_svg(smiles: str, w: int = 180, h: int = 120):
    try:
        svg = _mol_svg(smiles, max(60, min(w, 600)), max(40, min(h, 400)))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "not a structure")
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/jobs")
async def submit(request: Request, kind: str = Form(...), file: UploadFile = File(...)):
    if kind not in jobs.KINDS:
        raise HTTPException(400, "unknown kind")
    data = await file.read()
    if len(data) > MAX_UPLOAD[kind] * 1024 * 1024:
        return JSONResponse({"error": f"File is larger than {MAX_UPLOAD[kind]} MB."}, status_code=400)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if (kind == "pdf") != (ext == ".pdf") or (kind != "pdf" and ext not in (".png", ".jpg", ".jpeg")):
        return JSONResponse({"error": "Please upload a PDF for a paper, or a PNG/JPG image for a figure or table."},
                            status_code=400)
    visitor = _visitor(request)
    try:
        jobs.take_trial(kind, visitor)
    except jobs.TrialRefused as exc:
        return JSONResponse({"error": str(exc)}, status_code=429)
    tmp = tempfile.mkdtemp(prefix="fftm_upload_")
    path = os.path.join(tmp, "upload" + (ext if ext != ".jpeg" else ".jpg"))
    with open(path, "wb") as f:
        f.write(data)
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"kind": kind, "state": "queued", "step": "Waiting in line", "submitted": time.time()}
    _QUEUE.put((job_id, kind, path, visitor))
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    t0 = job.get("started") or job["submitted"]
    out = {k: job.get(k) for k in ("kind", "state", "step", "result", "error")}
    out["elapsed"] = round((job.get("finished") or time.time()) - t0)
    if out["result"] is not None:
        out["result"] = dict(out["result"], base=f"/files/{job_id}/")
    if job["state"] == "queued":
        out.update(_queue_info(job))
    return out


@app.get("/files/{job_id}/{name}")
def job_file(job_id: str, name: str):
    p = os.path.join(jobs.JOBS_DIR, os.path.basename(job_id), os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)
