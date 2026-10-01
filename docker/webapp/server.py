"""FlowFigTabMiner demo page (FastAPI).  Started by the image as ``web``:

    uvicorn server:app --app-dir docker/webapp --host 0.0.0.0 --port 7860

Examples are precomputed (docker/webapp/examples, built by
scripts/build_web_examples.py) and shown instantly.  A visitor's own figure,
table or paper runs live as a job: POST /api/jobs returns an id, the page polls
GET /api/jobs/{id}; every request is short, nothing depends on a long-lived
connection through a proxy.

On Cloud Run this process is the worker behind ``front.py``, which serves the
page itself and forwards /api/jobs and /files here.
"""
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import jobs  # noqa: E402  (uvicorn --app-dir docker/webapp)
import pages

MAX_UPLOAD = {"figure": 10, "table": 10, "pdf": 30}          # MB
# docker/entrypoint-full.sh fetches the weights alongside the page and names the
# file it writes when done; a local run has its weights already.
MODELS_READY = os.environ.get("FFTM_MODELS_READY")

app = FastAPI(title="FlowFigTabMiner demo", docs_url=None, redoc_url=None)
app.include_router(pages.router)
pages.mount_static(app)

_JOBS = {}                     # id -> {kind, state, step, submitted, started, finished, result, error}
_QUEUE = queue.Queue()
KIND_NAME = {"figure": "a figure", "table": "a table", "pdf": "a paper"}


def _visitor(request: Request) -> str:
    # behind front.py the visitor's address arrives in X-FFTM-Visitor; the worker
    # only accepts authenticated calls, so the header cannot come from outside
    forwarded = request.headers.get("x-fftm-visitor")
    if forwarded:
        return forwarded
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _worker():
    """One job at a time: the models share one CPU box and one set of keys."""
    while True:
        job_id, kind, path, visitor = _QUEUE.get()
        job = _JOBS[job_id]
        job.update(state="running", started=time.time(), step="Starting")
        while MODELS_READY and not os.path.exists(MODELS_READY):         # a fresh instance is still fetching its weights
            job["step"] = "Preparing the models"
            time.sleep(3)
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


def _queue_info(job):
    """Where a waiting job stands: jobs ahead and what is running now."""
    ahead = [j for j in _JOBS.values() if j["state"] in ("queued", "running") and j["submitted"] < job["submitted"]]
    running = next((j for j in ahead if j["state"] == "running"), None)
    return {"ahead": len(ahead),
            "now": {"kind": KIND_NAME[running["kind"]], "step": running.get("step")} if running else None}


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
