"""Light front of the Cloud Run demo: serves the page itself and forwards the
extraction API to the worker (``server.py`` in the full image).

    uvicorn front:app --app-dir docker/webapp --port 8080   # FFTM_WORKER_URL=https://...

The worker keeps its queue and jobs in memory, so it runs as a single
instance, and a single-instance service scaled to zero sometimes turns away the
first request ("no available instance").  The front has no state and starts
in a second or two, so it can scale freely: the page always answers, and a
cold worker only makes the first upload wait while the front retries.
"""
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

import pages

WORKER = os.environ["FFTM_WORKER_URL"].rstrip("/")
UPLOAD_WAIT = 240        # s: a cold worker needs about a minute before it takes a job
POLL_WAIT = 20           # s: the page polls again a few seconds later anyway

app = FastAPI(title="FlowFigTabMiner demo", docs_url=None, redoc_url=None)
app.include_router(pages.router)
pages.mount_static(app)

_token = {"value": None, "expires": 0.0}


def _auth_header() -> dict:
    """ID token for the worker, which only accepts authenticated calls (none for a local http worker)."""
    if not WORKER.startswith("https://"):
        return {}
    if time.time() > _token["expires"]:
        import google.auth.transport.requests
        from google.oauth2 import id_token
        _token["value"] = id_token.fetch_id_token(google.auth.transport.requests.Request(), WORKER)
        _token["expires"] = time.time() + 45 * 60            # tokens last an hour
    return {"Authorization": f"Bearer {_token['value']}"}


def _visitor(request: Request) -> str:
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def _not_ready(r: httpx.Response) -> bool:
    """Cloud Run's own refusal while the worker starts, not an answer from the worker's code
    (the worker answers in JSON, e.g. a used-up free try is a JSON 429)."""
    if r.status_code in (502, 503, 504):
        return True
    return r.status_code in (429, 500) and "application/json" not in r.headers.get("content-type", "")


def _forward(method: str, path: str, wait: float, headers=None, content=None) -> httpx.Response:
    deadline, pause = time.time() + wait, 2.0
    while True:
        try:
            r = httpx.request(method, WORKER + path, headers={**(headers or {}), **_auth_header()},
                              content=content, timeout=60)
            if not _not_ready(r) or time.time() > deadline:
                return r
        except httpx.TransportError:
            if time.time() > deadline:
                raise
        time.sleep(pause)
        pause = min(pause * 1.5, 10.0)


def _relay(r: httpx.Response) -> Response:
    keep = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "cache-control")}
    return Response(r.content, status_code=r.status_code, headers=keep)


def _unavailable() -> JSONResponse:
    return JSONResponse({"error": "The extraction server is not responding. Please try again in a few minutes."},
                        status_code=503)


@app.post("/api/jobs")
async def submit(request: Request):
    body = await request.body()
    headers = {"content-type": request.headers.get("content-type", ""), "x-fftm-visitor": _visitor(request)}
    try:
        # sync forwarding in a thread: the retry loop may wait for a cold worker
        r = await run_in_threadpool(_forward, "POST", "/api/jobs", UPLOAD_WAIT, content=body, headers=headers)
    except httpx.TransportError:
        return _unavailable()
    return _relay(r)


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    try:
        r = _forward("GET", f"/api/jobs/{job_id}", POLL_WAIT)
    except httpx.TransportError:
        return _unavailable()
    return _relay(r) if not _not_ready(r) else _unavailable()


@app.get("/files/{job_id}/{name}")
def job_file(job_id: str, name: str):
    try:
        r = _forward("GET", f"/files/{os.path.basename(job_id)}/{os.path.basename(name)}", POLL_WAIT)
    except httpx.TransportError:
        return _unavailable()
    return _relay(r)
