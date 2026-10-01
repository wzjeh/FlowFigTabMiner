"""What every deployment of the demo serves without the models: the page, its
precomputed examples, structure drawings and the trial limit.

``server.py`` (the whole demo in one process) and ``front.py`` (the light Cloud
Run front that forwards extraction jobs to a worker) both include this router.
"""
import functools
import json
import os

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "examples")
STATIC_DIR = os.path.join(HERE, "static")

router = APIRouter()


def mount_static(app: FastAPI):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")


def tries_per_address() -> int:
    return int(os.environ.get("FFTM_TRIAL_PER_ADDRESS", "3"))


@router.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@router.get("/api/limits")
def limits():
    return {"tries": tries_per_address()}


@router.get("/api/examples")
def examples():
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


@router.get("/api/mol.svg")
def mol_svg(smiles: str, w: int = 180, h: int = 120):
    try:
        svg = _mol_svg(smiles, max(60, min(w, 600)), max(40, min(h, 400)))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "not a structure")
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})
