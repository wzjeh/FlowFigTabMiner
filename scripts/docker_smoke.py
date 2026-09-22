"""Smoke test for the Docker image: every model loads and runs once, no API key.

  docker run --rm --entrypoint python -v $PWD/models:/app/models \
      ghcr.io/wzjeh/flowfigtabminer scripts/docker_smoke.py [paper.pdf]

Without a PDF the TF-ID step renders a blank page.  Exit code 0 only when all
five checks pass.
"""
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = {}


def check(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        results[name] = (True, f"{out} ({time.perf_counter() - t0:.1f}s)")
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not hide
        results[name] = (False, f"{type(exc).__name__}: {exc}")
    print(f"[{'ok' if results[name][0] else 'FAIL'}] {name}: {results[name][1]}", flush=True)


def page_image():
    if len(sys.argv) > 1:
        import fitz
        page = fitz.open(sys.argv[1])[0]
        pix = page.get_pixmap(dpi=100)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        return np.ascontiguousarray(img)
    return np.full((800, 600, 3), 255, dtype=np.uint8)


def tfid():
    from src.parsing.active_area_detector import ActiveAreaDetector
    det = ActiveAreaDetector()
    from PIL import Image
    out = det.detect_tables_figures(Image.fromarray(page_image()))
    return f"device={det.device} detections={str(out)[:60]}"


def yolo():
    from ultralytics import YOLO
    cfg = yaml.safe_load(open("config.yaml"))
    paths = []
    for section in cfg.values():
        if isinstance(section, dict):
            for v in section.values():
                if isinstance(v, dict) and "model_path" in v:
                    paths.append(v["model_path"])
    img = page_image()
    for p in paths:
        YOLO(p).predict(img, verbose=False)
    return f"{len(paths)} models"


def paddle():
    from src.extraction.common.ocr_backend import get_ocr_instance
    eng = get_ocr_instance()
    img = np.full((80, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Yield 85%", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    out = eng.ocr(img)
    return str(out)[:80]


def molnextr():
    from rdkit import Chem
    from rdkit.Chem import Draw
    from src.extraction.common.content_recognizer import ContentRecognizer
    img = np.array(Draw.MolToImage(Chem.MolFromSmiles("c1ccccc1Br"), size=(300, 300)))
    smi = ContentRecognizer().recognize_content(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), "Structure")
    return f"bromobenzene -> {smi}"


def weights():
    pdx = os.environ.get("PADDLE_PDX_CACHE_HOME", os.path.expanduser("~/.paddlex"))
    need = ["models/molnextr_model_best.pth", "models/tab-scheme-seg/best.pt",
            os.path.join(pdx, "official_models/PP-OCRv5_server_det/inference.pdiparams")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(missing)
    return "all present"


check("weights", weights)
check("yolo", yolo)
check("molnextr", molnextr)
check("paddleocr", paddle)
check("tfid", tfid)
sys.exit(0 if all(ok for ok, _ in results.values()) else 1)
