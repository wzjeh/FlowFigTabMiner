"""
tfid-service: FastAPI service wrapping TF-ID (Florence-2) detection.
Reads a PDF from GCS, runs paper filter + TF-ID, saves crops to GCS.

POST /detect
  Body: {"pdf_gcs_uri": "gs://...", "job_id": "..."}
  Response: see docstring below
"""
import os
import sys
import logging
import tempfile
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src is importable when running from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.paper_filter import filter_paper
from src.parsing.active_area_detector import ActiveAreaDetector
from src.services._gcs import download_blob, upload_file, list_blobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload Florence-2 model at startup so first request isn't slow."""
    global _detector
    logger.info("Loading TF-ID model at startup...")
    _detector = ActiveAreaDetector()
    logger.info("TF-ID model ready.")
    yield
    _detector = None

app = FastAPI(title="tfid-service", lifespan=lifespan)

def get_detector():
    return _detector


class DetectRequest(BaseModel):
    pdf_gcs_uri: str
    job_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(req: DetectRequest):
    """
    1. Download PDF from GCS to temp file.
    2. Run paper filter (first 3 pages keyword check).
    3. Run TF-ID detection (Florence-2).
    4. Crop figures/tables, upload to GCS intermediate bucket.
    5. Return lists of GCS URIs.

    Response schema:
    {
      "status": "success" | "filtered" | "error",
      "filter_reason": str,          // only when filtered
      "figures": ["gs://..."],       // GCS URIs of figure crops
      "tables":  ["gs://..."]        // GCS URIs of table crops
    }
    """
    job_id = req.job_id
    pdf_uri = req.pdf_gcs_uri

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download PDF
        local_pdf = os.path.join(tmpdir, "input.pdf")
        try:
            download_blob(pdf_uri, local_pdf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GCS download failed: {e}")

        # 2. Paper filter
        filter_result = filter_paper(local_pdf)
        if not filter_result["is_relevant"]:
            return {
                "status": "filtered",
                "filter_reason": filter_result["reason"],
                "figures": [],
                "tables": [],
            }

        # 3. TF-ID detection
        try:
            detector = get_detector()
            all_detections = detector.process_pdf(local_pdf)
        except Exception as e:
            logger.exception("TF-ID detection failed")
            raise HTTPException(status_code=500, detail=f"TF-ID error: {e}")

        # 4. Save crops locally then upload to GCS
        crop_dir = os.path.join(tmpdir, "crops")
        saved_paths = detector.save_crops(local_pdf, all_detections, crop_dir)

        bucket = "flowfigtabminer-data"
        figure_uris = []
        table_uris = []

        for local_path in saved_paths:
            rel = os.path.relpath(local_path, crop_dir)  # e.g. figures/page_1_figure_0.png
            gcs_blob = f"intermediate/{job_id}/{rel}"
            try:
                upload_file(local_path, bucket, gcs_blob)
                uri = f"gs://{bucket}/{gcs_blob}"
                if rel.startswith("figures"):
                    figure_uris.append(uri)
                else:
                    table_uris.append(uri)
            except Exception as e:
                logger.warning(f"Failed to upload {local_path}: {e}")

        return {
            "status": "success",
            "figures": figure_uris,
            "tables": table_uris,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
