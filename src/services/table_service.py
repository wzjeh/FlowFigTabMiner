"""
table-service: FastAPI service wrapping the table extraction pipeline.
Downloads a single table crop from GCS, runs filter + Gemini transcriber + MolNexTR, uploads CSV to GCS.

POST /extract
  Body: {"image_gcs_uri": "gs://...", "job_id": "..."}
"""
import os
import sys
# Disable PaddlePaddle OneDNN (MKL-DNN) BEFORE any paddle import
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_apply_mkldnn_pass"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLEPD_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import logging
import tempfile
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services._gcs import download_blob, upload_file, upload_string
from src.extraction.table.pipeline import TablePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="table-service")

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    with _pipeline_lock:                 # the startup warm-up and a first request build it once
        if _pipeline is None:
            _pipeline = _build_pipeline()
    return _pipeline


@app.on_event("startup")
def _warm_up():
    """Load every model when the instance starts, not inside the first visitor's request."""
    threading.Thread(target=get_pipeline, daemon=True).start()


def _build_pipeline():
    from src.extraction.table.table_vlm import TableTranscriber
    from src.llm.config import load_table_reader_config
    from src.llm.providers import get_vlm_provider
    cfg = load_table_reader_config("config.yaml")
    return TablePipeline(transcriber=TableTranscriber(vlm=get_vlm_provider(cfg.provider), cfg=cfg),
                         min_text_agreement=cfg.min_text_agreement)


class ExtractRequest(BaseModel):
    image_gcs_uri: str
    job_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest):
    """
    Response schema:
    {
      "status": "success" | "filtered_yolo" | "filtered_keywords" | "no_cells" | "error",
      "is_relevant": bool,
      "csv_gcs_uri": "gs://..." | null,
      "row_count": int,
      "caption_text": str
    }
    """
    job_id = req.job_id
    image_uri = req.image_gcs_uri
    image_name = os.path.splitext(os.path.basename(image_uri))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_img = os.path.join(tmpdir, f"{image_name}.png")
        try:
            download_blob(image_uri, local_img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GCS download failed: {e}")

        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir, exist_ok=True)

        try:
            pipeline = get_pipeline()
            result = pipeline.process_table(local_img, output_dir=out_dir)
        except Exception as e:
            logger.exception("Table pipeline failed")
            return {
                "status": "error",
                "is_relevant": False,
                "csv_gcs_uri": None,
                "row_count": 0,
                "caption_text": "",
            }

        if not result or (not result.get("is_valid") and result.get("reason") != "Filtered by YOLO"):
            return {
                "status": "error",
                "is_relevant": False,
                "csv_gcs_uri": None,
                "row_count": 0,
                "caption_text": "",
            }

        # Check for hard filter results (YOLO filter)
        if result.get("reason") == "Filtered by YOLO":
            return {
                "status": "filtered_yolo",
                "is_relevant": False,
                "csv_gcs_uri": None,
                "row_count": 0,
                "caption_text": "",
            }

        is_relevant = result.get("is_relevant", True)
        csv_path = result.get("csv_path")
        df = result.get("dataframe")
        n_rows = 0 if df is None else max(0, len(df) - int(result.get("header_row_count") or 0))
        caption_text = result.get("caption_text", "")

        # Upload CSV to GCS regardless of relevance (soft marking)
        csv_uri = None
        if csv_path and os.path.exists(csv_path):
            bucket = "flowfigtabminer-data"
            blob_name = f"output/{job_id}/tables/{image_name}.csv"
            try:
                upload_file(csv_path, bucket, blob_name)
                csv_uri = f"gs://{bucket}/{blob_name}"
            except Exception as e:
                logger.warning(f"CSV upload failed: {e}")

        status = "success" if is_relevant else "filtered_keywords"
        if not n_rows and is_relevant:
            status = "no_cells"

        return {
            "status": status,
            "is_relevant": is_relevant,
            "csv_gcs_uri": csv_uri,
            "row_count": n_rows,
            "caption_text": caption_text,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
