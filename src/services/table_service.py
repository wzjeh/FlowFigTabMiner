"""
table-service: FastAPI service wrapping the table extraction pipeline.
Downloads a single table crop from GCS, runs filter+TATR+OCR+MolNexTR, uploads CSV to GCS.

POST /extract
  Body: {"image_gcs_uri": "gs://...", "job_id": "..."}
"""
import os
import sys
import logging
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services._gcs import download_blob, upload_file, upload_string
from src.extraction.table.pipeline import TablePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="table-service")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TablePipeline(sequential_mode=False)
    return _pipeline


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
      "cell_count": int,
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
                "cell_count": 0,
                "caption_text": "",
            }

        if not result:
            return {
                "status": "error",
                "is_relevant": False,
                "csv_gcs_uri": None,
                "cell_count": 0,
                "caption_text": "",
            }

        # Check for hard filter results (YOLO filter)
        if result.get("reason") == "YOLO_filter":
            return {
                "status": "filtered_yolo",
                "is_relevant": False,
                "csv_gcs_uri": None,
                "cell_count": 0,
                "caption_text": "",
            }

        is_relevant = result.get("is_relevant", True)
        csv_path = result.get("csv_path")
        cells = result.get("cells", [])
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
        if not cells and is_relevant:
            status = "no_cells"

        return {
            "status": status,
            "is_relevant": is_relevant,
            "csv_gcs_uri": csv_uri,
            "cell_count": len(cells),
            "caption_text": caption_text,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
