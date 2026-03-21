"""
figure-service: FastAPI service wrapping the figure extraction pipeline.
Downloads a single figure crop from GCS, runs Steps 2-4, uploads CSV to GCS.

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
import paddle
try:
    paddle.set_flags({'FLAGS_use_mkldnn': False, 'FLAGS_pir_apply_mkldnn_pass': False})
except Exception:
    pass
import logging
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services._gcs import download_blob, upload_file
from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.figure.legend_matcher import LegendMatcher
from src.extraction.figure.coordinate_mapper import CoordinateMapper
from src.assembly.evidence_assembler import EvidenceAssembler
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="figure-service")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        cfg = load_config()
        fig_cfg = cfg.get("figures", {})
        macro_cfg = fig_cfg.get("step2_macro", {})
        micro_cfg = fig_cfg.get("step3_micro", {})

        macro_model = macro_cfg.get("model_path", "models/bestYOLOn-2-1.pt")
        micro_model = micro_cfg.get("model_path", "models/bestYOLOm-2-2.pt")
        macro_conf = macro_cfg.get("confidence_threshold", 0.5)
        micro_conf = micro_cfg.get("confidence_threshold", 0.25)

        yolo_macro = YoloDetector(model_path=macro_model)
        yolo_micro = Stage2Detector(model_path=micro_model)
        legend_matcher = LegendMatcher(yolo_model=yolo_micro)
        coord_mapper = CoordinateMapper()
        assembler = EvidenceAssembler()

        _pipeline = {
            "yolo_macro": yolo_macro,
            "yolo_micro": yolo_micro,
            "legend_matcher": legend_matcher,
            "coord_mapper": coord_mapper,
            "assembler": assembler,
            "macro_conf": macro_conf,
            "micro_conf": micro_conf,
        }
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
      "status": "success" | "no_scatter_points" | "no_data_points" | "error",
      "figure_type": "scatter" | "line" | "unknown",
      "csv_gcs_uri": "gs://...",
      "row_count": int
    }
    """
    import pandas as pd

    job_id = req.job_id
    image_uri = req.image_gcs_uri
    image_name = os.path.splitext(os.path.basename(image_uri))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_img = os.path.join(tmpdir, f"{image_name}.png")
        try:
            download_blob(image_uri, local_img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GCS download failed: {e}")

        p = get_pipeline()

        # Step 2: Macro cleaning
        try:
            macro_results = p["yolo_macro"].process_images(
                [local_img],
                output_base_dir=tmpdir,
                output_subdir_name="macro_cleaned",
                imgsz=1024,
            )
        except Exception as e:
            logger.exception("Macro detection failed")
            return {"status": "error", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        if not macro_results:
            return {"status": "no_scatter_points", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        item = macro_results[0]
        cleaned_plot_path = item["cleaned_image"]
        elements = item["elements"]
        macro_cleaned_dir = os.path.dirname(cleaned_plot_path)

        # Relevance check
        is_relevant, text_evidence = p["assembler"].check_relevance(image_name, macro_cleaned_dir)
        if not is_relevant:
            return {"status": "no_scatter_points", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        # Step 3: Micro detection + coord mapping
        try:
            micro_detections = p["yolo_micro"].detect(
                cleaned_plot_path, conf=p["micro_conf"], imgsz=1024, use_tiling=False
            )
        except Exception as e:
            logger.exception("Micro detection failed")
            return {"status": "error", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        points = [d for d in micro_detections if d["label"] in ["data_point", "marker"]]
        if not points:
            return {"status": "no_data_points", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        legend_crops = elements.get("legend", [])
        prototypes = p["legend_matcher"].parse_legend_crops(legend_crops)
        matched_points = p["legend_matcher"].match_points(points, prototypes, cleaned_plot_path)

        other_detections = [d for d in micro_detections if d["label"] not in ["data_point", "marker"]]
        full_detections = other_detections + matched_points

        try:
            df, _ = p["coord_mapper"].map_coordinates(full_detections, cleaned_plot_path)
        except Exception:
            df = pd.DataFrame()

        extraction_data = df.to_dict(orient="records") if not df.empty else [
            {"series": pt.get("series", "Unknown"), "x_pixel": pt["center"][0], "y_pixel": pt["center"][1]}
            for pt in matched_points
        ]

        # Step 4: Assemble
        json_path = p["assembler"].assemble(image_name, extraction_data, macro_cleaned_dir, text_evidence=text_evidence)
        if not json_path:
            return {"status": "no_data_points", "figure_type": "unknown", "csv_gcs_uri": None, "row_count": 0}

        # Build CSV from df or extraction_data
        if not df.empty:
            csv_content = df.to_csv(index=False)
            row_count = len(df)
        else:
            import io
            tmp_df = pd.DataFrame(extraction_data)
            csv_content = tmp_df.to_csv(index=False)
            row_count = len(tmp_df)

        # Upload CSV to GCS
        bucket = "flowfigtabminer-data"
        blob_name = f"output/{job_id}/figures/{image_name}.csv"
        try:
            from src.services._gcs import upload_string
            upload_string(csv_content, bucket, blob_name)
            csv_uri = f"gs://{bucket}/{blob_name}"
        except Exception as e:
            logger.warning(f"CSV upload failed: {e}")
            csv_uri = None

        figure_type = "scatter" if not df.empty else "unknown"
        return {
            "status": "success",
            "figure_type": figure_type,
            "csv_gcs_uri": csv_uri,
            "row_count": row_count,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
