import os
import uuid
import tempfile
import json
import logging
import shutil
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Try importing Google Cloud Storage for logging uploads
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from src.pipeline.figure_pipeline import FigurePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Figure Extraction Microservice", version="1.0.0")

# Global pipeline instance
pipeline: Optional[FigurePipeline] = None

# GCS settings - configure these via environment variables!
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
ENABLE_GCS_LOGGING = os.environ.get("ENABLE_GCS_LOGGING", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing FigurePipeline on startup...")
    pipeline = FigurePipeline()
    logger.info("FigurePipeline initialized successfully.")

def upload_to_gcs(bucket_name: str, destination_blob_name: str, file_path: str):
    """Uploads a file to the bucket."""
    if not GCP_AVAILABLE or not bucket_name:
        return
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        logger.info(f"File {file_path} uploaded to gcs://{bucket_name}/{destination_blob_name}.")
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")

def upload_json_to_gcs(bucket_name: str, destination_blob_name: str, data: dict):
    if not GCP_AVAILABLE or not bucket_name:
         return
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    except Exception as e:
         logger.error(f"Failed to upload JSON to GCS: {e}")

@app.post("/extract_figure")
async def extract_figure(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Extract data points from a standalone figure image.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported.")
        
    request_id = str(uuid.uuid4())
    logger.info(f"Received extraction request: {request_id} for file {file.filename}")
    
    # Create isolated temp working directory for this request
    temp_dir = tempfile.mkdtemp(prefix=f"figure_{request_id}_")
    
    # The figure pipeline usually writes into an output_base_dir. We'll give it the temp_dir.
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_img_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Write uploaded contents to temp file
        with open(temp_img_path, 'wb') as f:
            f.write(await file.read())
            
        # Optional: Log the incoming image to GCS in the background
        if ENABLE_GCS_LOGGING:
            gcs_img_path = f"uploads/figures/{request_id}_{file.filename}"
            background_tasks.add_task(upload_to_gcs, GCS_BUCKET_NAME, gcs_img_path, temp_img_path)
            
        # Process image using our FigurePipeline
        # The pipeline returns paths to the extracted JSON result files
        extracted_results_paths = pipeline.process_images([temp_img_path], output_base_dir=output_dir)
        
        extracted_data_list = []
        for res_path in extracted_results_paths:
            if os.path.exists(res_path):
                with open(res_path, 'r') as rf:
                    extracted_data_list.append(json.load(rf))
        
        # Prepare response payload
        payload = {
            "request_id": request_id,
            "figure_data": extracted_data_list,
            "message": f"Successfully processed figure. Found {len(extracted_data_list)} sub-plots or evidence blocks."
        }
        
        # Optional: Log output JSON to GCS to capture predictions
        if ENABLE_GCS_LOGGING:
            gcs_res_path = f"results/figures/{request_id}_result.json"
            background_tasks.add_task(upload_json_to_gcs, GCS_BUCKET_NAME, gcs_res_path, payload)

        return JSONResponse(content=payload)

    except Exception as e:
        logger.exception(f"Exception during processing request {request_id}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp directory in the background (to allow GCS uploads to finish)
        # Note: A more robust way is to make sure upload_to_gcs finishes before cleanup,
        # but for demonstration we'll do synchronous cleanup or just let temp files live briefly.
        # Background GCS tasks open file content at start or rely on path. If relying on path,
        # we must delay cleanup. As a safe default, we can just leave it to OS temp cleaning routines
        # or delete it immediately since we mapped `upload_to_gcs` directly.
        # Actually gcs python client reads directly from filename, so deleting it immediately will fail
        # the background upload. We'll read the blob into memory in future iterations or accept 
        # temp directory bloat for now until garbage collected.
        def safe_cleanup(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except:
                pass
        
        # Here we add it as a background task to happen AFTER uploads
        background_tasks.add_task(safe_cleanup, temp_dir)

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
