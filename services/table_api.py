import os
import io
import uuid
import tempfile
import json
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd

# Try importing Google Cloud Storage for logging uploads
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from src.extraction.table.pipeline import TablePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Table Extraction Microservice", version="1.0.0")

# Global pipeline instance
pipeline: Optional[TablePipeline] = None

# GCS settings - configure these via environment variables!
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
ENABLE_GCS_LOGGING = os.environ.get("ENABLE_GCS_LOGGING", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing TablePipeline on startup...")
    # Load model in sequential mode to save memory in cloud containers if needed,
    # or keep sequential_mode=False if you have enough RAM.
    pipeline = TablePipeline(sequential_mode=True)
    logger.info("TablePipeline initialized successfully.")

def upload_to_gcs(bucket_name: str, destination_blob_name: str, file_path: str):
    """Uploads a file to the bucket."""
    if not GCP_AVAILABLE or not bucket_name:
        logger.warning("GCS upload disabled or missing dependencies/bucket name.")
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
        logger.info(f"JSON data uploaded to gcs://{bucket_name}/{destination_blob_name}.")
    except Exception as e:
         logger.error(f"Failed to upload JSON to GCS: {e}")

@app.post("/extract_table")
async def extract_table(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported.")
        
    request_id = str(uuid.uuid4())
    logger.info(f"Received extraction request: {request_id} for file {file.filename}")
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    try:
        # Write uploaded contents to temp file
        with os.fdopen(fd, 'wb') as f:
            f.write(await file.read())
            
        # Optional: Log the incoming image to GCS in the background
        if ENABLE_GCS_LOGGING:
            gcs_img_path = f"uploads/{request_id}_{file.filename}"
            background_tasks.add_task(upload_to_gcs, GCS_BUCKET_NAME, gcs_img_path, temp_path)
            
        # Process image using our TablePipeline
        # output_dir=None means it won't write intermediate crops to disk
        result = pipeline.process_table(temp_path, output_dir=None)
        
        if not result.get('is_valid'):
            error_msg = result.get('reason', 'Unknown parsing error')
            logger.warning(f"Request {request_id} failed: {error_msg}")
            return JSONResponse(status_code=422, content={"is_valid": False, "reason": error_msg})

        # Extract pandas DataFrame from the result and convert strictly to JSON
        df: pd.DataFrame = result.get('dataframe')
        
        # Prepare response payload
        payload = {
            "request_id": request_id,
            "is_valid": True,
            "table_data": df.fillna("").to_dict(orient="records") if df is not None else [],
            "message": "Successfully extracted table grid."
        }
        
        # Optional: Log output JSON to GCS to capture predictions
        if ENABLE_GCS_LOGGING:
            gcs_res_path = f"results/{request_id}_result.json"
            background_tasks.add_task(upload_json_to_gcs, GCS_BUCKET_NAME, gcs_res_path, payload)

        return JSONResponse(content=payload)

    except Exception as e:
        logger.exception(f"Exception during processing request {request_id}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
             os.remove(temp_path)

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
