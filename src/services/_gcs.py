"""
GCS helper utilities shared by all services.
"""
import os
import re


def _parse_gcs_uri(uri: str):
    """Parse gs://bucket/blob into (bucket, blob)."""
    m = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return m.group(1), m.group(2)


def download_blob(gcs_uri: str, dest_path: str):
    from google.cloud import storage
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    blob.download_to_filename(dest_path)


def upload_file(local_path: str, bucket_name: str, blob_name: str):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def upload_string(content: str, bucket_name: str, blob_name: str, content_type="text/csv"):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type=content_type)


def list_blobs(bucket_name: str, prefix: str):
    from google.cloud import storage
    client = storage.Client()
    return [b.name for b in client.list_blobs(bucket_name, prefix=prefix)]
