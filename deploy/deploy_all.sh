#!/usr/bin/env bash
# Deploy all FlowFigTabMiner Cloud Run services to GCP project: FlowFigTabMiner
# Run from the project root directory.
set -euo pipefail

PROJECT_ID="gen-lang-client-0080522548"
REGION="us-central1"

echo "=== [0] Uploading model weights to GCS (only needed once) ==="
echo "    Skipping if already uploaded. Run manually if needed:"
echo "    gsutil -m cp -r models/ gs://flowfigtabminer-models/"
# Uncomment to upload:
# gsutil -m cp -r models/ gs://flowfigtabminer-models/

echo ""
echo "=== [1] Building & pushing tfid-service ==="
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --config deploy/cloudbuild-tfid.yaml \
  .

echo ""
echo "=== [2] Building & pushing figure-service ==="
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --config deploy/cloudbuild-figure.yaml \
  .

echo ""
echo "=== [3] Building & pushing table-service ==="
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --config deploy/cloudbuild-table.yaml \
  .

echo ""
echo "=== [4] Deploying services to Cloud Run (${REGION}) ==="
gcloud run services replace deploy/tfid-service.yaml \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

gcloud run services replace deploy/figure-service.yaml \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

gcloud run services replace deploy/table-service.yaml \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

echo ""
echo "=== [5] Setting public access (--allow-unauthenticated) ==="
for SVC in tfid-service figure-service table-service; do
  gcloud run services add-iam-policy-binding "${SVC}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --member="allUsers" \
    --role="roles/run.invoker"
done

echo ""
echo "=== Deployment complete ==="
echo "Service URLs:"
for SVC in tfid-service figure-service table-service; do
  URL=$(gcloud run services describe "${SVC}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(status.url)")
  echo "  ${SVC}: ${URL}"
done
