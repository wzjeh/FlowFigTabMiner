#!/bin/bash
# Fetch the custom weights on first run, then hand over to the pipeline CLI.
set -e

MODELS=/app/models
WEIGHTS_REPO="${FFTM_WEIGHTS_REPO:-wyzhaoc/FlowFigTabMiner-models}"

if [ ! -f "$MODELS/molnextr_model_best.pth" ]; then
    echo "[entrypoint] downloading weights from hf.co/$WEIGHTS_REPO into $MODELS (about 1.4 GB, once)"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$WEIGHTS_REPO', local_dir='$MODELS')"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "[entrypoint] WARNING: GEMINI_API_KEY is not set; steps 4.4/4.5/5 and the VLM readers will fail" >&2
fi

exec python -m src.pipeline.main "$@"
