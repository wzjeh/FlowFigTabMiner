#!/bin/bash
# Fetch the custom weights on first run, then hand over to the pipeline CLI.
set -e

MODELS=/app/models
WEIGHTS_REPO="${FFTM_WEIGHTS_REPO:-wyzhaoc/FlowFigTabMiner-models}"

if [ ! -f "$MODELS/molnextr_model_best.pth" ]; then
    echo "[entrypoint] downloading weights from hf.co/$WEIGHTS_REPO into $MODELS (about 1.4 GB, once)"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$WEIGHTS_REPO', local_dir='$MODELS')"
fi
# TF-ID (Florence-2, ~1 GB, with its remote code) into the HF cache on the same
# volume; a lazy download inside the first PDF job fails too often to rely on.
python - <<'PY'
import time
from huggingface_hub import snapshot_download
for attempt in range(3):
    try:
        snapshot_download("yifeihu/TF-ID-base")
        break
    except Exception as exc:
        print(f"[entrypoint] TF-ID download attempt {attempt + 1} failed: {exc}")
        time.sleep(10)
PY

if [ -z "$GEMINI_API_KEY" ] && [ "$1" != "web" ]; then
    echo "[entrypoint] WARNING: GEMINI_API_KEY is not set; steps 4.4/4.5/5 and the VLM readers will fail" >&2
fi

if [ "$1" = "web" ]; then
    shift
    exec uvicorn server:app --app-dir docker/webapp --host 0.0.0.0 --port "${PORT:-7860}" "$@"
fi
exec python -m src.pipeline.main "$@"
