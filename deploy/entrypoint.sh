#!/bin/bash
# At container startup, link GCS-mounted models into /app/models so hardcoded
# relative paths like "models/best.pt" resolve correctly.
#
# Files named in MODELS_LOCAL_COPY (space separated) are copied to local disk
# instead: torch.load reads a checkpoint in many small random reads, which over
# the GCS mount took 14 min for the 1.1 GB MolNexTR weights; one sequential
# copy takes well under a minute.
set -e
if [ -d "/models/models" ]; then
    rm -rf /app/models 2>/dev/null || true
    if [ -n "${MODELS_LOCAL_COPY:-}" ]; then
        mkdir -p /app/models
        for entry in /models/models/*; do
            ln -sf "$entry" "/app/models/$(basename "$entry")"
        done
        for name in $MODELS_LOCAL_COPY; do
            if [ -f "/models/models/$name" ]; then
                rm -f "/app/models/$name"
                t0=$(date +%s)
                cp "/models/models/$name" "/app/models/$name"
                echo "[entrypoint] copied $name to local disk in $(( $(date +%s) - t0 )) s"
            fi
        done
        echo "[entrypoint] /app/models: links to /models/models, local copies: $MODELS_LOCAL_COPY"
    else
        ln -sf /models/models /app/models
        echo "[entrypoint] /app/models -> /models/models"
    fi
else
    echo "[entrypoint] WARNING: /models/models not found, model paths may fail"
fi
exec "$@"
