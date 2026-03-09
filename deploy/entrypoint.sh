#!/bin/bash
# At container startup, symlink GCS-mounted models to /app/models
# so hardcoded relative paths like "models/best.pt" resolve correctly.
set -e
if [ -d "/models/models" ]; then
    rm -rf /app/models 2>/dev/null || true
    ln -sf /models/models /app/models
    echo "[entrypoint] /app/models -> /models/models"
else
    echo "[entrypoint] WARNING: /models/models not found, model paths may fail"
fi
exec "$@"
