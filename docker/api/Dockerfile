FROM python:3.12-slim

WORKDIR /app

# Install shared serving deps then API-specific deps.
# We do this in a single layer so Docker caches it until requirements change.
COPY requirements/serving.txt requirements/api.txt /tmp/req/
RUN pip install --no-cache-dir \
        -r /tmp/req/serving.txt \
        -r /tmp/req/api.txt && \
    # XGBoost 3.x pulls in nvidia-nccl-cu12 (303 MB) for distributed GPU training.
    # We only do single-machine CPU inference, so remove it to keep the image lean.
    pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true

# Copy only source code and config; models/ and data/ are mounted at runtime.
COPY src/ src/
COPY config/ config/

ENV PYTHONPATH=/app \
    USE_GCS=false

# models/ and data/ must be mounted via docker-compose volumes:
#   ./models -> /app/models   (backbone + WT-proxy pkl + QSARTrainer metadata)
#   ./data   -> /app/data     (interim CSVs for applicability domain fitting)

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
