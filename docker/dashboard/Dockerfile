FROM python:3.12-slim

WORKDIR /app

# Install shared serving deps then dashboard-specific deps (streamlit, altair, requests).
COPY requirements/serving.txt requirements/dashboard.txt /tmp/req/
RUN pip install --no-cache-dir \
        -r /tmp/req/serving.txt \
        -r /tmp/req/dashboard.txt && \
    # XGBoost 3.x pulls in nvidia-nccl-cu12 (303 MB) for distributed GPU training.
    # We only do single-machine CPU inference, so remove it to keep the image lean.
    pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true

# Copy source code and config; models/ and data/ are mounted at runtime.
COPY src/ src/
COPY config/ config/

# API_BASE_URL lets the dashboard reach the API by Docker service name.
# Override to http://127.0.0.1:8000 for local (non-Docker) use.
ENV PYTHONPATH=/app \
    USE_GCS=false \
    API_BASE_URL=http://api:8000

# models/ and data/ must be mounted via docker-compose volumes:
#   ./models -> /app/models   (final_ranked_candidates and docking JSON files)
#   ./data   -> /app/data     (generated/final_ranked_candidates.csv)

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
