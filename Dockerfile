FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_RUNTIME_DIR=/runtime \
    MLFLOW_TRACKING_URI=sqlite:////runtime/mlflow.db \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# XGBoost needs the OpenMP runtime; no compiler toolchain is retained.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip check

COPY src/ ./src/
RUN mkdir -p /runtime

EXPOSE 8000
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
