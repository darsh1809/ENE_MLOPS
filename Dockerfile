# ============================================================
# Production Dockerfile for Customer Segmentation API
# ============================================================
# Base: slim Python 3.11 — minimal attack surface, small image
FROM python:3.11-slim

# Metadata
LABEL maintainer="darshit1809"
LABEL project="customer-segmentation-mlops"

# ── Environment ────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    LOG_LEVEL=INFO \
    MLFLOW_TRACKING_URI=http://54.206.46.48:5000 \
    MLFLOW_EXPERIMENT=customer

# ── System deps ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────
# Copy requirements first (layer-caching optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────
COPY app/           ./app/
COPY wsgi.py        .
# Optional: pre-baked models for offline fallback
COPY models/        ./models/

# ── Non-root user (security best practice) ─────────────────
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# ── Expose API port ────────────────────────────────────────
EXPOSE ${PORT}

# ── Health check ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# ── Start with Gunicorn ────────────────────────────────────
CMD ["sh", "-c", \
     "gunicorn --bind 0.0.0.0:${PORT} \
               --workers 2 \
               --timeout 120 \
               --access-logfile - \
               --error-logfile - \
               'wsgi:create_app()'"]
