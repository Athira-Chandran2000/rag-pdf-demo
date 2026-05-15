# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — ArXiv RAG Demo for Hugging Face Spaces (Docker SDK)
#
# Build locally:
#   docker build -t arxiv-rag-demo .
#   docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... arxiv-rag-demo
#
# HF Spaces: auto-builds on every push to main via GitHub → HF Space link.
#
# Port 7860 is required by Hugging Face Spaces.
# GROQ_API_KEY must be set as a Space secret (never committed to Git).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces requirement) ────────────────────────────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Install pip upgrade first, then requirements (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download models at build time (avoids cold-start penalty) ─────────────
# all-MiniLM-L6-v2 (~22 MB) + Flashrank ms-marco-MiniLM-L-12-v2 (~60 MB)
# Both are baked into the image so first request is fast.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
from flashrank import Ranker; \
Ranker(model_name='ms-marco-MiniLM-L-12-v2') \
"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Create required directories with correct permissions ──────────────────────
RUN mkdir -p data/pdfs data/processed experiments/mlruns experiments/eval_results tests && \
    chown -R appuser:appuser /app

USER appuser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Run ───────────────────────────────────────────────────────────────────────
# kaggle_build.py checks if indexes exist — skips ingestion if they do (fast restarts).
# Then uvicorn starts the FastAPI app on HF Spaces' required port 7860.
CMD python kaggle_build.py && \
    uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 7860 \
        --log-level info \
        --workers 1
