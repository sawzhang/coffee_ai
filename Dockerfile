# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy dependency definition first for layer caching
COPY research/pyproject.toml research/pyproject.toml

# Install build deps and project deps
RUN pip install --no-cache-dir hatchling \
    && pip install --no-cache-dir "research/[api]"

# ── Stage 2: Runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ api/
COPY research/prepare_v2.py research/prepare_v2.py
COPY research/prepare.py research/prepare.py
COPY research/data/beans.json research/data/beans.json
COPY research/data/schema.json research/data/schema.json
COPY site/ site/

# Copy model files if they exist (use glob pattern; build won't fail if missing)
COPY research/model.jobli* research/model.pk* research/

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
