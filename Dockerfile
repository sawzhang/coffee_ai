# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution, then install all deps
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache \
    numpy scipy scikit-learn pandas matplotlib joblib \
    fastapi uvicorn pydantic

# ── Stage 2: Model training ─────────────────────────────────────────
FROM builder AS trainer

WORKDIR /app

# Copy research code and data for training
COPY research/ research/

# Train model to generate model.joblib and model_quantiles.joblib
RUN cd research && python3 train_v2.py

# ── Stage 3: Runtime ────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ api/
COPY research/prepare_v2.py research/prepare_v2.py
COPY research/prepare.py research/prepare.py
COPY research/flavor_wheel.py research/flavor_wheel.py
COPY research/data/beans.json research/data/beans.json
COPY research/data/schema.json research/data/schema.json
COPY research/data/grinder_calibration.json research/data/grinder_calibration.json
COPY research/data/brew_schema.json research/data/brew_schema.json
COPY site/ site/

# Copy trained models from trainer stage
COPY --from=trainer /app/research/model.joblib research/model.joblib
COPY --from=trainer /app/research/model_quantiles.joblib research/model_quantiles.joblib

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
