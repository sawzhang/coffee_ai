"""
Coffee Attribution AutoResearch — API Server
=============================================
FastAPI backend providing model prediction, factor attribution,
bean recommendation, and factor recombination endpoints.

Run: uvicorn api.server:app --reload --port 8000
  or: python3 -m api.server
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add research/ to path
RESEARCH_DIR = Path(__file__).parent.parent / "research"
sys.path.insert(0, str(RESEARCH_DIR))

from prepare_v2 import (  # noqa: E402
    encode_factors_v2, encode_factors_v2_extended,
    get_feature_names_v2, get_feature_names_v2_extended,
    load_data, FEATURE_DIM_V2, FEATURE_DIM_V2_EXTENDED,
    PROCESS_METHODS,
)

# ── App state (loaded once at startup, read-only thereafter) ─────────

class _AppState:
    """Container for app-wide state loaded during startup."""
    model = None
    quantile_models: dict = {}  # {"low": Pipeline, "high": Pipeline}
    beans_all: list = []
    feature_names: list = []
    bean_predictions: list = []  # pre-computed predictions for all beans
    _json_cache: dict = {}

    @classmethod
    def get_cached_json(cls, path: Path) -> dict | list | None:
        """Load a JSON file once and return cached content on subsequent calls."""
        key = str(path)
        if key not in cls._json_cache:
            if path.exists():
                with open(path) as f:
                    cls._json_cache[key] = json.load(f)
            else:
                return None
        return cls._json_cache[key]


state = _AppState()


def _load_model():
    """Load trained sklearn pipeline from joblib."""
    import joblib
    from sklearn.pipeline import Pipeline
    model_path = RESEARCH_DIR / "model.joblib"
    # Fallback to legacy pickle if joblib not found
    if not model_path.exists():
        model_path = RESEARCH_DIR / "model.pkl"
    if model_path.exists():
        loaded = joblib.load(model_path)
        if not isinstance(loaded, Pipeline):
            print(f"WARNING: loaded model is not a Pipeline, got {type(loaded)}")
        else:
            state.model = loaded
            print(f"Model loaded from {model_path}")
    else:
        print("WARNING: no model file found. Run train_v2.py first.")

    # Load quantile models for prediction intervals
    quantile_path = RESEARCH_DIR / "model_quantiles.joblib"
    if quantile_path.exists():
        state.quantile_models = joblib.load(quantile_path)
        print(f"Quantile models loaded from {quantile_path}")
    else:
        print("NOTE: no quantile models found. Prediction intervals unavailable.")


def _load_beans():
    """Load full bean dataset for recommendation."""
    beans_path = str(RESEARCH_DIR / "data" / "beans.json")
    train_data, val_data = load_data(beans_path)
    state.beans_all = train_data + val_data
    state.feature_names = get_feature_names_v2()
    print(f"Loaded {len(state.beans_all)} beans")


def _precompute_predictions():
    """Pre-compute predictions for all beans at startup for fast recommendations."""
    if not state.model or not state.beans_all:
        return
    import time
    start = time.time()
    for bean in state.beans_all:
        score = predict_single(bean)
        state.bean_predictions.append(score)
    elapsed = time.time() - start
    print(f"Pre-computed {len(state.bean_predictions)} predictions in {elapsed:.1f}s")


@asynccontextmanager
async def lifespan(app):
    _load_model()
    _load_beans()
    _precompute_predictions()
    yield


app = FastAPI(title="Coffee Attribution API", version="2.0", lifespan=lifespan)

# ── Logging middleware ────────────────────────────────────────────────
import logging
import time as _time

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("coffee_api")


@app.middleware("http")
async def log_requests(request, call_next):
    start = _time.time()
    response = await call_next(request)
    elapsed_ms = (_time.time() - start) * 1000
    if request.url.path.startswith("/api/"):
        logger.info(f"{request.method} {request.url.path} {response.status_code} {elapsed_ms:.0f}ms")
    return response


ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8080,http://localhost:8000,https://sawzhang.github.io"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────

class GFactors(BaseModel):
    variety: str = "Bourbon"
    altitude_m: float = Field(1600, ge=0, le=6000)
    country: str = "Colombia"
    region: str = "Huila"
    soil_type: str = "volcanic"
    shade_pct: float = Field(30, ge=0, le=100)
    latitude: float = Field(4, ge=-90, le=90)
    delta_t_c: float = Field(11, ge=0, le=40)

class PFactors(BaseModel):
    method: str = "washed"
    anaerobic: bool = False
    fermentation_hours: float = Field(24, ge=0, le=500)
    drying_method: str = "raised_bed"
    drying_days: float = Field(12, ge=0, le=90)

class RFactors(BaseModel):
    roast_level: str = "medium_light"
    first_crack_temp_c: float = 198
    drop_temp_c: float = 205
    dtr_pct: float = 22
    total_time_s: float = 600

class BFactors(BaseModel):
    method: str = "v60"
    grind_microns: float = 600
    water_temp_c: float = 93
    ratio: float = 15
    brew_time_s: float = 240
    water_tds_ppm: float = 120

class PredictRequest(BaseModel):
    G: GFactors = GFactors()
    P: PFactors = PFactors()
    R: RFactors = RFactors()
    B: BFactors = BFactors()

class UserPrefs(BaseModel):
    acidity: float = Field(5, ge=0, le=10)
    sweetness: float = Field(7, ge=0, le=10)
    complexity: float = Field(5, ge=0, le=10)
    fermentation: float = Field(5, ge=0, le=10)
    body: float = Field(5, ge=0, le=10)

class RecommendRequest(BaseModel):
    prefs: UserPrefs = UserPrefs()
    top_k: int = Field(10, ge=1, le=50)

class ExploreRequest(BaseModel):
    """Fix G factor, vary P to explore combinations."""
    G: GFactors = GFactors()
    vary_methods: list[str] = Field(default_factory=lambda: PROCESS_METHODS)
    vary_anaerobic: list[bool] = Field(default_factory=lambda: [False, True])
    vary_fermentation: list[float] = Field(default_factory=lambda: [24, 48, 72, 120])

class CompareRequest(BaseModel):
    bean_a: PredictRequest = PredictRequest()
    bean_b: PredictRequest = PredictRequest()


# ── Helpers ───────────────────────────────────────────────────────────

def bean_from_request(req: PredictRequest) -> dict:
    """Convert API request to bean dict for model."""
    return {
        "G": req.G.model_dump(),
        "P": req.P.model_dump(),
        "R": req.R.model_dump(),
        "B": req.B.model_dump(),
    }


def predict_single(bean: dict) -> float:
    """Predict score for a single bean using V2 features."""
    if state.model is None:
        return 80.0  # fallback
    try:
        features = encode_factors_v2_extended(bean).reshape(1, -1)
        pred = state.model.predict(features)[0]
    except Exception:
        features = encode_factors_v2(bean).reshape(1, -1)
        pred = state.model.predict(features)[0]
    return float(np.clip(pred, 60, 100))


def predict_interval(bean: dict) -> dict | None:
    """Predict 80% confidence interval using quantile models."""
    if not state.quantile_models:
        return None
    try:
        features = encode_factors_v2_extended(bean).reshape(1, -1)
    except Exception:
        features = encode_factors_v2(bean).reshape(1, -1)
    try:
        low = float(np.clip(state.quantile_models["low"].predict(features)[0], 60, 100))
        high = float(np.clip(state.quantile_models["high"].predict(features)[0], 60, 100))
        return {"score_low": round(low, 1), "score_high": round(high, 1)}
    except Exception:
        return None


def get_attribution(bean: dict) -> dict:
    """Compute per-factor attribution using model."""
    if state.model is None:
        return {"G": 0.5, "P": 0.3, "R": 0.1, "B": 0.1}

    try:
        features = encode_factors_v2_extended(bean)
        names = get_feature_names_v2_extended()
    except Exception:
        features = encode_factors_v2(bean)
        names = state.feature_names

    # Get feature importance from model
    model_step = state.model.named_steps.get("model")
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_)
    else:
        importances = np.ones(len(features))

    # Weight by feature activation
    activated = importances * np.abs(features)

    # Map to factors
    g_prefixes = ["altitude_m", "shade_pct", "latitude", "delta_t_c", "variety_", "soil_type_"]
    p_prefixes = ["fermentation_hours", "drying_days", "method_p_", "drying_method_", "anaerobic"]

    g_sum = p_sum = 0.0
    for i, name in enumerate(names):
        val = activated[i] if i < len(activated) else 0
        if any(name.startswith(p) for p in g_prefixes):
            g_sum += val
        elif any(name.startswith(p) for p in p_prefixes):
            p_sum += val

    total = g_sum + p_sum or 1.0
    return {
        "G": round(g_sum / total, 3),
        "P": round(p_sum / total, 3),
        "R": 0.0,  # no R variance in CQI data
        "B": 0.0,  # no B variance in CQI data
    }


def score_grade(score: float) -> str:
    if score >= 90:
        return "Outstanding"
    if score >= 85:
        return "Excellent"
    if score >= 80:
        return "Very Good"
    if score >= 75:
        return "Good"
    if score >= 70:
        return "Fair"
    return "Below Specialty"


def match_user_prefs(bean: dict, prefs: UserPrefs) -> float:
    """Score how well a bean matches user preferences (0-1)."""
    scores = bean.get("scores", {})

    # Map prefs to bean sub-scores
    acidity_match = 1.0 - abs(scores.get("acidity", 7.5) - (5 + prefs.acidity * 0.5)) / 5
    sweetness_match = 1.0 - abs(scores.get("sweetness", 7.5) - (5 + prefs.sweetness * 0.5)) / 5
    body_match = 1.0 - abs(scores.get("body", 7.5) - (5 + prefs.body * 0.5)) / 5

    # Fermentation preference: higher → prefer natural/anaerobic
    ferm_score = 0.5
    if prefs.fermentation > 7:
        if bean["P"]["method"] in ("natural", "honey_black") or bean["P"]["anaerobic"]:
            ferm_score = 1.0
    elif prefs.fermentation < 3:
        if bean["P"]["method"] == "washed" and not bean["P"]["anaerobic"]:
            ferm_score = 1.0

    # Complexity: higher → prefer more varieties, higher altitude
    complexity_score = min(1.0, bean["G"]["altitude_m"] / 2000) if prefs.complexity > 5 else 0.5

    total = (
        acidity_match * 0.25 +
        sweetness_match * 0.25 +
        body_match * 0.2 +
        ferm_score * 0.15 +
        complexity_score * 0.15
    )
    return max(0, min(1, total))


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/version")
def version():
    return {
        "api_version": app.version,
        "model_loaded": state.model is not None,
        "feature_dim": FEATURE_DIM_V2,
        "python_version": sys.version,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "beans_loaded": len(state.beans_all),
        "feature_dim": FEATURE_DIM_V2,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict coffee quality score from G/P/R/B factors."""
    bean = bean_from_request(req)
    score = predict_single(bean)
    attribution = get_attribution(bean)

    result = {
        "score": round(score, 1),
        "grade": score_grade(score),
        "attribution": attribution,
        "feature_dim": FEATURE_DIM_V2,
    }
    interval = predict_interval(bean)
    if interval:
        result.update(interval)
    return result


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Compare two beans side by side."""
    bean_a = bean_from_request(req.bean_a)
    bean_b = bean_from_request(req.bean_b)

    score_a = predict_single(bean_a)
    score_b = predict_single(bean_b)
    attr_a = get_attribution(bean_a)
    attr_b = get_attribution(bean_b)

    return {
        "bean_a": {
            "score": round(score_a, 1),
            "grade": score_grade(score_a),
            "attribution": attr_a,
        },
        "bean_b": {
            "score": round(score_b, 1),
            "grade": score_grade(score_b),
            "attribution": attr_b,
        },
        "delta": {
            "score": round(score_a - score_b, 1),
            "attribution": {k: round(attr_a[k] - attr_b.get(k, 0), 3) for k in attr_a},
        },
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """Recommend beans matching user taste profile."""
    if not state.beans_all:
        return {"beans": [], "message": "No beans loaded"}

    scored = []
    for i, bean in enumerate(state.beans_all):
        # Use pre-computed prediction if available, else compute on the fly
        pred_score = state.bean_predictions[i] if i < len(state.bean_predictions) else predict_single(bean)
        pref_match = match_user_prefs(bean, req.prefs)
        # Normalize both to 0-1 before blending (pred_score range: 60-100)
        quality_norm = (pred_score - 60) / 40
        combined = quality_norm * 0.6 + pref_match * 0.4
        scored.append({
            "name": bean["name"],
            "country": bean["G"]["country"],
            "variety": bean["G"]["variety"],
            "process": bean["P"]["method"],
            "altitude": bean["G"]["altitude_m"],
            "predicted_score": round(pred_score, 1),
            "pref_match": round(pref_match, 3),
            "combined_score": round(combined, 1),
            "actual_score": bean["scores"]["overall"],
        })

    scored.sort(key=lambda x: -x["combined_score"])
    return {"beans": scored[:req.top_k]}


@app.post("/api/explore")
def explore(req: ExploreRequest):
    """Factor recombination engine: fix G, vary P, predict outcomes."""
    results = []

    for method in req.vary_methods:
        for anaerobic in req.vary_anaerobic:
            for ferm_h in req.vary_fermentation:
                bean = {
                    "G": req.G.model_dump(),
                    "P": {
                        "method": method,
                        "anaerobic": anaerobic,
                        "fermentation_hours": ferm_h,
                        "drying_method": "raised_bed" if req.G.country in ("Ethiopia", "Kenya", "Rwanda") else "patio",
                        "drying_days": {"natural": 18, "washed": 10, "honey_yellow": 12, "honey_red": 15, "honey_black": 18, "wet_hulled": 8}.get(method, 12),
                    },
                    "R": {"roast_level": "medium_light", "first_crack_temp_c": 198, "drop_temp_c": 205, "dtr_pct": 22, "total_time_s": 600},
                    "B": {"method": "v60", "grind_microns": 600, "water_temp_c": 93, "ratio": 15, "brew_time_s": 240, "water_tds_ppm": 120},
                }

                score = predict_single(bean)
                attribution = get_attribution(bean)

                results.append({
                    "method": method,
                    "anaerobic": anaerobic,
                    "fermentation_hours": ferm_h,
                    "predicted_score": round(score, 1),
                    "grade": score_grade(score),
                    "attribution": attribution,
                })

    results.sort(key=lambda x: -x["predicted_score"])
    return {
        "base_variety": req.G.variety,
        "base_country": req.G.country,
        "base_altitude": req.G.altitude_m,
        "combinations": len(results),
        "results": results,
        "best": results[0] if results else None,
    }


@app.get("/api/model-info")
def model_info():
    """Return current model metadata."""
    model_json = Path(__file__).parent.parent / "site" / "data" / "model.json"
    cached = state.get_cached_json(model_json)
    if cached is not None:
        return cached
    return {"error": "model.json not found"}


@app.get("/api/beans/summary")
def beans_summary():
    """Return dataset summary."""
    summary_path = Path(__file__).parent.parent / "site" / "data" / "beans_summary.json"
    cached = state.get_cached_json(summary_path)
    if cached is not None:
        return cached
    return {"error": "beans_summary.json not found"}


@app.get("/api/experiments")
def experiments():
    """Return experiment history."""
    results_path = Path(__file__).parent.parent / "site" / "data" / "results.json"
    cached = state.get_cached_json(results_path)
    if cached is not None:
        return cached
    return []


# Mount static site
SITE_DIR = Path(__file__).parent.parent / "site"
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
