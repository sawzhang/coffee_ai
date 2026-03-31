"""
Coffee Attribution AutoResearch — API Server
=============================================
FastAPI backend providing model prediction, factor attribution,
bean recommendation, and factor recombination endpoints.

Run: uvicorn api.server:app --reload --port 8000
  or: python3 -m api.server
"""

import sys
import json
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add research/ to path
RESEARCH_DIR = Path(__file__).parent.parent / "research"
sys.path.insert(0, str(RESEARCH_DIR))

from prepare_v2 import (
    encode_factors_v2, get_feature_names_v2, load_data,
    FEATURE_DIM_V2, NUM_RANGES_V2, CAT_FIELDS_V2,
    VARIETIES, PROCESS_METHODS, SOIL_TYPES, DRYING_METHODS,
)

# ── Global state ──────────────────────────────────────────────────────
MODEL = None
TRAIN_DATA = None
VAL_DATA = None
BEANS_ALL = None
FEATURE_NAMES = None


def load_model():
    """Load trained sklearn pipeline from pickle."""
    global MODEL
    import pickle
    model_path = RESEARCH_DIR / "model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: {model_path} not found. Run train_v2.py first.")


def load_beans():
    """Load full bean dataset for recommendation."""
    global TRAIN_DATA, VAL_DATA, BEANS_ALL, FEATURE_NAMES
    beans_path = str(RESEARCH_DIR / "data" / "beans.json")
    TRAIN_DATA, VAL_DATA = load_data(beans_path)
    BEANS_ALL = TRAIN_DATA + VAL_DATA
    FEATURE_NAMES = get_feature_names_v2()
    print(f"Loaded {len(BEANS_ALL)} beans")


@asynccontextmanager
async def lifespan(app):
    load_model()
    load_beans()
    yield


app = FastAPI(title="Coffee Attribution API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────

class GFactors(BaseModel):
    variety: str = "Bourbon"
    altitude_m: float = 1600
    country: str = "Colombia"
    region: str = "Huila"
    soil_type: str = "volcanic"
    shade_pct: float = 30
    latitude: float = 4
    delta_t_c: float = 11

class PFactors(BaseModel):
    method: str = "washed"
    anaerobic: bool = False
    fermentation_hours: float = 24
    drying_method: str = "raised_bed"
    drying_days: float = 12

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


# ── Helpers ───────────────────────────────────────────────────────────

def bean_from_request(req: PredictRequest) -> dict:
    """Convert API request to bean dict for model."""
    return {
        "G": req.G.dict(),
        "P": req.P.dict(),
        "R": req.R.dict(),
        "B": req.B.dict(),
    }


def predict_single(bean: dict) -> float:
    """Predict score for a single bean using V2 features."""
    if MODEL is None:
        return 80.0  # fallback
    features = encode_factors_v2(bean).reshape(1, -1)
    pred = MODEL.predict(features)[0]
    return float(np.clip(pred, 60, 100))


def get_attribution(bean: dict) -> dict:
    """Compute per-factor attribution using model."""
    if MODEL is None:
        return {"G": 0.5, "P": 0.3, "R": 0.1, "B": 0.1}

    features = encode_factors_v2(bean)
    names = FEATURE_NAMES

    # Get feature importance from model
    model_step = MODEL.named_steps.get("model")
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
    if score >= 90: return "Outstanding"
    if score >= 85: return "Excellent"
    if score >= 80: return "Very Good"
    if score >= 75: return "Good"
    if score >= 70: return "Fair"
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

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "beans_loaded": len(BEANS_ALL) if BEANS_ALL else 0,
        "feature_dim": FEATURE_DIM_V2,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict coffee quality score from G/P/R/B factors."""
    bean = bean_from_request(req)
    score = predict_single(bean)
    attribution = get_attribution(bean)

    return {
        "score": round(score, 1),
        "grade": score_grade(score),
        "attribution": attribution,
        "feature_dim": FEATURE_DIM_V2,
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """Recommend beans matching user taste profile."""
    if not BEANS_ALL:
        return {"beans": [], "message": "No beans loaded"}

    scored = []
    for bean in BEANS_ALL:
        pred_score = predict_single(bean)
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
                    "G": req.G.dict(),
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
    if model_json.exists():
        with open(model_json) as f:
            return json.load(f)
    return {"error": "model.json not found"}


@app.get("/api/beans/summary")
def beans_summary():
    """Return dataset summary."""
    summary_path = Path(__file__).parent.parent / "site" / "data" / "beans_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {"error": "beans_summary.json not found"}


@app.get("/api/experiments")
def experiments():
    """Return experiment history."""
    results_path = Path(__file__).parent.parent / "site" / "data" / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


# Mount static site
SITE_DIR = Path(__file__).parent.parent / "site"
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
