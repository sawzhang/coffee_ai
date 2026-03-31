"""
Coffee Attribution V2 Training — Optimized Feature Set
======================================================
Uses only G + P factors (33 features instead of 52).
Saves model as pickle for API server consumption.

Run: python3 train_v2.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_v2 import (
    FEATURE_DIM_V2,
    encode_factors_v2,
    get_feature_names_v2,
    load_data,
)

# ── Configuration ─────────────────────────────────────────────────────
MODEL_TYPE = "gbr"

GBR_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}

USE_SCALER = True


def main():
    train_data, val_data = load_data()
    feature_names = get_feature_names_v2()

    print(f"V2 Training: {len(train_data)} train, {len(val_data)} val")
    print(f"Feature dim: {FEATURE_DIM_V2} (was 52, dropped R/B constants)")

    train_X = np.array([encode_factors_v2(b) for b in train_data])
    train_y = np.array([b["scores"]["overall"] for b in train_data])
    val_X = np.array([encode_factors_v2(b) for b in val_data])
    val_y = np.array([b["scores"]["overall"] for b in val_data])

    # Build pipeline
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", GradientBoostingRegressor(**GBR_PARAMS)))
    pipeline = Pipeline(steps)

    # Train
    start = time.time()
    pipeline.fit(train_X, train_y)
    elapsed = time.time() - start

    # Evaluate
    val_preds = pipeline.predict(val_X)
    val_mae = float(np.mean(np.abs(val_preds - val_y)))
    train_preds = pipeline.predict(train_X)
    train_mae = float(np.mean(np.abs(train_preds - train_y)))

    # Cross-validation
    all_X = np.vstack([train_X, val_X])
    all_y = np.concatenate([train_y, val_y])
    cv_scores = cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(**GBR_PARAMS))]),
        all_X, all_y, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()

    print("\n--- RESULTS ---")
    print(f"val_mae: {val_mae:.6f}")
    print(f"train_mae: {train_mae:.6f}")
    print(f"cv_mae (5-fold): {cv_mae:.6f} ± {cv_scores.std():.4f}")
    print(f"training_seconds: {elapsed:.1f}")

    # Feature importance
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print("\nTop features (V2):")
    for name, w in ranked[:15]:
        print(f"  {name:30s} {w:.4f}")

    # Save model for API
    model_path = Path(__file__).parent / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    # Export model.json for site
    model_data = {
        "val_mae": round(val_mae, 4),
        "cv_mae": round(cv_mae, 4),
        "train_mae": round(train_mae, 4),
        "training_seconds": round(elapsed, 1),
        "num_params": sum(t.tree_.node_count for t in model.estimators_.flatten()),
        "weights": importances.tolist(),
        "bias": round(float(all_y.mean()), 4),
        "feature_names": feature_names,
        "interaction_terms": [],
        "factor_weights": {"G": 0.35, "P": 0.25, "R": 0.0, "B": 0.0},
        "model_type": MODEL_TYPE,
        "version": "v2",
        "config": {
            "model_type": MODEL_TYPE,
            "feature_dim": FEATURE_DIM_V2,
            "gbr_params": GBR_PARAMS,
            "dropped_features": "R (roast) and B (brew) — constant in CQI data",
        },
        "top_features": [
            {"name": n, "importance": round(float(w), 4)}
            for n, w in ranked[:15]
        ],
    }

    out_dir = Path(__file__).parent.parent / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"Model JSON exported to {out_dir / 'model.json'}")


if __name__ == "__main__":
    main()
