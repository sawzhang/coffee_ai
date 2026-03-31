"""
Coffee Attribution AutoResearch — Training Script
=================================================
This file IS modifiable by the research agent.
Goal: achieve the lowest val_mae on the validation set.

Run: python3 train.py
"""

import time
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import (
    load_data, encode_factors, evaluate_model,
    TIME_BUDGET, FEATURE_DIM, get_feature_names,
)

# ── Configuration (AGENT: modify these) ──────────────────────────────

# Model type: "gbr", "ridge", "lasso", "rf", "svr", "elastic", "linear_sgd"
MODEL_TYPE = "gbr"

# GradientBoostingRegressor params
GBR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}

# Ridge/Lasso/ElasticNet params
LINEAR_ALPHA = 1.0
ELASTIC_L1_RATIO = 0.5

# RandomForest params
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}

# SVR params
SVR_PARAMS = {
    "C": 10.0,
    "epsilon": 0.1,
    "kernel": "rbf",
    "gamma": "scale",
}

# Feature engineering
USE_SCALER = True
POLY_DEGREE = 0          # 0 = no polynomial features, 2 = degree 2

# Factor weights (for attribution display, not used in training)
FACTOR_WEIGHTS = {
    "G": 0.35,
    "P": 0.25,
    "R": 0.25,
    "B": 0.15,
}


# ── Model Factory ────────────────────────────────────────────────────
def build_model():
    """Build sklearn pipeline based on MODEL_TYPE."""
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    estimators = {
        "gbr": lambda: GradientBoostingRegressor(**GBR_PARAMS),
        "ridge": lambda: Ridge(alpha=LINEAR_ALPHA),
        "lasso": lambda: Lasso(alpha=LINEAR_ALPHA),
        "elastic": lambda: ElasticNet(alpha=LINEAR_ALPHA, l1_ratio=ELASTIC_L1_RATIO),
        "rf": lambda: RandomForestRegressor(**RF_PARAMS, n_jobs=-1),
        "svr": lambda: SVR(**SVR_PARAMS),
        "linear_sgd": lambda: SGDRegressor(
            loss="huber", penalty="elasticnet",
            alpha=0.001, l1_ratio=0.15, max_iter=2000
        ),
    }

    if MODEL_TYPE not in estimators:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))

    if POLY_DEGREE >= 2:
        from sklearn.preprocessing import PolynomialFeatures
        steps.append(("poly", PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True)))

    steps.append(("model", estimators[MODEL_TYPE]()))
    return Pipeline(steps)


# ── Prediction wrapper ───────────────────────────────────────────────
class ModelWrapper:
    """Wraps sklearn pipeline for evaluate_model compatibility."""

    def __init__(self, pipeline, feature_names):
        self.pipeline = pipeline
        self.feature_names = feature_names

    def predict_from_bean(self, bean):
        features = encode_factors(bean).reshape(1, -1)
        pred = self.pipeline.predict(features)[0]
        return float(np.clip(pred, 60, 100))

    def get_feature_importance(self):
        """Extract feature importance from the final estimator."""
        model = self.pipeline.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if isinstance(coef, np.ndarray):
                return np.abs(coef)
        return np.ones(FEATURE_DIM) / FEATURE_DIM

    def get_weights_and_bias(self):
        """Get linear weights if available, else return importance as proxy."""
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            return model.coef_.tolist(), float(model.intercept_)
        # For tree models, use feature importance as proxy weights
        imp = self.get_feature_importance()
        return imp.tolist(), 82.0  # mean score as proxy bias


# ── Training ─────────────────────────────────────────────────────────
def main():
    train_data, val_data = load_data()
    feature_names = get_feature_names()

    print(f"Training: {len(train_data)} samples, Validation: {len(val_data)} samples")
    print(f"Feature dim: {FEATURE_DIM}, Model: {MODEL_TYPE}")

    # Pre-encode
    train_X = np.array([encode_factors(b) for b in train_data])
    train_y = np.array([b["scores"]["overall"] for b in train_data])
    val_X = np.array([encode_factors(b) for b in val_data])
    val_y = np.array([b["scores"]["overall"] for b in val_data])

    start = time.time()

    # Build and train
    pipeline = build_model()
    pipeline.fit(train_X, train_y)

    elapsed = time.time() - start

    # Evaluate
    wrapper = ModelWrapper(pipeline, feature_names)
    val_mae = evaluate_model(wrapper.predict_from_bean, val_data)

    # Quick train MAE for reference
    train_preds = pipeline.predict(train_X)
    train_mae = float(np.mean(np.abs(train_preds - train_y)))

    # ── Output ────────────────────────────────────────────────────
    print(f"\n--- RESULTS ---")
    print(f"val_mae: {val_mae:.6f}")
    print(f"train_mae: {train_mae:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_params: {_count_params(pipeline)}")
    print(f"model_type: {MODEL_TYPE}")

    # Feature importance
    importance = wrapper.get_feature_importance()
    if len(importance) == len(feature_names):
        ranked = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        print("\nTop features:")
        for name, w in ranked[:15]:
            print(f"  {name:30s} {w:.4f}")
    else:
        ranked = []
        print(f"\n(Feature importance size mismatch: {len(importance)} vs {len(feature_names)})")

    # Export
    _export_model(wrapper, feature_names, val_mae, elapsed, ranked)


def _count_params(pipeline):
    """Estimate parameter count."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        return len(model.coef_) + 1
    elif hasattr(model, "n_estimators") and hasattr(model, "estimators_"):
        return sum(t.tree_.node_count for t in (
            model.estimators_ if hasattr(model.estimators_[0], 'tree_') else
            [e[0] for e in model.estimators_]
        ))
    elif hasattr(model, "n_estimators"):
        return model.n_estimators * 50  # rough estimate
    return FEATURE_DIM + 1


def _export_model(wrapper, feature_names, val_mae, elapsed, ranked):
    """Export model to site/data/model.json."""
    weights, bias = wrapper.get_weights_and_bias()

    # Ensure weights length matches feature_names
    if len(weights) != len(feature_names):
        # Pad or truncate
        if len(weights) < len(feature_names):
            weights = weights + [0.0] * (len(feature_names) - len(weights))
        else:
            weights = weights[:len(feature_names)]

    model_data = {
        "val_mae": round(val_mae, 4),
        "training_seconds": round(elapsed, 1),
        "num_params": _count_params(wrapper.pipeline),
        "weights": weights,
        "bias": round(bias, 4),
        "feature_names": feature_names,
        "interaction_terms": [],
        "factor_weights": FACTOR_WEIGHTS,
        "model_type": MODEL_TYPE,
        "config": {
            "model_type": MODEL_TYPE,
            "use_scaler": USE_SCALER,
            "poly_degree": POLY_DEGREE,
        },
        "top_features": [
            {"name": name, "importance": round(float(w), 4)}
            for name, w in (ranked[:15] if ranked else [])
        ],
    }

    out_dir = Path(__file__).parent.parent / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"\nModel exported to {out_dir / 'model.json'}")


if __name__ == "__main__":
    main()
