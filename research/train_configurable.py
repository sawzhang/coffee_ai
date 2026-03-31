"""
Config-driven training script for the experiment runner.
Reads experiment_config.json and builds the appropriate model.
"""

import time
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from prepare import (
    load_data, encode_factors, evaluate_model,
    TIME_BUDGET, FEATURE_DIM, get_feature_names,
)

CONFIG_PATH = Path(__file__).parent / "experiment_config.json"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def build_model(cfg):
    """Build sklearn pipeline from config dict."""
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    model_type = cfg.get("MODEL_TYPE", "gbr")

    if model_type == "gbr":
        estimator = GradientBoostingRegressor(
            n_estimators=cfg.get("GBR_N", 200),
            max_depth=cfg.get("GBR_DEPTH", 4),
            learning_rate=cfg.get("GBR_LR", 0.05),
            subsample=cfg.get("GBR_SUB", 0.8),
            min_samples_leaf=cfg.get("GBR_LEAF", 5),
            max_features="sqrt",
        )
    elif model_type == "rf":
        estimator = RandomForestRegressor(
            n_estimators=cfg.get("RF_N", 500),
            max_depth=cfg.get("RF_DEPTH", 8),
            min_samples_leaf=cfg.get("RF_LEAF", 5),
            max_features="sqrt",
            n_jobs=-1,
        )
    elif model_type == "ridge":
        estimator = Ridge(alpha=cfg.get("LINEAR_ALPHA", 1.0))
    elif model_type == "lasso":
        estimator = Lasso(alpha=cfg.get("LINEAR_ALPHA", 0.01))
    elif model_type == "elastic":
        estimator = ElasticNet(
            alpha=cfg.get("LINEAR_ALPHA", 0.1),
            l1_ratio=cfg.get("ELASTIC_L1", 0.5),
        )
    elif model_type == "svr":
        estimator = SVR(
            C=cfg.get("SVR_C", 10.0),
            epsilon=cfg.get("SVR_EPS", 0.1),
            kernel="rbf",
            gamma="scale",
        )
    elif model_type == "linear_sgd":
        estimator = SGDRegressor(
            loss="huber", penalty="elasticnet",
            alpha=0.001, l1_ratio=0.15, max_iter=2000,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps = [("scaler", StandardScaler())]

    poly_degree = cfg.get("POLY_DEGREE", 0)
    if poly_degree >= 2:
        from sklearn.preprocessing import PolynomialFeatures
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, interaction_only=True)))

    steps.append(("model", estimator))
    return Pipeline(steps), model_type


def get_importance(pipeline, n_features):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_).flatten()
        if len(coef) == n_features:
            return np.abs(coef)
    return np.ones(n_features) / n_features


def get_weights_bias(pipeline, n_features):
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coef = np.array(model.coef_).flatten()
        intercept = float(np.array(model.intercept_).flatten()[0]) if hasattr(model.intercept_, '__len__') else float(model.intercept_)
        if len(coef) == n_features:
            return coef.tolist(), intercept
    imp = get_importance(pipeline, n_features)
    if len(imp) == n_features:
        return imp.tolist(), 82.0
    return [0.0] * n_features, 82.0


def main():
    cfg = load_config()
    train_data, val_data = load_data()
    feature_names = get_feature_names()

    train_X = np.array([encode_factors(b) for b in train_data])
    train_y = np.array([b["scores"]["overall"] for b in train_data])

    pipeline, model_type = build_model(cfg)

    print(f"Training: {len(train_data)} samples, Val: {len(val_data)}, Model: {model_type}")

    start = time.time()
    pipeline.fit(train_X, train_y)
    elapsed = time.time() - start

    # Evaluate
    def predict_bean(bean):
        feat = encode_factors(bean).reshape(1, -1)
        return float(np.clip(pipeline.predict(feat)[0], 60, 100))

    val_mae = evaluate_model(predict_bean, val_data)
    train_preds = pipeline.predict(train_X)
    train_mae = float(np.mean(np.abs(train_preds - train_y)))

    # Count params
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        num_params = len(np.array(model.coef_).flatten()) + 1
    elif hasattr(model, "n_estimators"):
        num_params = getattr(model, "n_estimators", 100) * 50
    else:
        num_params = FEATURE_DIM + 1

    print(f"\n--- RESULTS ---")
    print(f"val_mae: {val_mae:.6f}")
    print(f"train_mae: {train_mae:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_params: {num_params}")
    print(f"model_type: {model_type}")

    # Feature importance
    importance = get_importance(pipeline, FEATURE_DIM)
    if len(importance) == len(feature_names):
        ranked = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        print("\nTop features:")
        for name, w in ranked[:15]:
            print(f"  {name:30s} {w:.4f}")
    else:
        ranked = []

    # Export model.json
    weights, bias = get_weights_bias(pipeline, FEATURE_DIM)
    model_data = {
        "val_mae": round(val_mae, 4),
        "training_seconds": round(elapsed, 1),
        "num_params": num_params,
        "weights": weights,
        "bias": round(bias, 4),
        "feature_names": feature_names,
        "interaction_terms": [],
        "factor_weights": {"G": 0.35, "P": 0.25, "R": 0.25, "B": 0.15},
        "model_type": model_type,
        "config": cfg,
        "top_features": [
            {"name": n, "importance": round(float(w), 4)}
            for n, w in (ranked[:15] if ranked else [])
        ],
    }

    out_dir = Path(__file__).parent.parent / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"\nModel exported to {out_dir / 'model.json'}")


if __name__ == "__main__":
    main()
