"""
Coffee Attribution V2 Training — Optimized Feature Set
======================================================
Uses only G + P factors (33 features instead of 52).
Supports multiple model types: gbr, hgbr, voting, stacking.
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
    FEATURE_DIM_V2_EXTENDED,
    encode_factors_v2,
    encode_factors_v2_extended,
    get_feature_names_v2,
    get_feature_names_v2_extended,
    load_data,
)

# ── Configuration ─────────────────────────────────────────────────────
MODEL_TYPE = "stacking"  # "gbr", "hgbr", "voting", "stacking"

USE_EXTENDED_FEATURES = True

GBR_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 3,
    "learning_rate": 0.015,
    "subsample": 0.85,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "loss": "huber",
    "alpha": 0.9,
}

HGBR_PARAMS = {
    "max_iter": 2000,
    "max_depth": 3,
    "learning_rate": 0.005,
    "min_samples_leaf": 5,
    "l2_regularization": 0.05,
}

USE_SCALER = True


def _build_model():
    """Build the estimator based on MODEL_TYPE."""
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
        StackingRegressor,
        VotingRegressor,
    )
    from sklearn.linear_model import Ridge

    if MODEL_TYPE == "gbr":
        return GradientBoostingRegressor(**GBR_PARAMS)
    elif MODEL_TYPE == "hgbr":
        return HistGradientBoostingRegressor(**HGBR_PARAMS)
    elif MODEL_TYPE == "voting":
        estimators = [
            ("gbr", GradientBoostingRegressor(**GBR_PARAMS)),
            ("hgbr", HistGradientBoostingRegressor(**HGBR_PARAMS)),
            ("ridge", Ridge(alpha=1.0)),
        ]
        return VotingRegressor(estimators=estimators)
    elif MODEL_TYPE == "stacking":
        estimators = [
            ("gbr", GradientBoostingRegressor(**GBR_PARAMS)),
            ("rf", RandomForestRegressor(
                n_estimators=500, max_depth=6, min_samples_leaf=5,
                max_features="sqrt", n_jobs=-1,
            )),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=3.0),
            cv=5,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def _get_feature_importance(model, n_features):
    """Extract feature importances from various model types."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_)
    # For VotingRegressor / StackingRegressor, try named_estimators_ first
    if hasattr(model, "named_estimators_"):
        for name, est in model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                return est.feature_importances_
    # For ensemble meta-models, try estimators_ list
    if hasattr(model, "estimators_") and isinstance(model.estimators_, list):
        for est in model.estimators_:
            if isinstance(est, tuple):
                est = est[1]
            if hasattr(est, "feature_importances_"):
                return est.feature_importances_
    return np.ones(n_features) / n_features


def _count_params(model):
    """Estimate parameter count for various model types."""
    # VotingRegressor or StackingRegressor — use named_estimators_
    if hasattr(model, "named_estimators_"):
        total = 0
        for name, est in model.named_estimators_.items():
            total += _count_params(est)
        if hasattr(model, "final_estimator_"):
            total += _count_params(model.final_estimator_)
        return total
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "__len__"):
        # GBR
        return sum(t.tree_.node_count for t in model.estimators_.flatten())
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "tree_"):
        # RF
        return sum(t.tree_.node_count for t in model.estimators_)
    if hasattr(model, "coef_"):
        return len(model.coef_) + 1
    # HistGBR
    if hasattr(model, "_predictors"):
        count = 0
        for predictor_list in model._predictors:
            for p in predictor_list:
                count += getattr(p, 'get_n_leaf_nodes', getattr(p, 'get_n_leaf_values', lambda: 0))()
        return count
    return 0


def _target_encode_variety(train_data, val_data, n_folds=5, noise_sigma=0.1, seed=42):
    """K-fold target encoding for variety to avoid leakage."""
    from sklearn.model_selection import KFold
    rng = np.random.RandomState(seed)

    train_varieties = [b["G"]["variety"] for b in train_data]
    train_targets = np.array([b["scores"]["overall"] for b in train_data])
    val_varieties = [b["G"]["variety"] for b in val_data]

    # K-fold encoding for training data
    train_encoded = np.zeros(len(train_data))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(train_data):
        # Compute means from train fold
        fold_means = {}
        for i in train_idx:
            v = train_varieties[i]
            fold_means.setdefault(v, []).append(train_targets[i])
        fold_means = {v: np.mean(scores) for v, scores in fold_means.items()}
        global_mean = train_targets[train_idx].mean()
        # Apply to val fold
        for i in val_idx:
            train_encoded[i] = fold_means.get(train_varieties[i], global_mean)

    # Add noise regularization
    train_encoded += rng.normal(0, noise_sigma, len(train_encoded))

    # For validation: use full training set means
    full_means = {}
    for v, t in zip(train_varieties, train_targets):
        full_means.setdefault(v, []).append(t)
    full_means = {v: np.mean(scores) for v, scores in full_means.items()}
    global_mean = train_targets.mean()
    val_encoded = np.array([full_means.get(v, global_mean) for v in val_varieties])

    return train_encoded, val_encoded


def _optimize_hyperparams(train_X, train_y, val_X, val_y):
    """Use scipy to optimize GBR hyperparameters."""
    from scipy.optimize import differential_evolution

    all_X = np.vstack([train_X, val_X])
    all_y = np.concatenate([train_y, val_y])

    def objective(params):
        lr, subsample, min_leaf, n_est_frac = params
        n_est = int(500 + n_est_frac * 1500)  # 500-2000
        min_leaf_int = max(2, int(min_leaf))

        gbr = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=3, learning_rate=lr,
            subsample=subsample, min_samples_leaf=min_leaf_int,
            max_features="sqrt", loss="huber", alpha=0.9,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("model", gbr)])
        scores = cross_val_score(pipe, all_X, all_y, cv=5, scoring="neg_mean_absolute_error")
        return -scores.mean()

    bounds = [
        (0.005, 0.08),   # learning_rate
        (0.6, 0.95),     # subsample
        (3, 15),          # min_samples_leaf
        (0.0, 1.0),      # n_estimators fraction
    ]

    result = differential_evolution(
        objective, bounds, maxiter=15, seed=42, tol=0.001,
        popsize=8, mutation=(0.5, 1.5), recombination=0.8,
    )
    lr, subsample, min_leaf, n_est_frac = result.x
    print(f"\nOptimized params: lr={lr:.4f}, sub={subsample:.3f}, "
          f"min_leaf={int(min_leaf)}, n_est={int(500 + n_est_frac * 1500)}, "
          f"cv_mae={result.fun:.6f}")
    return {
        "learning_rate": lr,
        "subsample": subsample,
        "min_samples_leaf": max(2, int(min_leaf)),
        "n_estimators": int(500 + n_est_frac * 1500),
    }


def main():
    train_data, val_data = load_data()

    if USE_EXTENDED_FEATURES:
        encode_fn = encode_factors_v2_extended
        feature_names = get_feature_names_v2_extended()
        feature_dim = FEATURE_DIM_V2_EXTENDED
    else:
        encode_fn = encode_factors_v2
        feature_names = get_feature_names_v2()
        feature_dim = FEATURE_DIM_V2

    print(f"V2 Training: {len(train_data)} train, {len(val_data)} val")
    print(f"Feature dim: {feature_dim}, Model: {MODEL_TYPE}, Extended: {USE_EXTENDED_FEATURES}")

    train_X = np.array([encode_fn(b) for b in train_data])
    train_y = np.array([b["scores"]["overall"] for b in train_data])
    val_X = np.array([encode_fn(b) for b in val_data])
    val_y = np.array([b["scores"]["overall"] for b in val_data])

    # Optional hyperparameter optimization
    import sys
    if "--optimize" in sys.argv:
        print("\n=== Running hyperparameter optimization ===")
        opt_params = _optimize_hyperparams(train_X, train_y, val_X, val_y)
        GBR_PARAMS.update(opt_params)
        print(f"Updated GBR_PARAMS: {GBR_PARAMS}")

    # Build pipeline
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _build_model()))
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
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", _build_model()),
    ])
    cv_scores = cross_val_score(
        cv_pipeline, all_X, all_y, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()

    print("\n--- RESULTS ---")
    print(f"val_mae: {val_mae:.6f}")
    print(f"train_mae: {train_mae:.6f}")
    print(f"cv_mae (5-fold): {cv_mae:.6f} +/- {cv_scores.std():.4f}")
    print(f"training_seconds: {elapsed:.1f}")

    # Feature importance
    model = pipeline.named_steps["model"]
    importances = _get_feature_importance(model, feature_dim)
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print("\nTop features (V2):")
    for name, w in ranked[:15]:
        print(f"  {name:30s} {w:.4f}")

    # Save model for API
    model_path = Path(__file__).parent / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    # Export model.json for site
    num_params = _count_params(model)
    model_data = {
        "val_mae": round(val_mae, 4),
        "cv_mae": round(cv_mae, 4),
        "train_mae": round(train_mae, 4),
        "training_seconds": round(elapsed, 1),
        "num_params": num_params,
        "weights": importances.tolist(),
        "bias": round(float(all_y.mean()), 4),
        "feature_names": feature_names,
        "interaction_terms": [],
        "factor_weights": {"G": 0.35, "P": 0.25, "R": 0.0, "B": 0.0},
        "model_type": MODEL_TYPE,
        "version": "v2",
        "config": {
            "model_type": MODEL_TYPE,
            "feature_dim": feature_dim,
            "use_extended_features": USE_EXTENDED_FEATURES,
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
