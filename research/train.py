"""
Coffee Attribution AutoResearch — Training Script
=================================================
Thin wrapper: by default delegates to train_v2 (V2 pipeline, G+P only).
Use --legacy flag to run the original 52-feature pipeline.

Run:
  python3 train.py           # V2 pipeline (33 features, recommended)
  python3 train.py --legacy  # Original 52-feature pipeline
"""

import sys


def _legacy_main():
    """Original 52-feature training pipeline (kept for reference)."""
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

    # ── Configuration ──────────────────────────────────────────────
    MODEL_TYPE = "gbr"
    GBR_PARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    }
    USE_SCALER = True
    FACTOR_WEIGHTS = {"G": 0.35, "P": 0.25, "R": 0.25, "B": 0.15}

    train_data, val_data = load_data()
    feature_names = get_feature_names()

    print(f"[LEGACY] Training: {len(train_data)} samples, Validation: {len(val_data)} samples")
    print(f"Feature dim: {FEATURE_DIM}, Model: {MODEL_TYPE}")

    train_X = np.array([encode_factors(b) for b in train_data])
    train_y = np.array([b["scores"]["overall"] for b in train_data])
    val_X = np.array([encode_factors(b) for b in val_data])
    val_y = np.array([b["scores"]["overall"] for b in val_data])

    start = time.time()
    steps = []
    if USE_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", GradientBoostingRegressor(**GBR_PARAMS)))
    pipeline = Pipeline(steps)
    pipeline.fit(train_X, train_y)
    elapsed = time.time() - start

    class _Wrapper:
        def predict_from_bean(self, bean):
            features = encode_factors(bean).reshape(1, -1)
            pred = pipeline.predict(features)[0]
            return float(np.clip(pred, 60, 100))

    wrapper = _Wrapper()
    val_mae = evaluate_model(wrapper.predict_from_bean, val_data)
    train_preds = pipeline.predict(train_X)
    train_mae = float(np.mean(np.abs(train_preds - train_y)))

    print(f"\n--- RESULTS (LEGACY) ---")
    print(f"val_mae: {val_mae:.6f}")
    print(f"train_mae: {train_mae:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"model_type: {MODEL_TYPE}")

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("\nTop features:")
        for name, w in ranked[:15]:
            print(f"  {name:30s} {w:.4f}")

    # Export
    out_dir = Path(__file__).parent.parent / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else [0.0] * FEATURE_DIM
    model_data = {
        "val_mae": round(val_mae, 4),
        "training_seconds": round(elapsed, 1),
        "num_params": sum(t.tree_.node_count for t in model.estimators_.flatten()),
        "weights": weights,
        "bias": round(float(train_y.mean()), 4),
        "feature_names": feature_names,
        "interaction_terms": [],
        "factor_weights": FACTOR_WEIGHTS,
        "model_type": MODEL_TYPE,
        "config": {"model_type": MODEL_TYPE, "use_scaler": USE_SCALER, "legacy": True},
        "top_features": [
            {"name": name, "importance": round(float(w), 4)}
            for name, w in (ranked[:15] if ranked else [])
        ],
    }
    with open(out_dir / "model.json", "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"\nModel exported to {out_dir / 'model.json'}")


def main():
    if "--legacy" in sys.argv:
        print("Running LEGACY 52-feature pipeline...")
        _legacy_main()
    else:
        from train_v2 import main as train_v2_main
        train_v2_main()


if __name__ == "__main__":
    main()
