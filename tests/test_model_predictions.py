import copy
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path(__file__).parent.parent / "research" / "model.joblib"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("model.joblib not found; run train_v2.py first")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture
def base_bean():
    return {
        "name": "Test Bean",
        "G": {
            "variety": "Typica",
            "altitude_m": 1600,
            "country": "Colombia",
            "region": "Huila",
            "soil_type": "volcanic",
            "shade_pct": 30,
            "latitude": 4,
            "delta_t_c": 11,
        },
        "P": {
            "method": "washed",
            "anaerobic": False,
            "fermentation_hours": 24,
            "drying_method": "raised_bed",
            "drying_days": 12,
        },
        "R": {
            "roast_level": "medium_light",
            "first_crack_temp_c": 198,
            "drop_temp_c": 205,
            "dtr_pct": 22,
            "total_time_s": 600,
        },
        "B": {
            "method": "v60",
            "grind_microns": 600,
            "water_temp_c": 93,
            "ratio": 15,
            "brew_time_s": 240,
            "water_tds_ppm": 120,
        },
        "scores": {"overall": 83},
    }


from prepare_v2 import (
    encode_factors_v2_extended,
    FEATURE_DIM_V2_EXTENDED,
)


class TestModelStructure:
    def test_model_is_pipeline(self, model):
        from sklearn.pipeline import Pipeline
        assert isinstance(model, Pipeline)

    def test_model_accepts_extended_features(self, model, base_bean):
        features = encode_factors_v2_extended(base_bean).reshape(1, -1)
        assert features.shape[1] == FEATURE_DIM_V2_EXTENDED
        pred = model.predict(features)
        assert pred.shape == (1,)


class TestModelPredictions:
    def test_prediction_in_valid_range(self, model, base_bean):
        features = encode_factors_v2_extended(base_bean).reshape(1, -1)
        pred = model.predict(features)[0]
        assert 60 <= pred <= 100

    def test_high_altitude_gesha_scores_above_82(self, model, base_bean):
        bean = copy.deepcopy(base_bean)
        bean["G"]["variety"] = "Gesha"
        bean["G"]["altitude_m"] = 2200
        bean["G"]["country"] = "Ethiopia"
        bean["G"]["latitude"] = 6.5
        features = encode_factors_v2_extended(bean).reshape(1, -1)
        pred = model.predict(features)[0]
        assert pred > 82

    def test_altitude_monotonicity(self, model, base_bean):
        high = copy.deepcopy(base_bean)
        high["G"]["altitude_m"] = 2300
        low = copy.deepcopy(base_bean)
        low["G"]["altitude_m"] = 900

        pred_high = model.predict(encode_factors_v2_extended(high).reshape(1, -1))[0]
        pred_low = model.predict(encode_factors_v2_extended(low).reshape(1, -1))[0]
        assert pred_high > pred_low

    def test_batch_prediction_deterministic(self, model, base_bean):
        features = encode_factors_v2_extended(base_bean).reshape(1, -1)
        batch = np.vstack([features, features])
        preds = model.predict(batch)
        assert preds[0] == preds[1]
