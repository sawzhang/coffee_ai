import json
from pathlib import Path

import pytest

from prepare_v2 import get_feature_names_v2_extended, FEATURE_DIM_V2_EXTENDED

MODEL_JSON_PATH = Path(__file__).parent.parent / "site" / "data" / "model.json"


@pytest.fixture(scope="module")
def model_data():
    if not MODEL_JSON_PATH.exists():
        pytest.skip("site/data/model.json not found")
    with open(MODEL_JSON_PATH) as f:
        return json.load(f)


class TestModelJsonSchema:
    def test_has_required_keys(self, model_data):
        required = {
            "val_mae", "cv_mae", "train_mae", "weights", "bias",
            "feature_names", "factor_weights", "model_type", "version", "config",
        }
        assert required.issubset(model_data.keys())

    def test_version_is_v2(self, model_data):
        assert model_data["version"] == "v2"

    def test_feature_names_match_extended(self, model_data):
        expected = get_feature_names_v2_extended()
        assert model_data["feature_names"] == expected

    def test_weights_length_matches_feature_names(self, model_data):
        assert len(model_data["weights"]) == len(model_data["feature_names"])

    def test_feature_dim_matches(self, model_data):
        assert len(model_data["feature_names"]) == FEATURE_DIM_V2_EXTENDED


class TestModelJsonValues:
    def test_mae_values_positive_and_reasonable(self, model_data):
        for key in ("val_mae", "cv_mae", "train_mae"):
            val = model_data[key]
            assert 0 < val < 5, f"{key}={val} out of range"

    def test_factor_weights_has_expected_keys(self, model_data):
        fw = model_data["factor_weights"]
        assert set(fw.keys()) == {"G", "P", "R", "B"}

    def test_factor_weights_r_and_b_zero(self, model_data):
        fw = model_data["factor_weights"]
        assert fw["R"] == 0.0
        assert fw["B"] == 0.0

    def test_factor_weights_g_and_p_positive(self, model_data):
        fw = model_data["factor_weights"]
        assert fw["G"] > 0
        assert fw["P"] > 0
