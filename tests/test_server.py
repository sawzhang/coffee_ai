import sys
from pathlib import Path

import pytest

# Add research/ to path before importing server
sys.path.insert(0, str(Path(__file__).parent.parent / "research"))

from api.server import (
    get_attribution,
    match_user_prefs,
    predict_single,
    score_grade,
    UserPrefs,
)


class TestScoreGrade:
    def test_outstanding(self):
        assert score_grade(92) == "Outstanding"

    def test_excellent(self):
        assert score_grade(87) == "Excellent"

    def test_very_good(self):
        assert score_grade(82) == "Very Good"

    def test_good(self):
        assert score_grade(77) == "Good"

    def test_fair(self):
        assert score_grade(72) == "Fair"

    def test_below_specialty(self):
        assert score_grade(65) == "Below Specialty"


class TestMatchUserPrefs:
    def test_returns_in_range(self, sample_bean):
        prefs = UserPrefs(acidity=5, sweetness=7, complexity=5, fermentation=5, body=5)
        score = match_user_prefs(sample_bean, prefs)
        assert 0 <= score <= 1

    def test_high_fermentation_prefers_natural(self, sample_bean):
        """High fermentation preference should favor natural/anaerobic."""
        prefs_high = UserPrefs(acidity=5, sweetness=5, complexity=5, fermentation=9, body=5)
        prefs_low = UserPrefs(acidity=5, sweetness=5, complexity=5, fermentation=1, body=5)

        # Washed bean should match low-fermentation preference better
        score_high = match_user_prefs(sample_bean, prefs_high)
        score_low = match_user_prefs(sample_bean, prefs_low)
        assert score_low > score_high


class TestPredictSingle:
    def test_score_in_range(self, sample_bean):
        score = predict_single(sample_bean)
        assert 60 <= score <= 100

    def test_fallback_when_no_model(self, sample_bean):
        """Without a loaded model, should return 80.0 fallback."""
        from api.server import state
        original = state.model
        state.model = None
        try:
            score = predict_single(sample_bean)
            assert score == 80.0
        finally:
            state.model = original


class TestGetAttribution:
    def test_attribution_keys(self, sample_bean):
        attr = get_attribution(sample_bean)
        assert "G" in attr and "P" in attr
        assert all(v >= 0 for v in attr.values())

    def test_fallback_when_no_model(self, sample_bean):
        """Without a loaded model, should return default attribution."""
        from api.server import state
        original = state.model
        state.model = None
        try:
            attr = get_attribution(sample_bean)
            assert attr == {"G": 0.5, "P": 0.3, "R": 0.1, "B": 0.1}
        finally:
            state.model = original


class TestScoreGradeBoundaries:
    def test_exactly_90_is_outstanding(self):
        assert score_grade(90.0) == "Outstanding"

    def test_just_below_90_is_excellent(self):
        assert score_grade(89.99) == "Excellent"

    def test_exactly_85_is_excellent(self):
        assert score_grade(85.0) == "Excellent"

    def test_just_below_85_is_very_good(self):
        assert score_grade(84.99) == "Very Good"

    def test_exactly_80_is_very_good(self):
        assert score_grade(80.0) == "Very Good"

    def test_exactly_75_is_good(self):
        assert score_grade(75.0) == "Good"

    def test_exactly_70_is_fair(self):
        assert score_grade(70.0) == "Fair"

    def test_score_100(self):
        assert score_grade(100) == "Outstanding"

    def test_score_0(self):
        assert score_grade(0) == "Below Specialty"

    def test_score_60(self):
        assert score_grade(60) == "Below Specialty"


class TestMatchUserPrefsEdgeCases:
    def test_all_zeros_prefs_in_range(self, sample_bean):
        prefs = UserPrefs(acidity=0, sweetness=0, complexity=0, fermentation=0, body=0)
        score = match_user_prefs(sample_bean, prefs)
        assert 0 <= score <= 1

    def test_all_max_prefs_in_range(self, sample_bean):
        prefs = UserPrefs(acidity=10, sweetness=10, complexity=10, fermentation=10, body=10)
        score = match_user_prefs(sample_bean, prefs)
        assert 0 <= score <= 1

    def test_bean_without_scores_uses_defaults(self):
        bean = {
            "name": "Bare Bean",
            "G": {
                "variety": "Bourbon",
                "altitude_m": 1500,
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
        }
        prefs = UserPrefs(acidity=5, sweetness=5, complexity=5, fermentation=5, body=5)
        score = match_user_prefs(bean, prefs)
        assert 0 <= score <= 1
