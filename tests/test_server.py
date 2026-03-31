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
    def test_fallback_when_no_model(self, sample_bean):
        """Without a loaded model, should return 80.0 fallback."""
        score = predict_single(sample_bean)
        assert score == 80.0

    def test_score_in_range(self, sample_bean):
        score = predict_single(sample_bean)
        assert 60 <= score <= 100


class TestGetAttribution:
    def test_fallback_when_no_model(self, sample_bean):
        """Without a loaded model, should return default attribution."""
        attr = get_attribution(sample_bean)
        assert "G" in attr and "P" in attr
        assert all(v >= 0 for v in attr.values())
