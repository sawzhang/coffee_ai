import sys
from pathlib import Path

import pytest

# Add research/ to path so prepare_v2 can be imported
RESEARCH_DIR = Path(__file__).parent.parent / "research"
sys.path.insert(0, str(RESEARCH_DIR))


@pytest.fixture
def sample_bean():
    """A known-good bean dict for testing."""
    return {
        "name": "Test Ethiopian Typica",
        "G": {
            "variety": "Typica",
            "altitude_m": 1900,
            "country": "Ethiopia",
            "region": "Yirgacheffe",
            "soil_type": "volcanic",
            "shade_pct": 40,
            "latitude": 6.5,
            "delta_t_c": 13,
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
        "scores": {
            "overall": 87.5,
            "aroma": 8.2,
            "acidity": 8.5,
            "sweetness": 8.0,
            "body": 7.8,
            "aftertaste": 8.1,
            "balance": 8.0,
        },
    }
