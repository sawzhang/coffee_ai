"""
Coffee Attribution AutoResearch — V2 Data Preparation
=====================================================
Drops constant R/B features (CQI standard cupping protocol).
Only encodes G + P factors that have real variance in the dataset.

Feature count: 28 (down from 52)
  - Numerical (6): altitude_m, shade_pct, latitude, delta_t_c, fermentation_hours, drying_days
  - Categorical (21): variety(12) + method_p(6) + soil_type(5) + drying_method(3) — wait that's 26
  - Actually: variety(12) + method_p(6) + soil_type(5) = 23 categorical
  - But drying_method has variance too (raised_bed vs patio), keep it: +3 = 26
  - Boolean (1): anaerobic
  - Total: 6 + 26 + 1 = 33

Actually, let me compute it properly below.
"""

import json
import numpy as np
from pathlib import Path
from prepare import load_data as _load_data, evaluate_model, TIME_BUDGET

# ── V2 Constants: G + P features only ─────────────────────────────────

VARIETIES = [
    "Gesha", "Bourbon", "Typica", "SL28", "SL34", "Caturra",
    "Catuai", "Pacamara", "Ethiopian Heirloom", "74158", "Castillo", "Catimor"
]
PROCESS_METHODS = ["washed", "natural", "honey_yellow", "honey_red", "honey_black", "wet_hulled"]
SOIL_TYPES = ["volcanic", "clay", "loam", "sandy", "laterite"]
DRYING_METHODS = ["raised_bed", "patio", "mechanical"]

# Only numerical features with actual variance in CQI data
NUM_RANGES_V2 = {
    "altitude_m": (800, 2400),
    "shade_pct": (0, 80),
    "latitude": (-25, 25),
    "delta_t_c": (5, 20),
    "fermentation_hours": (0, 200),
    "drying_days": (1, 30),
}

# Only categorical fields with variance
CAT_FIELDS_V2 = {
    "variety": VARIETIES,
    "method_p": PROCESS_METHODS,
    "soil_type": SOIL_TYPES,
    "drying_method": DRYING_METHODS,
}

BOOL_FIELDS_V2 = ["anaerobic"]

# Premium varieties (base_q > 8.0 in _VARIETY_PROFILES)
PREMIUM_VARIETIES = {"Gesha", "74158", "Ethiopian Heirloom", "SL28", "SL34"}

# Engineered feature names
ENGINEERED_FEATURES = ["ferm_capped", "ferm_risk", "premium_anaerobic"]

# 6 numerical + 26 categorical + 1 boolean + 3 engineered = 36
FEATURE_DIM_V2 = (
    len(NUM_RANGES_V2)
    + sum(len(v) for v in CAT_FIELDS_V2.values())
    + len(BOOL_FIELDS_V2)
    + len(ENGINEERED_FEATURES)
)


def encode_factors_v2(bean):
    """Encode only G + P features (no constant R/B)."""
    features = []

    # Numerical
    num_sources = {
        "altitude_m": bean["G"]["altitude_m"],
        "shade_pct": bean["G"]["shade_pct"],
        "latitude": abs(bean["G"]["latitude"]),
        "delta_t_c": bean["G"]["delta_t_c"],
        "fermentation_hours": bean["P"]["fermentation_hours"],
        "drying_days": bean["P"]["drying_days"],
    }
    for key in NUM_RANGES_V2:
        lo, hi = NUM_RANGES_V2[key]
        val = num_sources[key]
        features.append((val - lo) / (hi - lo) if hi > lo else 0.0)

    # Categorical
    cat_sources = {
        "variety": bean["G"]["variety"],
        "method_p": bean["P"]["method"],
        "soil_type": bean["G"]["soil_type"],
        "drying_method": bean["P"]["drying_method"],
    }
    for field, categories in CAT_FIELDS_V2.items():
        val = cat_sources[field]
        for cat in categories:
            features.append(1.0 if val == cat else 0.0)

    # Boolean
    anaerobic = bean["P"]["anaerobic"]
    features.append(1.0 if anaerobic else 0.0)

    # Engineered features
    ferm_h = bean["P"]["fermentation_hours"]
    variety = bean["G"]["variety"]
    features.append(min(ferm_h / 120.0, 1.0))  # ferm_capped: saturation at 120h
    features.append(1.0 if ferm_h > 150 and not anaerobic else 0.0)  # ferm_risk
    features.append(1.0 if variety in PREMIUM_VARIETIES and anaerobic else 0.0)  # premium_anaerobic

    return np.array(features, dtype=np.float64)


def get_feature_names_v2():
    """Return ordered feature names for V2 encoding."""
    names = list(NUM_RANGES_V2.keys())
    for field, categories in CAT_FIELDS_V2.items():
        for cat in categories:
            names.append(f"{field}_{cat}")
    for b in BOOL_FIELDS_V2:
        names.append(b)
    names.extend(ENGINEERED_FEATURES)
    return names


# ── V2 Extended Features: interaction terms ──────────────────────────

TOP_VARIETIES_FOR_INTERACTION = ["Typica", "Bourbon", "Caturra", "Gesha", "SL28"]

EXTENDED_FEATURE_NAMES = (
    [f"altitude_x_{v}" for v in TOP_VARIETIES_FOR_INTERACTION]
    + ["altitude_x_latitude", "altitude_x_delta_t", "latitude_x_delta_t"]
)

FEATURE_DIM_V2_EXTENDED = FEATURE_DIM_V2 + len(EXTENDED_FEATURE_NAMES)


def encode_factors_v2_extended(bean):
    """Encode G + P features plus interaction terms.

    Extra features (4):
      - altitude_m_norm * variety_onehot for Typica, Bourbon, Caturra
      - altitude_m_norm * abs(latitude_norm)
    """
    base = encode_factors_v2(bean)

    alt_lo, alt_hi = NUM_RANGES_V2["altitude_m"]
    alt_norm = (bean["G"]["altitude_m"] - alt_lo) / (alt_hi - alt_lo)

    lat_lo, lat_hi = NUM_RANGES_V2["latitude"]
    lat_norm = (abs(bean["G"]["latitude"]) - lat_lo) / (lat_hi - lat_lo)

    dt_lo, dt_hi = NUM_RANGES_V2["delta_t_c"]
    dt_norm = (bean["G"]["delta_t_c"] - dt_lo) / (dt_hi - dt_lo)

    variety = bean["G"]["variety"]
    interactions = []
    for v in TOP_VARIETIES_FOR_INTERACTION:
        interactions.append(alt_norm * (1.0 if variety == v else 0.0))
    interactions.append(alt_norm * lat_norm)
    interactions.append(alt_norm * dt_norm)
    interactions.append(lat_norm * dt_norm)

    return np.concatenate([base, np.array(interactions, dtype=np.float64)])


def get_feature_names_v2_extended():
    """Return ordered feature names for V2 extended encoding."""
    return get_feature_names_v2() + EXTENDED_FEATURE_NAMES


def load_data(path="data/beans.json", seed=42):
    """Delegate to original load_data."""
    return _load_data(path, seed)


def evaluate_model_v2(predict_fn, val_data):
    """Same as original evaluate_model."""
    return evaluate_model(predict_fn, val_data)


# Verify dimensions
assert FEATURE_DIM_V2 == len(get_feature_names_v2()), \
    f"Dim mismatch: {FEATURE_DIM_V2} vs {len(get_feature_names_v2())}"

assert FEATURE_DIM_V2_EXTENDED == len(get_feature_names_v2_extended()), \
    f"Extended dim mismatch: {FEATURE_DIM_V2_EXTENDED} vs {len(get_feature_names_v2_extended())}"

if __name__ == "__main__":
    print(f"V2 Feature Dimension: {FEATURE_DIM_V2}")
    print(f"V2 Extended Feature Dimension: {FEATURE_DIM_V2_EXTENDED}")
    print(f"Features: {get_feature_names_v2()}")
    print(f"Extended features: {EXTENDED_FEATURE_NAMES}")
    print(f"\nDropped from V1 (52 → {FEATURE_DIM_V2}):")
    print("  - R factors: roast_level(4), first_crack_temp_c, drop_temp_c, dtr_pct, total_time_s")
    print("  - B factors: method_b(6), grind_microns, water_temp_c, ratio, brew_time_s, water_tds_ppm")
    print(f"  - Removed {52 - FEATURE_DIM_V2} constant/noise features")
