"""
Coffee Attribution AutoResearch — Data Preparation & Evaluation
==============================================================
THIS FILE IS IMMUTABLE. The research agent must NOT modify it.

Provides:
- generate_seed_data(n)  → creates synthetic coffee bean dataset
- load_data(path)        → loads dataset, returns train/val split
- encode_factors(bean)   → converts bean dict to numerical feature vector
- evaluate_model(fn, data) → computes val_mae (lower is better)
"""

import json
import os
import sys
import hashlib
import numpy as np
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────
TIME_BUDGET = 300  # seconds per experiment

VARIETIES = [
    "Gesha", "Bourbon", "Typica", "SL28", "SL34", "Caturra",
    "Catuai", "Pacamara", "Ethiopian Heirloom", "74158", "Castillo", "Catimor"
]
PROCESS_METHODS = ["washed", "natural", "honey_yellow", "honey_red", "honey_black", "wet_hulled"]
SOIL_TYPES = ["volcanic", "clay", "loam", "sandy", "laterite"]
ROAST_LEVELS = ["light", "medium_light", "medium", "medium_dark"]
DRYING_METHODS = ["raised_bed", "patio", "mechanical"]
BREW_METHODS = ["v60", "kalita", "chemex", "aeropress", "french_press", "espresso"]

# Numerical feature ranges for min-max normalization
NUM_RANGES = {
    "altitude_m": (800, 2400),
    "shade_pct": (0, 80),
    "latitude": (-25, 25),
    "delta_t_c": (5, 20),
    "fermentation_hours": (0, 200),
    "drying_days": (1, 30),
    "first_crack_temp_c": (185, 210),
    "drop_temp_c": (190, 230),
    "dtr_pct": (15, 35),
    "total_time_s": (420, 900),
    "grind_microns": (200, 1200),
    "water_temp_c": (80, 100),
    "ratio": (1, 18),
    "brew_time_s": (25, 300),
    "water_tds_ppm": (30, 250),
}

# Categorical field → category list (for one-hot encoding)
CAT_FIELDS = {
    "variety": VARIETIES,
    "method_p": PROCESS_METHODS,
    "soil_type": SOIL_TYPES,
    "roast_level": ROAST_LEVELS,
    "drying_method": DRYING_METHODS,
    "method_b": BREW_METHODS,
}

# Boolean fields
BOOL_FIELDS = ["anaerobic"]

# Compute feature dimension
FEATURE_DIM = len(NUM_RANGES) + sum(len(v) for v in CAT_FIELDS.values()) + len(BOOL_FIELDS)


# ── Feature Encoding ──────────────────────────────────────────────────
def encode_factors(bean):
    """Convert a bean dict into a flat numerical feature vector."""
    features = []

    # Numerical features (min-max normalized to [0, 1])
    num_sources = {
        "altitude_m": bean["G"]["altitude_m"],
        "shade_pct": bean["G"]["shade_pct"],
        "latitude": abs(bean["G"]["latitude"]),  # use absolute latitude
        "delta_t_c": bean["G"]["delta_t_c"],
        "fermentation_hours": bean["P"]["fermentation_hours"],
        "drying_days": bean["P"]["drying_days"],
        "first_crack_temp_c": bean["R"]["first_crack_temp_c"],
        "drop_temp_c": bean["R"]["drop_temp_c"],
        "dtr_pct": bean["R"]["dtr_pct"],
        "total_time_s": bean["R"]["total_time_s"],
        "grind_microns": bean["B"]["grind_microns"],
        "water_temp_c": bean["B"]["water_temp_c"],
        "ratio": bean["B"]["ratio"],
        "brew_time_s": bean["B"]["brew_time_s"],
        "water_tds_ppm": bean["B"]["water_tds_ppm"],
    }
    for key in NUM_RANGES:
        lo, hi = NUM_RANGES[key]
        val = num_sources[key]
        features.append((val - lo) / (hi - lo) if hi > lo else 0.0)

    # Categorical features (one-hot)
    cat_sources = {
        "variety": bean["G"]["variety"],
        "method_p": bean["P"]["method"],
        "soil_type": bean["G"]["soil_type"],
        "roast_level": bean["R"]["roast_level"],
        "drying_method": bean["P"]["drying_method"],
        "method_b": bean["B"]["method"],
    }
    for field, categories in CAT_FIELDS.items():
        val = cat_sources[field]
        for cat in categories:
            features.append(1.0 if val == cat else 0.0)

    # Boolean features
    features.append(1.0 if bean["P"]["anaerobic"] else 0.0)

    return np.array(features, dtype=np.float64)


def get_feature_names():
    """Return ordered list of feature names matching encode_factors output."""
    names = list(NUM_RANGES.keys())
    for field, categories in CAT_FIELDS.items():
        for cat in categories:
            names.append(f"{field}_{cat}")
    for b in BOOL_FIELDS:
        names.append(b)
    return names


# ── Data Loading ──────────────────────────────────────────────────────
def load_data(path="data/beans.json", seed=42):
    """Load bean dataset and split into train/val (80/20)."""
    with open(path) as f:
        beans = json.load(f)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(beans))
    split = int(len(beans) * 0.8)

    train = [beans[i] for i in indices[:split]]
    val = [beans[i] for i in indices[split:]]
    return train, val


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate_model(predict_fn, val_data):
    """
    Evaluate a scoring model on validation data.

    predict_fn: takes a bean dict, returns predicted overall score (0-100).
    Returns: mean absolute error (lower is better).
    """
    errors = []
    for bean in val_data:
        try:
            pred = predict_fn(bean)
            actual = bean["scores"]["overall"]
            errors.append(abs(pred - actual))
        except Exception:
            errors.append(40.0)  # penalty for crashes
    return np.mean(errors)


# ── Synthetic Data Generation ─────────────────────────────────────────

# Domain knowledge: variety quality tiers and characteristics
_VARIETY_PROFILES = {
    "Gesha":              {"base_q": 9.0, "aroma_boost": 2.0, "acid_boost": 1.0, "sweet_boost": 1.5},
    "74158":              {"base_q": 8.5, "aroma_boost": 1.8, "acid_boost": 0.8, "sweet_boost": 1.2},
    "Ethiopian Heirloom": {"base_q": 8.3, "aroma_boost": 1.5, "acid_boost": 1.2, "sweet_boost": 1.0},
    "SL28":               {"base_q": 8.2, "aroma_boost": 0.8, "acid_boost": 1.8, "sweet_boost": 0.8},
    "SL34":               {"base_q": 8.0, "aroma_boost": 0.7, "acid_boost": 1.5, "sweet_boost": 0.8},
    "Pacamara":           {"base_q": 7.8, "aroma_boost": 1.0, "acid_boost": 1.0, "sweet_boost": 1.0},
    "Bourbon":            {"base_q": 7.5, "aroma_boost": 0.5, "acid_boost": 0.5, "sweet_boost": 1.5},
    "Typica":             {"base_q": 7.3, "aroma_boost": 0.4, "acid_boost": 0.6, "sweet_boost": 1.2},
    "Caturra":            {"base_q": 7.0, "aroma_boost": 0.3, "acid_boost": 0.5, "sweet_boost": 0.8},
    "Catuai":             {"base_q": 6.8, "aroma_boost": 0.2, "acid_boost": 0.4, "sweet_boost": 0.7},
    "Castillo":           {"base_q": 6.5, "aroma_boost": 0.2, "acid_boost": 0.3, "sweet_boost": 0.5},
    "Catimor":            {"base_q": 6.0, "aroma_boost": 0.1, "acid_boost": 0.2, "sweet_boost": 0.3},
}

# Country → typical region, altitude range, soil, latitude
_ORIGINS = [
    {"country": "Ethiopia", "regions": ["Yirgacheffe", "Guji", "Sidamo", "Limu"], "alt": (1800, 2300), "soil": "volcanic", "lat": (6, 9), "dt": (10, 18), "shade": (30, 70)},
    {"country": "Kenya", "regions": ["Nyeri", "Kiambu", "Kirinyaga", "Murang'a"], "alt": (1600, 2100), "soil": "volcanic", "lat": (-1, 1), "dt": (10, 16), "shade": (10, 40)},
    {"country": "Panama", "regions": ["Boquete", "Volcan", "Renacimiento"], "alt": (1400, 2000), "soil": "volcanic", "lat": (8, 9), "dt": (8, 14), "shade": (30, 60)},
    {"country": "Colombia", "regions": ["Huila", "Nariño", "Cauca", "Tolima"], "alt": (1400, 2100), "soil": "volcanic", "lat": (1, 7), "dt": (8, 15), "shade": (20, 50)},
    {"country": "Guatemala", "regions": ["Antigua", "Huehuetenango", "Atitlán"], "alt": (1300, 1900), "soil": "volcanic", "lat": (14, 16), "dt": (10, 18), "shade": (30, 60)},
    {"country": "Costa Rica", "regions": ["Tarrazú", "West Valley", "Central Valley"], "alt": (1200, 1800), "soil": "volcanic", "lat": (9, 11), "dt": (8, 14), "shade": (20, 50)},
    {"country": "Brazil", "regions": ["Cerrado", "Mogiana", "Sul de Minas"], "alt": (900, 1400), "soil": "loam", "lat": (-15, -22), "dt": (8, 15), "shade": (0, 30)},
    {"country": "Indonesia", "regions": ["Aceh", "Lintong", "Toraja", "Java"], "alt": (900, 1600), "soil": "volcanic", "lat": (-2, 5), "dt": (5, 10), "shade": (40, 70)},
    {"country": "Yemen", "regions": ["Haraz", "Bani Matar", "Matari"], "alt": (1500, 2200), "soil": "sandy", "lat": (13, 16), "dt": (12, 20), "shade": (10, 30)},
    {"country": "China", "regions": ["Yunnan Pu'er", "Yunnan Baoshan", "Hainan"], "alt": (1000, 1800), "soil": "laterite", "lat": (18, 25), "dt": (8, 15), "shade": (20, 50)},
]

# Roast level ↔ variety compatibility (bonus to score)
_ROAST_VARIETY_COMPAT = {
    ("Gesha", "light"): 1.5, ("Gesha", "medium_light"): 1.0,
    ("74158", "light"): 1.2, ("74158", "medium_light"): 1.0,
    ("Ethiopian Heirloom", "light"): 0.8, ("Ethiopian Heirloom", "medium_light"): 1.2,
    ("SL28", "medium"): 1.5, ("SL28", "medium_light"): 1.0,
    ("Bourbon", "medium"): 1.2, ("Bourbon", "medium_dark"): 1.0,
    ("Typica", "medium"): 1.0, ("Typica", "medium_light"): 0.8,
    ("Caturra", "medium"): 0.8,
    ("Catimor", "medium_dark"): 0.5,
}


def _compute_score(bean, rng):
    """Compute realistic cupping scores based on domain knowledge."""
    variety = bean["G"]["variety"]
    vp = _VARIETY_PROFILES[variety]

    # G-factor contribution
    alt_norm = (bean["G"]["altitude_m"] - 800) / 1600  # 0-1
    dt_norm = (bean["G"]["delta_t_c"] - 5) / 15
    shade_norm = bean["G"]["shade_pct"] / 80
    soil_bonus = 0.5 if bean["G"]["soil_type"] == "volcanic" else 0.0
    g_score = vp["base_q"] + alt_norm * 1.5 + dt_norm * 0.8 + shade_norm * 0.4 + soil_bonus

    # P-factor contribution
    method = bean["P"]["method"]
    anaerobic = bean["P"]["anaerobic"]
    ferm_h = bean["P"]["fermentation_hours"]

    p_base = {"washed": 0.0, "natural": 0.3, "honey_yellow": 0.1, "honey_red": 0.2,
              "honey_black": 0.3, "wet_hulled": -0.2}[method]
    anaerobic_bonus = 0.8 if anaerobic else 0.0
    ferm_bonus = min(ferm_h / 120, 1.0) * 0.6  # diminishing returns
    # Risk: very long fermentation can hurt
    ferm_risk = -0.5 if ferm_h > 150 and not anaerobic else 0.0
    drying_bonus = 0.3 if bean["P"]["drying_method"] == "raised_bed" else 0.0
    p_score = p_base + anaerobic_bonus + ferm_bonus + ferm_risk + drying_bonus

    # R-factor contribution
    roast = bean["R"]["roast_level"]
    dtr = bean["R"]["dtr_pct"]
    compat_key = (variety, roast)
    roast_compat = _ROAST_VARIETY_COMPAT.get(compat_key, 0.0)
    # DTR sweet spot: 20-25% is optimal for specialty
    dtr_opt = 1.0 - abs(dtr - 22.5) / 10
    r_score = roast_compat + dtr_opt * 0.5

    # B-factor contribution (brewing quality)
    temp = bean["B"]["water_temp_c"]
    ratio = bean["B"]["ratio"]
    # Light roast needs higher temp
    if roast in ("light", "medium_light"):
        temp_opt = 1.0 - abs(temp - 93) / 10
    else:
        temp_opt = 1.0 - abs(temp - 90) / 10
    ratio_opt = 1.0 - abs(ratio - 15.5) / 5
    b_score = temp_opt * 0.3 + ratio_opt * 0.3

    # Interaction effects
    # G×P: high quality genetics + anaerobic = synergy
    gp_interact = 0.6 if vp["base_q"] > 8.0 and anaerobic else 0.0
    # R×B: roast-brew alignment
    rb_interact = 0.4 if (roast == "light" and temp > 92) or (roast == "medium" and 88 < temp < 93) else 0.0
    # G×R: variety-roast compatibility amplifier
    gr_interact = roast_compat * 0.3

    # Combine with weights
    raw = (g_score * 0.35 + p_score * 0.25 + r_score * 0.25 + b_score * 0.15
           + gp_interact + rb_interact + gr_interact)

    # Scale to SCA-style 60-100 range
    # raw typically ranges ~2.5 to 5.5 → map to 65-95
    overall = np.clip(raw * 10.0 + 40, 62, 97)

    # Add realistic noise
    overall += rng.normal(0, 1.2)
    overall = np.clip(overall, 60, 100)

    # Derive sub-scores from overall + variety profile + noise
    base_sub = overall / 12  # roughly 5-8.3
    scores = {
        "overall": round(float(overall), 1),
        "aroma": round(np.clip(base_sub + vp["aroma_boost"] * 0.3 + rng.normal(0, 0.2), 5.5, 10), 1),
        "acidity": round(np.clip(base_sub + vp["acid_boost"] * 0.3 + rng.normal(0, 0.2), 5.5, 10), 1),
        "sweetness": round(np.clip(base_sub + vp["sweet_boost"] * 0.3 + rng.normal(0, 0.2), 5.5, 10), 1),
        "body": round(np.clip(base_sub + rng.normal(0, 0.3), 5.5, 10), 1),
        "aftertaste": round(np.clip(base_sub + rng.normal(0, 0.25), 5.5, 10), 1),
        "balance": round(np.clip(base_sub + rng.normal(0, 0.2), 5.5, 10), 1),
    }
    return scores


def generate_seed_data(n=150, seed=2024):
    """Generate a synthetic but realistic specialty coffee dataset."""
    rng = np.random.RandomState(seed)
    beans = []

    for i in range(n):
        # Pick origin
        origin = _ORIGINS[rng.randint(len(_ORIGINS))]

        # Pick variety (weighted: better varieties more likely from higher-quality origins)
        if origin["country"] in ("Ethiopia",):
            variety_pool = ["Gesha", "Ethiopian Heirloom", "74158", "Bourbon", "Typica"]
        elif origin["country"] in ("Kenya",):
            variety_pool = ["SL28", "SL34", "Bourbon"]
        elif origin["country"] in ("Panama",):
            variety_pool = ["Gesha", "Caturra", "Catuai", "Typica"]
        elif origin["country"] in ("Colombia",):
            variety_pool = ["Caturra", "Castillo", "Bourbon", "Typica", "Gesha"]
        elif origin["country"] in ("Brazil",):
            variety_pool = ["Bourbon", "Catuai", "Typica", "Caturra"]
        elif origin["country"] in ("China",):
            variety_pool = ["Typica", "Catimor", "Caturra", "Bourbon"]
        else:
            variety_pool = VARIETIES[:8]

        variety = variety_pool[rng.randint(len(variety_pool))]
        region = origin["regions"][rng.randint(len(origin["regions"]))]

        alt_lo, alt_hi = origin["alt"]
        lat_lo, lat_hi = origin["lat"]
        dt_lo, dt_hi = origin["dt"]
        sh_lo, sh_hi = origin["shade"]

        # Processing
        method = PROCESS_METHODS[rng.randint(len(PROCESS_METHODS))]
        anaerobic = rng.random() < 0.25  # 25% chance
        if method == "washed":
            ferm_h = rng.uniform(12, 48)
        elif method == "natural":
            ferm_h = rng.uniform(24, 120)
        elif method.startswith("honey"):
            ferm_h = rng.uniform(18, 72)
        else:
            ferm_h = rng.uniform(12, 36)
        if anaerobic:
            ferm_h *= 1.5  # anaerobic tends to go longer

        # Roasting
        roast = ROAST_LEVELS[rng.randint(len(ROAST_LEVELS))]
        fc_temp = rng.uniform(192, 205)
        drop_offset = {"light": (0, 5), "medium_light": (3, 10),
                       "medium": (8, 18), "medium_dark": (15, 28)}[roast]
        drop_temp = fc_temp + rng.uniform(*drop_offset)
        dtr = rng.uniform(17, 30)
        total_time = rng.uniform(480, 780)

        # Brewing
        brew_method = BREW_METHODS[rng.randint(len(BREW_METHODS))]
        if brew_method == "espresso":
            grind = rng.uniform(200, 400)
            water_temp = rng.uniform(90, 96)
            ratio = rng.uniform(1, 3)
            brew_time = rng.uniform(25, 35)
        else:
            grind = rng.uniform(400, 1000)
            water_temp = rng.uniform(85, 98)
            ratio = rng.uniform(13, 17)
            brew_time = rng.uniform(120, 280)
        water_tds = rng.uniform(50, 180)

        bean = {
            "id": f"bean_{i:04d}",
            "name": f"{region} {variety} {method.replace('_', ' ').title()}",
            "G": {
                "variety": variety,
                "altitude_m": round(rng.uniform(alt_lo, alt_hi)),
                "country": origin["country"],
                "region": region,
                "soil_type": origin["soil"],
                "shade_pct": round(rng.uniform(sh_lo, sh_hi), 1),
                "latitude": round(rng.uniform(lat_lo, lat_hi), 2),
                "delta_t_c": round(rng.uniform(dt_lo, dt_hi), 1),
            },
            "P": {
                "method": method,
                "anaerobic": anaerobic,
                "fermentation_hours": round(ferm_h, 1),
                "drying_method": DRYING_METHODS[rng.randint(len(DRYING_METHODS))],
                "drying_days": round(rng.uniform(3, 25)),
            },
            "R": {
                "roast_level": roast,
                "first_crack_temp_c": round(fc_temp, 1),
                "drop_temp_c": round(drop_temp, 1),
                "dtr_pct": round(dtr, 1),
                "total_time_s": round(total_time),
            },
            "B": {
                "method": brew_method,
                "grind_microns": round(grind),
                "water_temp_c": round(water_temp, 1),
                "ratio": round(ratio, 1),
                "brew_time_s": round(brew_time),
                "water_tds_ppm": round(water_tds),
            },
        }

        bean["scores"] = _compute_score(bean, rng)
        beans.append(bean)

    return beans


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--generate" in sys.argv:
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        beans = generate_seed_data(150)
        out_path = data_dir / "beans.json"
        with open(out_path, "w") as f:
            json.dump(beans, f, indent=2, ensure_ascii=False)

        # Print summary stats
        scores = [b["scores"]["overall"] for b in beans]
        print(f"Generated {len(beans)} beans → {out_path}")
        print(f"Score range: {min(scores):.1f} – {max(scores):.1f}")
        print(f"Mean: {np.mean(scores):.1f}, Std: {np.std(scores):.1f}")
        print(f"Feature dimension: {FEATURE_DIM}")

        # Country distribution
        countries = {}
        for b in beans:
            c = b["G"]["country"]
            countries[c] = countries.get(c, 0) + 1
        print("Countries:", dict(sorted(countries.items(), key=lambda x: -x[1])))
    else:
        print("Usage: python prepare.py --generate")
        print(f"  Feature dimension: {FEATURE_DIM}")
