"""
Coffee Data Ingestion Pipeline
==============================
Converts real-world coffee datasets (CQI/TidyTuesday) into our beans.json format.

Usage:
  python3 ingest.py                    # Process all raw data
  python3 ingest.py --stats            # Show dataset statistics
"""

import csv
import json
import sys
import re
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_PATH = Path(__file__).parent / "data" / "beans.json"

# ── Mapping tables ────────────────────────────────────────

# Normalize variety names to our schema
VARIETY_MAP = {
    "bourbon": "Bourbon",
    "caturra": "Caturra",
    "catuai": "Catuai",
    "typica": "Typica",
    "catimor": "Catimor",
    "castillo": "Castillo",
    "gesha": "Gesha",
    "geisha": "Gesha",
    "sl28": "SL28",
    "sl34": "SL34",
    "sl 28": "SL28",
    "sl 34": "SL34",
    "pacamara": "Pacamara",
    "ethiopian heirlooms": "Ethiopian Heirloom",
    "ethiopian heirloom": "Ethiopian Heirloom",
    "ethiopian yirgacheffe": "Ethiopian Heirloom",
    "74158": "74158",
    "74110": "74158",  # similar Ethiopian selection
    "mundo novo": "Typica",  # Bourbon × Typica hybrid
    "maragogipe": "Typica",  # Typica mutation
    "marigojipe": "Typica",
    "hawaiian kona": "Typica",  # Kona is Typica-based
    "java": "Typica",  # Java is Typica lineage
    "blue mountain": "Typica",  # Blue Mountain is Typica
    "yellow bourbon": "Bourbon",
    "red bourbon": "Bourbon",
    "pink bourbon": "Bourbon",
    "mandheling": "Typica",
    "ruiru 11": "Catimor",  # Catimor lineage
    "arusha": "Typica",  # Typica variant
    "moka peaberry": "Bourbon",
    "other": None,
    "na": None,
    "none": None,
    "": None,
}

# Normalize processing methods
PROCESS_MAP = {
    "washed / wet": "washed",
    "washed": "washed",
    "wet": "washed",
    "fully washed": "washed",
    "natural / dry": "natural",
    "natural": "natural",
    "dry": "natural",
    "pulped natural / honey": "honey_yellow",
    "pulped natural": "honey_yellow",
    "honey": "honey_yellow",
    "semi-washed / semi-pulped": "honey_red",
    "semi-washed": "honey_red",
    "semi washed": "honey_red",
    "semi-pulped": "honey_red",
    "wet hulled": "wet_hulled",
    "wet-hulled": "wet_hulled",
    "giling basah": "wet_hulled",
    "other": None,
    "na": None,
    "none": None,
    "": None,
}

# Country → typical attributes
COUNTRY_DEFAULTS = {
    "Ethiopia": {"soil": "volcanic", "shade": 45, "lat": 7.5, "dt": 14},
    "Kenya": {"soil": "volcanic", "shade": 25, "lat": 0, "dt": 13},
    "Colombia": {"soil": "volcanic", "shade": 35, "lat": 4, "dt": 11},
    "Guatemala": {"soil": "volcanic", "shade": 45, "lat": 15, "dt": 14},
    "Brazil": {"soil": "loam", "shade": 15, "lat": -18, "dt": 11},
    "Costa Rica": {"soil": "volcanic", "shade": 35, "lat": 10, "dt": 11},
    "Honduras": {"soil": "clay", "shade": 30, "lat": 14, "dt": 10},
    "El Salvador": {"soil": "volcanic", "shade": 40, "lat": 13, "dt": 12},
    "Panama": {"soil": "volcanic", "shade": 45, "lat": 8.5, "dt": 11},
    "Mexico": {"soil": "loam", "shade": 35, "lat": 17, "dt": 12},
    "Peru": {"soil": "clay", "shade": 30, "lat": -8, "dt": 10},
    "Indonesia": {"soil": "volcanic", "shade": 50, "lat": -2, "dt": 7},
    "India": {"soil": "laterite", "shade": 40, "lat": 12, "dt": 10},
    "Tanzania": {"soil": "volcanic", "shade": 30, "lat": -4, "dt": 13},
    "Rwanda": {"soil": "volcanic", "shade": 25, "lat": -2, "dt": 12},
    "Burundi": {"soil": "clay", "shade": 20, "lat": -3, "dt": 11},
    "Uganda": {"soil": "volcanic", "shade": 35, "lat": 1, "dt": 10},
    "China": {"soil": "laterite", "shade": 30, "lat": 22, "dt": 11},
    "Yemen": {"soil": "sandy", "shade": 15, "lat": 15, "dt": 16},
    "Nicaragua": {"soil": "volcanic", "shade": 35, "lat": 13, "dt": 10},
    "Papua New Guinea": {"soil": "volcanic", "shade": 50, "lat": -5, "dt": 8},
    "Philippines": {"soil": "volcanic", "shade": 45, "lat": 10, "dt": 8},
    "Thailand": {"soil": "clay", "shade": 40, "lat": 18, "dt": 10},
    "Vietnam": {"soil": "laterite", "shade": 30, "lat": 14, "dt": 9},
    "Taiwan": {"soil": "loam", "shade": 35, "lat": 23, "dt": 10},
    "Haiti": {"soil": "clay", "shade": 40, "lat": 19, "dt": 10},
    "Myanmar": {"soil": "clay", "shade": 35, "lat": 20, "dt": 12},
    "Laos": {"soil": "laterite", "shade": 40, "lat": 18, "dt": 10},
    "Malawi": {"soil": "clay", "shade": 25, "lat": -14, "dt": 12},
    "Zambia": {"soil": "clay", "shade": 20, "lat": -13, "dt": 13},
    "Mauritius": {"soil": "volcanic", "shade": 30, "lat": -20, "dt": 8},
}

DEFAULT_ATTRS = {"soil": "loam", "shade": 30, "lat": 5, "dt": 10}


def parse_altitude(alt_str):
    """Parse altitude string to float meters."""
    if not alt_str or alt_str.strip() in ("", "NA", "None", "nan"):
        return None
    alt_str = alt_str.strip()
    try:
        val = float(alt_str)
        # If value seems like feet (>3000), convert
        if val > 3000:
            val = val * 0.3048
        return max(400, min(3000, val))
    except ValueError:
        # Try to extract number
        nums = re.findall(r'[\d.]+', alt_str)
        if nums:
            val = float(nums[0])
            if val > 3000:
                val = val * 0.3048
            return max(400, min(3000, val))
        return None


def normalize_country(name):
    """Normalize country names."""
    if not name:
        return None
    name = name.strip()
    fixes = {
        "Cote d?Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "United States (Hawaii)": "Hawaii",
        "United States (Puerto Rico)": "Puerto Rico",
        "United States": "Hawaii",
    }
    return fixes.get(name, name)


def normalize_variety(raw):
    """Map raw variety name to schema variety."""
    if not raw:
        return None
    key = raw.strip().lower()
    # Direct lookup
    if key in VARIETY_MAP:
        return VARIETY_MAP[key]
    # Partial match
    for pattern, mapped in VARIETY_MAP.items():
        if pattern and pattern in key:
            return mapped
    return None


def normalize_process(raw):
    """Map raw processing method to schema method."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in PROCESS_MAP:
        return PROCESS_MAP[key]
    for pattern, mapped in PROCESS_MAP.items():
        if pattern and pattern in key:
            return mapped
    return None


def load_tidytuesday():
    """Load and normalize TidyTuesday coffee ratings CSV."""
    path = RAW_DIR / "tidytuesday_coffee.csv"
    if not path.exists():
        print(f"  Not found: {path}")
        return []

    beans = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Parse scores
            try:
                overall = float(row["total_cup_points"])
                if overall < 60 or overall > 100:
                    continue
            except (ValueError, KeyError):
                continue

            # Normalize fields
            country = normalize_country(row.get("country_of_origin", ""))
            if not country:
                continue

            variety = normalize_variety(row.get("variety", ""))
            process = normalize_process(row.get("processing_method", ""))
            altitude = parse_altitude(row.get("altitude_mean_meters", ""))

            # Skip if too many missing fields
            if not variety and not process:
                continue

            # Fill defaults
            if not variety:
                variety = "Typica"  # most common default
            if not process:
                process = "washed"  # most common default
            if not altitude:
                altitude = 1400  # reasonable default

            # Country-based defaults
            c_defaults = COUNTRY_DEFAULTS.get(country, DEFAULT_ATTRS)

            # Parse sub-scores (CQI format: 0-10 scale)
            def score(field, default=7.5):
                try:
                    v = float(row.get(field, default))
                    return round(max(5.0, min(10.0, v)), 1)
                except (ValueError, TypeError):
                    return default

            region = row.get("region", "").strip() or country

            bean = {
                "id": f"cqi_{i:04d}",
                "name": f"{region} {variety} {process.replace('_', ' ').title()}",
                "G": {
                    "variety": variety,
                    "altitude_m": round(altitude),
                    "country": country,
                    "region": region[:50],
                    "soil_type": c_defaults["soil"],
                    "shade_pct": round(c_defaults["shade"], 1),
                    "latitude": round(c_defaults["lat"], 2),
                    "delta_t_c": round(c_defaults["dt"], 1),
                },
                "P": {
                    "method": process,
                    "anaerobic": False,  # CQI data predates anaerobic trend
                    "fermentation_hours": {"washed": 24, "natural": 48, "honey_yellow": 36,
                                           "honey_red": 48, "honey_black": 60,
                                           "wet_hulled": 18}.get(process, 24),
                    "drying_method": "raised_bed" if country in ("Ethiopia", "Kenya", "Rwanda") else "patio",
                    "drying_days": {"natural": 18, "honey_yellow": 12, "honey_red": 15,
                                    "honey_black": 18, "washed": 10, "wet_hulled": 8}.get(process, 12),
                },
                "R": {
                    "roast_level": "medium_light",  # CQI cupping standard: light-medium sample roast
                    "first_crack_temp_c": 198,
                    "drop_temp_c": 205,
                    "dtr_pct": 22,
                    "total_time_s": 600,
                },
                "B": {
                    "method": "v60",  # standard cupping method approximation
                    "grind_microns": 600,
                    "water_temp_c": 93,
                    "ratio": 15,
                    "brew_time_s": 240,
                    "water_tds_ppm": 120,
                },
                "scores": {
                    "overall": round(overall, 1),
                    "aroma": score("aroma"),
                    "acidity": score("acidity"),
                    "sweetness": score("sweetness"),
                    "body": score("body"),
                    "aftertaste": score("aftertaste"),
                    "balance": score("balance"),
                },
            }
            beans.append(bean)

    return beans


def load_cqi_arabica():
    """Load CQI arabica CSV (R-style column names)."""
    path = RAW_DIR / "cqi_arabica.csv"
    if not path.exists():
        print(f"  Not found: {path}")
        return []

    beans = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                overall = float(row.get("Total.Cup.Points", 0))
                if overall < 60 or overall > 100:
                    continue
            except ValueError:
                continue

            country = normalize_country(row.get("Country.of.Origin", ""))
            if not country:
                continue

            variety = normalize_variety(row.get("Variety", ""))
            process = normalize_process(row.get("Processing.Method", ""))
            altitude = parse_altitude(row.get("altitude_mean_meters",
                                              row.get("Altitude", "")))

            if not variety:
                variety = "Typica"
            if not process:
                process = "washed"
            if not altitude:
                altitude = 1400

            c_defaults = COUNTRY_DEFAULTS.get(country, DEFAULT_ATTRS)

            def score(field, default=7.5):
                try:
                    v = float(row.get(field, default))
                    return round(max(5.0, min(10.0, v)), 1)
                except (ValueError, TypeError):
                    return default

            region = row.get("Region", "").strip() or country

            bean = {
                "id": f"cqi_ar_{i:04d}",
                "name": f"{region} {variety} {process.replace('_', ' ').title()}",
                "G": {
                    "variety": variety,
                    "altitude_m": round(altitude),
                    "country": country,
                    "region": region[:50],
                    "soil_type": c_defaults["soil"],
                    "shade_pct": round(c_defaults["shade"], 1),
                    "latitude": round(c_defaults["lat"], 2),
                    "delta_t_c": round(c_defaults["dt"], 1),
                },
                "P": {
                    "method": process,
                    "anaerobic": False,
                    "fermentation_hours": {"washed": 24, "natural": 48, "honey_yellow": 36,
                                           "honey_red": 48, "honey_black": 60,
                                           "wet_hulled": 18}.get(process, 24),
                    "drying_method": "raised_bed" if country in ("Ethiopia", "Kenya", "Rwanda") else "patio",
                    "drying_days": {"natural": 18, "honey_yellow": 12, "honey_red": 15,
                                    "honey_black": 18, "washed": 10, "wet_hulled": 8}.get(process, 12),
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
                    "overall": round(overall, 1),
                    "aroma": score("Aroma"),
                    "acidity": score("Acidity"),
                    "sweetness": score("Sweetness"),
                    "body": score("Body"),
                    "aftertaste": score("Aftertaste"),
                    "balance": score("Balance"),
                },
            }
            beans.append(bean)

    return beans


def deduplicate(beans):
    """Remove near-duplicate beans (same country+variety+process+score)."""
    seen = set()
    unique = []
    for b in beans:
        key = (b["G"]["country"], b["G"]["variety"], b["P"]["method"],
               b["scores"]["overall"])
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def print_stats(beans):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total beans: {len(beans)}")

    scores = [b["scores"]["overall"] for b in beans]
    import numpy as np
    print(f"Score range: {min(scores):.1f} – {max(scores):.1f}")
    print(f"Score mean:  {np.mean(scores):.1f}")
    print(f"Score std:   {np.std(scores):.1f}")
    print(f"Score median:{np.median(scores):.1f}")

    print(f"\nCountries ({len(set(b['G']['country'] for b in beans))}):")
    for c, n in Counter(b["G"]["country"] for b in beans).most_common(15):
        print(f"  {c:25s} {n:4d}")

    print(f"\nVarieties ({len(set(b['G']['variety'] for b in beans))}):")
    for v, n in Counter(b["G"]["variety"] for b in beans).most_common():
        print(f"  {v:25s} {n:4d}")

    print(f"\nProcessing methods:")
    for m, n in Counter(b["P"]["method"] for b in beans).most_common():
        print(f"  {m:25s} {n:4d}")

    altitudes = [b["G"]["altitude_m"] for b in beans]
    print(f"\nAltitude range: {min(altitudes)}m – {max(altitudes)}m")
    print(f"Altitude mean:  {np.mean(altitudes):.0f}m")


def main():
    print("Loading TidyTuesday coffee ratings...")
    tt_beans = load_tidytuesday()
    print(f"  → {len(tt_beans)} beans")

    print("Loading CQI arabica data...")
    cqi_beans = load_cqi_arabica()
    print(f"  → {len(cqi_beans)} beans")

    # Merge and deduplicate
    all_beans = tt_beans + cqi_beans
    print(f"\nTotal before dedup: {len(all_beans)}")
    beans = deduplicate(all_beans)
    print(f"Total after dedup:  {len(beans)}")

    # Re-assign IDs
    for i, b in enumerate(beans):
        b["id"] = f"bean_{i:04d}"

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(beans, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH}")

    if "--stats" in sys.argv:
        print_stats(beans)
    else:
        # Always show brief stats
        scores = [b["scores"]["overall"] for b in beans]
        print(f"Score range: {min(scores):.1f} – {max(scores):.1f}")
        print(f"Countries: {len(set(b['G']['country'] for b in beans))}")
        print(f"Varieties: {len(set(b['G']['variety'] for b in beans))}")

    print("\nDone. Run 'python3 train.py' to train on real data.")


if __name__ == "__main__":
    main()
