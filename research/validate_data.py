"""Validate beans.json data quality and schema consistency."""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    data_dir = Path(__file__).parent / "data"

    # Load data
    with open(data_dir / "beans.json") as f:
        beans = json.load(f)

    print(f"Loaded {len(beans)} beans")
    errors = []

    # Required fields
    required_g = ["variety", "altitude_m", "country", "region", "soil_type", "shade_pct", "latitude", "delta_t_c"]
    required_p = ["method", "anaerobic", "fermentation_hours", "drying_method", "drying_days"]
    required_scores = ["overall"]

    for i, bean in enumerate(beans):
        bid = bean.get("id", f"bean_{i}")

        # Check structure
        for key in ["G", "P", "R", "B", "scores"]:
            if key not in bean:
                errors.append(f"{bid}: missing top-level key '{key}'")

        if "G" in bean:
            for field in required_g:
                if field not in bean["G"]:
                    errors.append(f"{bid}: missing G.{field}")
            # Range checks
            alt = bean["G"].get("altitude_m", 0)
            if not (0 <= alt <= 6000):
                errors.append(f"{bid}: altitude_m={alt} out of range [0, 6000]")
            lat = bean["G"].get("latitude", 0)
            if not (-90 <= lat <= 90):
                errors.append(f"{bid}: latitude={lat} out of range [-90, 90]")

        if "P" in bean:
            for field in required_p:
                if field not in bean["P"]:
                    errors.append(f"{bid}: missing P.{field}")
            ferm = bean["P"].get("fermentation_hours", 0)
            if not (0 <= ferm <= 500):
                errors.append(f"{bid}: fermentation_hours={ferm} out of range")

        if "scores" in bean:
            for field in required_scores:
                if field not in bean["scores"]:
                    errors.append(f"{bid}: missing scores.{field}")
            overall = bean["scores"].get("overall", 0)
            if not (50 <= overall <= 100):
                errors.append(f"{bid}: overall score={overall} out of range [50, 100]")

    # Check encoding roundtrip
    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_v2 import encode_factors_v2, FEATURE_DIM_V2

    for bean in beans[:10]:
        features = encode_factors_v2(bean)
        if len(features) != FEATURE_DIM_V2:
            errors.append(f"Feature dim mismatch: got {len(features)}, expected {FEATURE_DIM_V2}")
            break
        if not np.all(np.isfinite(features)):
            errors.append(f"{bean.get('id', '?')}: non-finite features after encoding")

    # Summary
    scores = [b["scores"]["overall"] for b in beans if "scores" in b and "overall" in b["scores"]]
    print(f"Score range: {min(scores):.1f} – {max(scores):.1f}")
    print(f"Score mean: {np.mean(scores):.1f}, std: {np.std(scores):.1f}")

    countries = set(b["G"]["country"] for b in beans if "G" in b)
    print(f"Countries ({len(countries)}): {sorted(countries)}")

    varieties = set(b["G"]["variety"] for b in beans if "G" in b)
    print(f"Varieties ({len(varieties)}): {sorted(varieties)}")

    if errors:
        print(f"\n{len(errors)} ERRORS found:")
        for e in errors[:20]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nData validation PASSED")

if __name__ == "__main__":
    main()
