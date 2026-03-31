# Coffee AI Data Roadmap

## Current Data Limitations

| Issue | Status | Impact |
|-------|--------|--------|
| R/B factors constant (CQI cupping protocol) | By design | Model blind to roast/brew |
| soil_type, shade_pct, delta_t_c are synthetic | Imputed by ingest.py | Model may learn noise |
| No tasting note descriptors | Missing entirely | Cannot predict flavor profiles |
| No brew parameter variation | Fixed protocol | Cannot optimize recipes |
| Only arabica (966 beans) | 28 robusta unused | Limited generalization |

## Phase A: Clean + Enrich Existing Data (Week 1-2)

### A1: Audit and tag data provenance
For each field in beans.json, mark whether it comes from real CQI data or was imputed:
- **Real CQI**: country, region, altitude, variety, processing_method, scores (overall + sub-scores)
- **Imputed/synthetic**: soil_type, shade_pct, delta_t_c, drying_method, drying_days, anaerobic, fermentation_hours

Action: Add `_provenance` field to each bean. Train a "clean features only" model variant
using only CQI-real fields to measure how much signal comes from real vs imputed data.

### A2: Scrape public cupping note databases
Sources with structured tasting notes + origin metadata:
- **Sweet Maria's** (sweetmarias.com/coffee): ~500 green coffees with detailed cupping notes,
  origin, variety, process, altitude. Notes use near-SCA vocabulary.
- **Coffee Review** (coffeereview.com): 6000+ reviews with scores + descriptor text.
  Requires NLP to extract structured descriptors.
- **CoffeeGeek** tasting database
- **Specialty Coffee Transaction Guide** (published annually by SCA): price + quality data

Output: 2000+ beans with `tasting_notes: ["jasmine", "citric", "cocoa", ...]` field.

### A3: Integrate CQI robusta + expand arabica
- Add 28 CQI robusta beans with `species` field
- Re-download CQI arabica data with ALL fields (the current ingest.py drops several
  columns that exist in the raw CSV: Owner, Farm.Name, In.Country.Partner, etc.)
- The raw CQI data has more detail than what ingest.py extracts

## Phase B: Brew Parameter Dataset (Week 3-6)

### B1: Community brew log schema
Design a standardized brew log that captures:
```json
{
  "bean_id": "...",
  "roast_date": "2026-03-15",
  "brew_date": "2026-03-25",
  "days_off_roast": 10,
  "equipment": {
    "brewer": "v60_02",
    "grinder": "comandante_c40",
    "grind_clicks": 24,
    "grind_microns_est": 700,
    "kettle": "fellow_stagg"
  },
  "recipe": {
    "dose_g": 15,
    "water_g": 250,
    "ratio": 16.7,
    "water_temp_c": 93,
    "total_time_s": 195,
    "bloom_time_s": 30,
    "bloom_water_g": 45,
    "pours": 3,
    "technique": "rao_spin"
  },
  "water": {
    "tds_ppm": 120,
    "gh_ppm": 50,
    "kh_ppm": 40,
    "recipe": "third_wave_water_classic"
  },
  "result": {
    "tds_out": 1.35,
    "extraction_yield_pct": 20.1,
    "drawdown_time_s": 150
  },
  "taste": {
    "overall": 4,
    "acidity_intensity": 3,
    "acidity_quality": "bright_citric",
    "sweetness": 4,
    "bitterness": 1,
    "body": 3,
    "aftertaste": 4,
    "balance_feedback": "slightly_under",
    "notes_detected": ["lemon", "brown_sugar", "tea_like"],
    "adjust_suggestion": "grind_finer"
  }
}
```

### B2: Grinder calibration database
Map common grinder settings to microns:
- Comandante C40: click 15=400μm, 20=550μm, 25=700μm, 30=850μm
- 1Zpresso JX/J-Max: click mappings
- Timemore C2/C3: click mappings
- Baratza Encore/Virtuoso: setting mappings
- Fellow Ode / Ode Gen 2: setting mappings

This normalizes the most variable input (grind size) across equipment.

### B3: Extraction science reference data
From published research:
- Moroney et al. 2019: coffee extraction kinetics model
- Hendon et al. 2014: water mineral impact on extraction
- Rao "Everything but Espresso": empirical extraction curves
- Perger extraction yield vs TDS charts

Use these to build a physics-informed extraction model as a prior,
then refine with community brew log data.

## Phase C: Flavor Descriptor Corpus (Week 4-8)

### C1: SCA Flavor Wheel encoding
Encode the SCA flavor wheel as a hierarchical taxonomy:
```
Level 1 (9): Floral, Fruity, Sour/Fermented, Green/Vegetal, Other, Roasted,
             Spices, Nutty/Cocoa, Sweet
Level 2 (30): Floral→{Floral, Black Tea}, Fruity→{Berry, Dried Fruit,
              Other Fruit, Citrus Fruit}, ...
Level 3 (85): Berry→{Blackberry, Raspberry, Blueberry, Strawberry}, ...
```

### C2: NLP pipeline for tasting notes
Build a parser that extracts SCA wheel descriptors from free-text cupping notes:
- Input: "Bright with notes of bergamot, juicy stone fruit acidity,
          honey sweetness, medium body with a long cocoa finish"
- Output: ["bergamot", "stone_fruit", "honey", "cocoa"],
          {acidity: "bright_juicy", body: "medium", finish: "long"}

Use an LLM (Claude API) to batch-process tasting note text into
structured SCA wheel labels. Human-verify a sample for quality.

### C3: Roast profile paired cupping data
Partner with or scrape from:
- Cropster (roast profile + cupping data platform, widely used by roasters)
- Ikawa (home sample roaster with paired roast curves + cupping notes)
- Roasters who publish "same lot, different roast" comparisons

Target: 200+ paired entries (same green bean, 2-3 roast profiles, cupping notes for each).

## Data Volume Targets

| Dataset | Current | Phase A | Phase B | Phase C |
|---------|---------|---------|---------|---------|
| Beans with scores | 966 | 2000+ | 2000+ | 3000+ |
| Beans with tasting notes | 0 | 1500+ | 1500+ | 3000+ |
| Brew logs | 0 | 0 | 500+ | 1000+ |
| Roast profile pairs | 0 | 0 | 0 | 200+ |
| Water chemistry entries | 0 | 0 | 200+ | 500+ |

## Impact on Model Quality

Current: val_mae = 1.84 on CQI scores (noise floor ~1.2)

With Phase A (clean data + tasting notes):
- Flavor profile multi-label classification becomes possible
- "Clean features only" model likely has similar val_mae (confirming imputed features are noise)
- NEW metric: flavor descriptor F1 score

With Phase B (brew parameters):
- Extraction yield prediction becomes possible
- Recipe optimization loop becomes possible
- NEW metric: extraction yield MAE, user satisfaction rate

With Phase C (full flavor corpus):
- Flavor similarity engine becomes possible
- Roast impact modeling becomes possible
- The system transitions from "CQI score predictor" to "flavor experience predictor"
