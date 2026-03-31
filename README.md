# Coffee Attribution AutoResearch

> Specialty coffee multi-factor attribution system, built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

**Live Demo**: [GitHub Pages](https://sawzhang.github.io/coffee_ai/) | **API**: [Render](https://coffee-ai-6fhq.onrender.com/api/health)

## What It Does

Predicts specialty coffee quality scores from growing, processing, roasting, and brewing factors using a stacking ensemble model (GBR + RF, val_mae = 1.84 on 966 CQI arabica beans).

```
Coffee Score = f(G, P, R, B, U)

G = Genetics/Geography    (variety, altitude, soil, climate)
P = Processing            (method, fermentation, drying)
R = Roasting              (level, DTR, temperature profile)
B = Brewing               (method, grind, water temp, ratio)
U = User Preferences      (acidity, sweetness, body, complexity)
```

## Features

- **Interactive Scorer** — Adjust G/P factors, get real-time quality predictions with 80% confidence intervals
- **Factor Recombination Engine** — Fix genetics, explore all processing combinations
- **Personalized Recommendations** — Set taste preferences, get matched beans
- **Bean Comparison** — Side-by-side scoring with attribution deltas
- **Bilingual UI** — Full Chinese/English support with browser auto-detection
- **AutoResearch Protocol** — AI agent iteratively optimizes model via `train.py`

## Architecture

```
research/          Python ML engine (966 CQI beans, 44 features, stacking ensemble)
api/               FastAPI backend (/predict, /recommend, /explore, /compare)
site/              Static SPA (Chart.js, bilingual, mobile responsive)
```

```
train_v2.py → model.joblib ──→ API (Render)
                             └→ model.json → site (GitHub Pages, local fallback)
```

## Quick Start

### Local development

```bash
# Install dependencies
cd research && pip install -e ".[dev,api]"

# Train model
python3 train_v2.py

# Start API server
cd .. && uvicorn api.server:app --reload --port 8000

# Preview site (separate terminal)
python3 -m http.server --directory site 8080
```

### Docker

```bash
docker compose up --build
# API + site at http://localhost:8000
```

### Autonomous research

```bash
cd research
# Launch Claude Code with program.md as context
# Agent iterates on train.py, tracking experiments in results.tsv
python3 train_v2.py --optimize  # or run hyperparameter optimization
```

## Model Performance

| Metric | Value |
|--------|-------|
| val_mae | **1.840** |
| cv_mae (5-fold) | 1.817 |
| Feature dim | 44 (36 base + 8 interactions) |
| Dataset | 966 CQI arabica beans |
| Top features | altitude (0.11), altitude x latitude (0.11), altitude x Typica (0.10) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict quality score + confidence interval |
| `/api/compare` | POST | Side-by-side bean comparison |
| `/api/recommend` | POST | Personalized bean recommendations |
| `/api/explore` | POST | Factor recombination engine |
| `/api/model-info` | GET | Model metadata and weights |
| `/api/health` | GET | Health check |

## Project Structure

```
coffee_ai/
├── api/server.py              FastAPI backend (10 endpoints)
├── research/
│   ├── prepare_v2.py          Feature encoding (G+P, 44 features)
│   ├── train_v2.py            Stacking ensemble training
│   ├── validate_data.py       Data quality checks
│   ├── program.md             AutoResearch protocol
│   └── data/beans.json        966 CQI arabica beans
├── site/
│   ├── index.html             SPA (7 interactive sections)
│   ├── js/{app,scoring,charts,i18n,data}.js
│   └── data/{model,results,beans_summary}.json
├── tests/                     59 tests (API, model, data, edge cases)
├── Dockerfile                 3-stage build (deps → train → runtime)
├── render.yaml                Render deployment config
└── .github/workflows/
    ├── ci.yml                 Lint + test + model quality gate + Docker
    └── pages.yml              GitHub Pages auto-deploy
```

## Tech Stack

**ML**: scikit-learn (GBR stacking + quantile regression), numpy, pandas
**API**: FastAPI, Pydantic, uvicorn
**Frontend**: Vanilla JS, Chart.js, CSS custom properties
**Deploy**: GitHub Pages (static), Render (API Docker), GitHub Actions CI
**Dev**: pytest, ruff, uv

## License

MIT
