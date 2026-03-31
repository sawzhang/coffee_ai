# Coffee Attribution AutoResearch

Specialty coffee multi-factor attribution system, built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

## Architecture

```
Coffee Score = f(G, P, R, B, U)

G = Genetics/Geography    (variety, altitude, soil, climate)
P = Processing            (method, fermentation, drying)
R = Roasting              (level, DTR, temperature profile)
B = Brewing               (method, grind, water temp, ratio)
U = User Profile          (taste preferences, browser-side)
```

**Two parts:**

- `research/` — Local Python research engine. An AI agent iteratively optimizes `train.py` to minimize prediction error (val_mae).
- `site/` — Static web visualization deployed to GitHub Pages. Shows factor weights, feature importance, experiment history, and an interactive scorer.

## Quick Start

### 1. Generate data & run baseline

```bash
cd research
python3 prepare.py --generate    # Generate 150 synthetic coffee beans
python3 train.py                 # Run baseline model
python3 export_results.py        # Export results to site/data/
```

### 2. Preview site locally

```bash
python3 -m http.server --directory site 8080
# Open http://localhost:8080
```

### 3. Run autonomous research

```bash
cd research
git checkout -b autoresearch/run1
# Launch Claude Code with program.md as context
# Agent will iterate on train.py, tracking experiments in results.tsv
```

### 4. Deploy to GitHub Pages

Push to `main` — GitHub Actions automatically deploys `site/` to Pages.

## Data Flow

```
[Local]                          [GitHub]
train.py (agent modifies)
    ↓
results.tsv (experiment log)
    ↓
export_results.py
    ↓
site/data/*.json  →  git push  →  GitHub Actions  →  GitHub Pages
```

## Project Structure

```
├── research/
│   ├── prepare.py          # Immutable: data loading, encoding, evaluation
│   ├── train.py            # Agent-modified: scoring model
│   ├── program.md          # Agent instructions (autoresearch protocol)
│   ├── export_results.py   # TSV → JSON for web
│   └── data/beans.json     # Coffee bean dataset
├── site/
│   ├── index.html          # Single-page app
│   ├── css/style.css
│   ├── js/                 # Scoring engine, charts, app logic
│   └── data/               # model.json, results.json, beans_summary.json
└── .github/workflows/
    └── pages.yml           # Auto-deploy on push
```
