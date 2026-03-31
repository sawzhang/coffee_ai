# Coffee Attribution AutoResearch — Agent Instructions

## Objective

You are an autonomous research agent optimizing a **specialty coffee scoring model**.
Your single metric: **val_mae** (validation mean absolute error). **Lower is better.**

Current baseline: **val_mae = 1.90** (GradientBoostingRegressor on 966 real CQI beans)

## Rules

1. You may ONLY modify `train.py`
2. `prepare.py` and `ingest.py` are READ-ONLY
3. Available packages: numpy, scipy, scikit-learn, pandas, matplotlib
4. You CANNOT install new packages
5. You CANNOT modify the evaluation function
6. Each experiment has a **5-minute wall-clock time budget** (CPU only, no GPU)
7. **NEVER STOP** — once the loop begins, keep experimenting

## Dataset

- **966 real CQI coffee beans** (arabica, from Coffee Quality Institute)
- 52 features: 15 numerical + 36 one-hot categorical + 1 boolean
- Target: `scores.overall` (range: 63.1–90.6, mean: 82.1, std: 2.9)
- Train/val split: 80/20 (772 train, 194 val, fixed seed=42)
- **Key characteristics**:
  - Altitude is the strongest single predictor
  - Latitude (abs) captures climate zone effects
  - Variety has moderate importance (Typica most common at 504/966)
  - Processing method matters (washed=686, natural=216)
  - R and B factors are constant (CQI standard cupping protocol) — model may learn to ignore them

## Train.py Architecture

```python
# Current configuration knobs:
MODEL_TYPE = "gbr"          # Options: gbr, ridge, lasso, elastic, rf, svr, linear_sgd
GBR_PARAMS = {...}          # GradientBoostingRegressor hyperparams
RF_PARAMS = {...}           # RandomForestRegressor hyperparams
SVR_PARAMS = {...}          # SVR hyperparams
LINEAR_ALPHA = 1.0          # Regularization for Ridge/Lasso/ElasticNet
ELASTIC_L1_RATIO = 0.5
USE_SCALER = True           # StandardScaler preprocessing
POLY_DEGREE = 0             # Polynomial feature expansion (0=off, 2=interactions)
```

The agent modifies these configs OR restructures the training pipeline entirely.

## Experiment Protocol

```
1. Read current train.py
2. Propose a modification (document what and why)
3. git add train.py && git commit -m "experiment: <description>"
4. Run: python3 train.py > run.log 2>&1
5. Extract val_mae from output
6. IF val_mae < previous best:
     → STATUS: KEPT
     → Record in results.tsv
   ELSE:
     → STATUS: DISCARDED
     → git reset --hard HEAD~1
     → Record in results.tsv
7. GOTO 1
```

## results.tsv Format

```
commit	val_mae	num_params	status	description
```

Initialize with header if file doesn't exist.

## Experiment Ideas (Priority-Ordered)

### Phase 1: Hyperparameter Tuning (GBR baseline)
1. **Run baseline as-is** → establish val_mae = 1.90
2. Increase n_estimators to 500 (more trees)
3. Try max_depth = 3, 5, 6 (shallower vs deeper)
4. Try learning_rate = 0.01, 0.03, 0.1 (slower = more trees needed)
5. Try subsample = 0.6, 0.7, 0.9
6. Try min_samples_leaf = 3, 10, 15
7. Grid search: n_estimators × learning_rate × max_depth

### Phase 2: Model Comparison
8. Switch to RandomForestRegressor (tune n_estimators, max_depth)
9. Try Ridge regression (tune alpha)
10. Try SVR with RBF kernel (tune C, epsilon, gamma)
11. Try ElasticNet (tune alpha, l1_ratio)
12. Try LassoCV or RidgeCV for auto-tuned regularization

### Phase 3: Feature Engineering
13. Add POLY_DEGREE = 2 (interaction features) with Ridge/Lasso
14. Custom interaction features: altitude×variety, process×drying
15. Feature selection via sklearn.feature_selection (mutual_info, f_regression)
16. PCA on one-hot features to reduce dimensionality
17. Target encoding for variety (mean score per variety)

### Phase 4: Ensemble Methods
18. VotingRegressor: GBR + RF + Ridge average
19. StackingRegressor: GBR + RF base, Ridge meta
20. BaggingRegressor wrapping SVR
21. AdaBoostRegressor

### Phase 5: Advanced
22. HistGradientBoostingRegressor (faster, handles NaN)
23. Custom loss function (Huber loss for robustness)
24. Cross-validated hyperparameter search (GridSearchCV within time budget)
25. Feature importance-based pruning → retrain on top-K features
26. Quantile regression for prediction intervals

## Key Domain Knowledge

From the data and initial analysis:
- **Altitude is king**: 0.27 feature importance, strongest single predictor
- **Latitude matters**: 0.18 importance, captures climate zone
- **Temperature differential (delta_t_c)**: 0.09, affects cherry maturation
- **Drying method**: raised_bed vs patio is significant
- **Shade percentage**: moderate impact
- **R and B factors are CONSTANT** in CQI data (standard cupping protocol)
  → These features carry no information. Consider dropping them to reduce noise.
- Score distribution is tight (std=2.9) → small MAE improvements matter

## Export

After each KEPT experiment, the model auto-exports to `../site/data/model.json`.
After completing a batch, run `python3 export_results.py` to update results.json.
