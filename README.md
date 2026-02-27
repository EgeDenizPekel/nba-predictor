# NBA Match Outcome Predictor

Binary classification of NBA home team win/loss. The emphasis is on DS and ML rigor: leakage-free feature engineering, calibrated probability outputs, SHAP interpretability, and honest evaluation. FastAPI + React are secondary layers.

## What this project demonstrates

- Leakage-free rolling feature engineering from real time-series sports data
- Model selection on Brier score (calibration matters more than raw accuracy)
- SHAP-based interpretability with feature ablation to quantify each group's contribution
- COVID data handling as a first-class problem, not an afterthought
- Era drift analysis - empirically determining how far back training data improves vs. hurts generalization
- Quantified limitation analysis - systematic error on games with absent star players

## Architecture

```
Kaggle NBA Games dataset (self-contained, 19 seasons)
   |
   v
Data Ingestion (src/data/ingest.py)
  - Aggregate player-level box scores to team level
  - Build game index from games.csv (regular season filter via GAME_ID)
   |
   v
Feature Engineering (src/data/features.py)
  - Leakage-free rolling stats (.shift(1) discipline)
  - Derived advanced metrics: OffRtg, DefRtg, Pace, TS%
  - 29 features: recent form, efficiency, rest, standing, streak, COVID context
   |
   v
MLflow Experiment Tracking (src/models/train.py)
  - Logistic Regression, Random Forest, XGBoost (Optuna-tuned), PyTorch MLP
  - Full sklearn Pipeline artifacts (preprocessor + model, no training-serving skew)
   |
   v
FastAPI
  /predict/upcoming     - live schedule + predictions (6hr TTL cache via nba_api)
  /predict/game/{id}
  /teams/{team_id}/stats
  /analysis/feature-importance
  /analysis/home-advantage
   |
   v
React Dashboard
```

## Data

**Source:** [NBA Games dataset by Nathan Lauga](https://www.kaggle.com/datasets/nathanlauga/nba-games) (Kaggle). Fully self-contained - no API calls needed for the training pipeline.

**Why Kaggle instead of nba_api for historical data:** stats.nba.com aggressively rate-limits sustained bulk requests. The Kaggle dataset covers 19 complete seasons with no rate limiting, and the player-level box scores allow deriving advanced metrics (OffRtg, DefRtg, Pace) from raw counts rather than making ~11,000 per-game API calls. `nba_api` is used only in the FastAPI layer for live upcoming schedule queries.

**Coverage:** 2003-04 through 2021-22 - 19 seasons, 22,796 regular season games.

**Chronological split:**
- Train: 2003-04 through 2017-18 (15 seasons, ~18,200 rows)
- Val: 2018-19, 2019-20 pre-bubble (~2,200 rows)
- Test: 2020-21, 2021-22 (~2,300 rows)

Training window start is confirmed via rolling window drift analysis rather than hardcoded.

**COVID handling:**
- 2019-20 bubble games (July-Oct 2020): excluded from training. Home/away designations are meaningless at a neutral site. Used separately to empirically measure the home advantage effect.
- 2020-21 season (home arenas, no fans): included with `is_no_fans_season = 1` feature. The SHAP contribution of this flag quantifies the crowd effect on home advantage.

## Features

All 29 features are computed from data available before tip-off. No game-day stats.

| Group | Features |
|---|---|
| Recent form | Win% over last 5, 10, 20 games (rolling, per team) |
| Offensive efficiency | OffRtg rolling avg (last 10 games) - derived from raw counts |
| Defensive efficiency | DefRtg rolling avg (last 10 games) - derived from raw counts |
| Pace / shooting | Pace and TS% rolling avg (last 10 games) - derived from raw counts |
| Rest | Days since last game, back-to-back flag |
| Home/away splits | Team's home win% vs away win% this season |
| Standing | Current season win% |
| Streak | Signed integer (+N win, -N losing) |
| Context | `is_early_season`, `is_no_fans_season` |

OffRtg, DefRtg, Pace, and TS% are derived during feature engineering from raw counting stats (FGA, FTA, OREB, TOV) rather than fetched from a separate API endpoint.

**Deliberately excluded:**
- Net Rating: derived from OffRtg - DefRtg, causes multicollinearity in LR
- H2H record: 2-4 games per season is noise
- Injury data: pre-game availability is unreliable (documented as primary limitation)

## Results

**Selection criterion:** val Brier score. Calibration matters more than raw accuracy - a well-calibrated 65% model is more useful than a 67% overconfident one.

Baseline Brier score (always predict the mean home win rate): ~0.245

| Model | val Brier | test Brier | val AUC-ROC |
|---|---|---|---|
| **XGBoost (Optuna-tuned)** | **0.2145** | **0.2286** | **0.7035** |
| MLP (PyTorch) | 0.2162 | 0.2264 | 0.6964 |
| Logistic Regression | 0.2159 | 0.2278 | 0.6986 |
| Random Forest | 0.2187 | 0.2308 | 0.6891 |

XGBoost selected as the primary model (best val Brier, best val AUC). MLP edges it on test Brier by 0.002 - within noise, and val is the selection criterion.

**XGBoost best hyperparameters (50 Optuna trials):**
- `n_estimators=708`, `max_depth=3`, `learning_rate=0.0102`
- `subsample=0.632`, `colsample_bytree=0.798`, `min_child_weight=15`
- `reg_alpha=1.53`, `reg_lambda=3.88`

The shallow depth (3) and heavy regularization reflect the signal-to-noise characteristics of sports prediction. `max_depth=3` with 708 trees is the classic "many weak learners" pattern emerging naturally from the search.

## Analytical Findings

Results from `notebooks/04_insights.ipynb`:

1. **Era drift** - rolling window training reveals whether 2003-2010 data helps or hurts predictions on 2018+ games, empirically justifying the training window start
2. **Home advantage post-COVID** - season-by-season home win% with COVID bubble and no-fans seasons annotated; pre-2010 rate ~59-61%, post-2018 ~54-55%
3. **The no-fans effect** - SHAP contribution of `is_no_fans_season`: what does removing the crowd actually do to home advantage?
4. **Back-to-back penalty** - conditional win probability by B2B scenario, controlling for rolling team quality
5. **Feature importance via SHAP** - not built-in importance; directional effects showing which of the 29 features actually drives predictions
6. **Where the model fails** - error analysis on high-uncertainty predictions; primary structural gap is injury data

## Setup

```bash
pip install -r requirements.txt
```

### 1. Data ingestion

Download the [NBA Games dataset by Nathan Lauga](https://www.kaggle.com/datasets/nathanlauga/nba-games) from Kaggle. Place `games_details.csv`, `games.csv`, and `teams.csv` in `data/`.

```bash
python -m src.data.ingest
```

Produces `data/raw/team_gamelogs/` (19 season files) and `data/processed/game_list.csv` (22,767 games, bubble flags resolved).

### 2. Feature engineering

```bash
python -m src.data.features
```

Produces `data/processed/features.csv` - 22,767 rows x 34 columns, leakage-verified.

```bash
python -m src.data.features --verify-only   # re-run leakage check on existing file
```

### 3. Model training

```bash
python -m src.models.train                  # train all 4 models
python -m src.models.train --tune-xgb       # run Optuna (50 trials) before training XGBoost
python -m src.models.train --model xgb      # train a single model
```

All runs tracked in MLflow. View at `http://localhost:5000` after:

```bash
mlflow ui
```

### 4. Inference

```bash
python -m src.models.predict --best         # load best model by val Brier, predict on test set
python -m src.models.predict --run-id <id>  # load a specific MLflow run
```

## Project Status

- [x] Phase 1 - Data Ingestion
- [x] Phase 2 - Feature Engineering
- [x] Phase 3 - Model Training + Evaluation
- [ ] Phase 4 - FastAPI
- [ ] Phase 5 - React Dashboard
- [ ] Phase 6 - Deploy
