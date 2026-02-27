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
  /predict/games?date=YYYY-MM-DD  - all games on a date with predictions + actuals
  /predict/game/{game_id}         - single game with full feature breakdown
  /teams/{team_id}/stats          - most recent rolling stats for a team
  /analysis/feature-importance    - SHAP-based feature ranking
  /analysis/home-advantage        - season-by-season home win% with COVID annotations
  /games/seasons                  - available seasons + date ranges (for UI date picker)
   |
   v
React Dashboard
```

## Data

**Source:** [NBA Games dataset by Nathan Lauga](https://www.kaggle.com/datasets/nathanlauga/nba-games) (Kaggle). Fully self-contained - no API calls needed for the training pipeline.

**Why Kaggle instead of nba_api:** stats.nba.com aggressively rate-limits sustained bulk requests. The Kaggle dataset covers 19 complete seasons with no rate limiting, and the player-level box scores allow deriving advanced metrics (OffRtg, DefRtg, Pace) from raw counts rather than making ~11,000 per-game API calls. The API serves historical predictions only - no nba_api dependency anywhere in the stack.

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

## Limitations

These are structural gaps in the model, not oversights. Each is documented here because understanding where a model fails is as important as knowing where it succeeds.

**Injury and availability data (primary gap)**
The model has no knowledge of who is playing. When a star player sits out - whether for load management, injury, or DNP-CD - the rolling team stats look the same as if they were playing. This is the largest single source of systematic error. Games where a top-5 player is unexpectedly absent show significantly higher prediction error. The dataset (Kaggle box scores) does not include pre-game availability reports, and building that would require a separate data source (e.g. NBA injury reports or Vegas line movement as a proxy).

**Late-season motivation**
Rolling win% doesn't distinguish between a team that is bad and a team that is intentionally losing to improve draft position (tanking), or resting starters ahead of the playoffs. Both look like poor recent form. The model will underestimate teams going through deliberate load management stretches.

**Roster discontinuity from trades**
A mid-season blockbuster trade changes team quality overnight, but the rolling windows (L5, L10, L20) take several games to reflect the new reality. In the week or two after a major trade the model's features are stale by construction.

**Early-season cold start**
The `is_early_season` flag covers the first 15 games, but rolling windows are still thin in games 1-5. Win% over the last 5 games with only 3 observations is noisy regardless of flagging. Teams with significant roster turnover (free agency, draft) are essentially unknown quantities until mid-November.

**AUC ceiling around 0.70**
NBA game outcomes have high inherent variance. Even with perfect pre-game information, single-game prediction is hard - a starter rolling an ankle in warmups, an officiating crew with unusual tendencies, or a cold-shooting night from a normally elite scorer are all unforeseeable. Professional sportsbooks with access to injury reports, line movement, and sharp money barely exceed 0.72 AUC over large samples. The gap between this model's 0.70 and the theoretical ceiling is narrow.

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

### 5. Frontend

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
npm run build   # production build to frontend/dist/
```

Requires the API server running on port 8000 (configure via `VITE_API_URL` in `frontend/.env`).

**Views:**
- **Games tab** - fixed left sidebar with a custom calendar (month + year dropdowns, 2003-10-28 to 2022-04-10) and a scrollable game list. Five filter pills: All, Correct, Wrong, Uncertain, and Big Misses (confident wrong - >70% predicted, incorrect). Wrong and Big Misses pills turn red when active. Cards are compact single-row: stacked abbreviations, thin low-contrast probability bar with a 50% midpoint marker, home win%, and an outcome icon. Border colors: green = confident correct, red = confident wrong, grey = low confidence. Clicking a game opens the detail panel.
- **Detail panel** - structured header with three stacked rows: Prediction (4xl probability + "P(Home Win)" label), Outcome (final score + winner), Evaluation (badge with hover tooltip showing what the model predicted at what confidence vs what actually happened - tooltip uses the predicted team's probability, not the raw home win probability). Two-column layout below: Left: 14 features in five groups with winner shading and hover tooltips; rows where home/away differ by less than 5% of the feature's typical range dim to show only the discriminative features. Right: local SHAP diverging bar chart (top 12 features, blue = pushes home win left, amber = pushes away win right, axis label "Values are additive contributions to the log-odds of home win"), and a Model Confidence Context panel (4 confidence buckets, active bucket highlighted, filled dot on the bar marking the current game's accuracy position).
- **No Spoiler mode** - toggle in the header navbar. Hides all outcome signals: card border colors go neutral, winner bolding removed, outcome icons hidden, Correct/Wrong/Big Misses filter pills hidden (only All and Uncertain remain), Outcome and Evaluation rows hidden in the detail panel. All pre-game data (features, SHAP, calibration) remains visible. Simulates using the model as a live forecasting tool.
- **Analysis tab** - three panels: SHAP feature importance (horizontal bar chart, color-coded by feature group, Top 10 / Show all toggle, defaulting to top 10); Home Advantage Trend (line chart 2003-04 to 2021-22, bubble/no-fans inline labels, pre-2010 and post-2018 era average reference lines); Calibration Reliability diagram (bars = observed accuracy per confidence bucket, dashed line = perfect calibration, n_games in tooltip).

### 6. API

```bash
uvicorn api.main:app --reload   # start dev server at http://localhost:8000
```

Startup loads features.csv, game_list.csv, and the best MLflow pipeline into memory (~1s). SHAP importance is pre-computed on 2000 test-season rows at startup.

```bash
# Smoke test endpoints
curl "http://localhost:8000/predict/games?date=2021-11-15"
curl "http://localhost:8000/predict/game/0022100199"
curl "http://localhost:8000/teams/1610612747/stats"
curl "http://localhost:8000/analysis/feature-importance"
curl "http://localhost:8000/analysis/home-advantage"
curl "http://localhost:8000/games/seasons"

# Interactive docs
open http://localhost:8000/docs

# Tests
pytest tests/test_api.py -v
```

## Project Status

- [x] Phase 1 - Data Ingestion
- [x] Phase 2 - Feature Engineering
- [x] Phase 3 - Model Training + Evaluation
- [x] Phase 4 - FastAPI
- [x] Phase 5 - React Dashboard
- [ ] Phase 6 - Deploy
