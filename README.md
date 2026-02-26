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
Feature Engineering (leakage-free rolling stats, .shift(1) discipline)
   |
   v
MLflow Experiment Tracking
  - Logistic Regression, Random Forest, XGBoost (Optuna-tuned), PyTorch MLP
  - Full sklearn Pipeline artifacts (preprocessor + model)
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

**Why Kaggle instead of nba_api:** The initial design used `nba_api` (stats.nba.com) for bulk historical data ingestion. In practice, stats.nba.com aggressively rate-limits and times out on sustained bulk requests - after several hours of retries and circuit breaker logic, the endpoint remained blocked. The Kaggle dataset turned out to be a better fit anyway: it covers 19 complete seasons with no rate limiting, and the player-level box scores allow us to derive advanced metrics (OffRtg, DefRtg, Pace) from raw counts rather than making ~11,000 per-game API calls. `nba_api` is still used in the FastAPI layer for live upcoming game schedule queries, where single-call latency is acceptable.

**Coverage:** 2003-04 through 2021-22 - 19 seasons, 22,796 regular season games.

**Chronological split:** determined empirically via model drift analysis. Training window start year is chosen based on where older data stops improving generalization on recent seasons. Validate and test sets use the most recent complete seasons.

**COVID handling:**
- 2019-20 bubble games (July-Oct 2020): excluded from training. Home/away designations are meaningless at a neutral site. Used separately to empirically measure the home advantage effect.
- 2020-21 season (home arenas, no fans): included with `is_no_fans_season = 1` feature. The SHAP contribution of this flag quantifies the crowd effect on home advantage.

## Features

All features are computed from data available before tip-off. No game-day stats.

| Group | Features |
|---|---|
| Recent form | Win% over last 5, 10, 20 games (rolling, per team) |
| Offensive efficiency | OffRtg rolling avg (last 10 games) - derived from raw counts |
| Defensive efficiency | DefRtg rolling avg (last 10 games) - derived from raw counts |
| Pace | Possessions per game (last 10) - derived from raw counts |
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

## Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline; examine coefficients |
| Random Forest | Ensemble baseline; Gini importance |
| XGBoost | Primary candidate; Optuna-tuned |
| PyTorch MLP | Breadth; expected to underperform XGBoost on this data size |

**Selection criterion:** Brier score. A well-calibrated 65% model is more useful than a 67% overconfident one.

## Analytical Findings

1. **Era drift** - does including 2003-2010 data help or hurt predictions on 2018+ games? Rolling window training answers this empirically and justifies the training cutoff.
2. **Home advantage post-COVID** - season-by-season home win% across all seasons with COVID annotations
3. **The no-fans effect** - SHAP contribution of `is_no_fans_season`: what does removing the crowd actually do to home advantage?
4. **Back-to-back penalty** - how much does B2B status drop predicted win probability, controlling for team quality?
5. **Feature importance via SHAP** - not built-in importance; directional effects and interaction terms
6. **Where the model fails** - error analysis on games with absent star players (20+ PPG)

## Setup

```bash
pip install -r requirements.txt
```

### Data

All raw data is gitignored (large, regeneratable).

1. Download the [NBA Games dataset by Nathan Lauga](https://www.kaggle.com/datasets/nathanlauga/nba-games) from Kaggle
2. Place `games_details.csv`, `games.csv`, and `teams.csv` in `data/`
3. Run ingestion:
```bash
python -m src.data.ingest
```

Covers 2003-04 through 2021-22 (19 seasons, 22,796 games).

```bash
python -m src.data.ingest --build-game-list-only   # rebuild game_list.csv without re-aggregating
```

## Project Status

- [x] Phase 1 - Data Ingestion
- [ ] Phase 2 - Feature Engineering
- [ ] Phase 3 - Model Training + Evaluation
- [ ] Phase 4 - FastAPI
- [ ] Phase 5 - React Dashboard
- [ ] Phase 6 - Deploy
