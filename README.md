# NBA Match Outcome Predictor

Binary classification of NBA home team win/loss. The emphasis is on DS and ML rigor: leakage-free feature engineering, calibrated probability outputs, SHAP interpretability, and honest evaluation. FastAPI + React are secondary layers.

## What this project demonstrates

- Leakage-free rolling feature engineering from real time-series sports data
- Model selection on Brier score (calibration matters more than raw accuracy)
- SHAP-based interpretability with feature ablation to quantify each group's contribution
- COVID data handling as a first-class problem, not an afterthought
- Quantified limitation analysis - systematic error on games with absent star players

## Architecture

```
nba_api
   |
   v
Data Ingestion (cached CSVs, rate-limited, ~3hr wall-clock)
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
  /predict/upcoming     - live schedule + predictions (6hr TTL cache)
  /predict/game/{id}
  /teams/{team_id}/stats
  /analysis/feature-importance
  /analysis/home-advantage
   |
   v
React Dashboard
```

## Data

9 seasons: 2015-16 through 2023-24. ~11,000 regular season games.

**Chronological split:** train 2015-2021, validate 2022, test 2023-2024.

**COVID handling:**
- 2019-20 bubble games (July-Oct 2020): excluded from training. Home/away designations are meaningless at a neutral site. Used separately to empirically measure the home advantage effect.
- 2020-21 season (home arenas, no fans): included with `is_no_fans_season = 1` feature. The SHAP contribution of this flag quantifies the crowd effect on home advantage.

## Features

All features are computed from data available before tip-off. No game-day stats.

| Group | Features |
|---|---|
| Recent form | Win% over last 5, 10, 20 games (rolling, per team) |
| Offensive efficiency | OffRtg rolling avg (last 10 games) |
| Defensive efficiency | DefRtg rolling avg (last 10 games) |
| Pace | Possessions per game (last 10) |
| Rest | Days since last game, back-to-back flag |
| Home/away splits | Team's home win% vs away win% this season |
| Standing | Current season win% |
| Streak | Signed integer (+N win, -N losing) |
| Context | `is_early_season`, `is_no_fans_season` |

**Deliberately excluded:**
- Net Rating: derived from OffRtg - DefRtg, causes multicollinearity in LR
- H2H record: 2-4 games per season is noise
- Injury data: pre-game availability is unreliable from nba_api (documented as primary limitation)

## Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline; examine coefficients |
| Random Forest | Ensemble baseline; Gini importance |
| XGBoost | Primary candidate; Optuna-tuned |
| PyTorch MLP | Breadth; expected to underperform XGBoost on this data size |

**Selection criterion:** Brier score. A well-calibrated 65% model is more useful than a 67% overconfident one.

## Analytical Findings

1. **Home advantage post-COVID** - season-by-season home win% from 2015-2024 with COVID seasons annotated
2. **The no-fans effect** - SHAP contribution of `is_no_fans_season`: what does removing the crowd actually do to home advantage?
3. **Back-to-back penalty** - how much does B2B status drop predicted win probability, controlling for team quality?
4. **Feature importance via SHAP** - not built-in importance; directional effects and interaction terms
5. **Model drift** - does a 2015-trained model degrade on 2024 games?
6. **Where the model fails** - error analysis on games with absent star players (20+ PPG)

## Setup

```bash
pip install -r requirements.txt

# Re-run data ingestion (2-3hr for full 9 seasons due to per-game API calls)
python -m src.data.ingest --seasons all

# Or skip the slow boxscore step to get the game list only
python -m src.data.ingest --seasons all --skip-boxscores
```

Raw data is gitignored (large, regeneratable). The ingestion script is checkpointed - safe to interrupt and resume.

## Project Status

- [x] Phase 1 - Data Ingestion
- [ ] Phase 2 - Feature Engineering
- [ ] Phase 3 - Model Training + Evaluation
- [ ] Phase 4 - FastAPI
- [ ] Phase 5 - React Dashboard
- [ ] Phase 6 - Deploy
