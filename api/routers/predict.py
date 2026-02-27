"""
NBA Predictor API - Prediction endpoints.

GET /predict/games?date=YYYY-MM-DD  All games on a date with predictions + actuals.
GET /predict/game/{game_id}         Single game with full feature breakdown.
"""

from __future__ import annotations

import math
from datetime import date as date_type

import numpy as np
import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_state
from api.schemas import GameDetail, GameSummary, TeamFeatures
from src.models.predict import predict_proba
from src.models.train import FEATURE_COLS

router = APIRouter(prefix="/predict", tags=["predict"])

# Columns from game_list_df that features_df doesn't have
_GAME_LIST_COLS = [
    "GAME_ID",
    "HOME_TEAM_ID",
    "HOME_TEAM_ABBR",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_ABBR",
    "HOME_PTS",
    "AWAY_PTS",
]


def _nan_to_none(val) -> float | None:
    """Convert NaN / numpy scalars to Python float or None."""
    if val is None:
        return None
    try:
        fval = float(val)
        return None if math.isnan(fval) else fval
    except (TypeError, ValueError):
        return None


def _extract_team_features(row: dict, prefix: str) -> TeamFeatures:
    """
    Build a TeamFeatures instance from a flat row dict by stripping the given prefix.

    prefix should be "home_" or "away_".
    """

    def f(key: str) -> float | None:
        return _nan_to_none(row.get(f"{prefix}{key}"))

    def i(key: str, default: int = 0) -> int:
        val = row.get(f"{prefix}{key}")
        if val is None:
            return default
        try:
            fval = float(val)
            return default if math.isnan(fval) else int(fval)
        except (TypeError, ValueError):
            return default

    return TeamFeatures(
        win_pct_l5=f("win_pct_l5"),
        win_pct_l10=f("win_pct_l10"),
        win_pct_l20=f("win_pct_l20"),
        off_rtg_l10=f("off_rtg_l10"),
        def_rtg_l10=f("def_rtg_l10"),
        pace_l10=f("pace_l10"),
        ts_pct_l10=f("ts_pct_l10"),
        rest_days=f("rest_days"),
        is_b2b=i("is_b2b"),
        season_win_pct=f("season_win_pct"),
        home_win_pct_season=f("home_win_pct_season"),
        away_win_pct_season=f("away_win_pct_season"),
        streak=i("streak"),
        is_early_season=i("is_early_season"),
    )


def _compute_game_shap(pipeline, feat_row) -> dict[str, float]:
    """Compute per-feature SHAP values for a single game row using native XGBoost."""
    preproc = pipeline[:-1]
    X_transformed = preproc.transform(feat_row[FEATURE_COLS])
    booster = pipeline.named_steps["model"].get_booster()
    dmatrix = xgb.DMatrix(X_transformed)
    # pred_contribs: (1, n_features + 1) - last col is bias term
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    return {f: float(v) for f, v in zip(FEATURE_COLS, shap_matrix[0, :-1])}


@router.get("/games", response_model=list[GameSummary])
async def predict_games(
    date: date_type,
    state: AppState = Depends(get_state),
) -> list[GameSummary]:
    """Return all games on a date with model predictions and actual results."""
    date_str = date.isoformat()

    gl_today = state.game_list_df[state.game_list_df["GAME_DATE"] == date_str]
    if gl_today.empty:
        return []

    feat_today = state.features_df[
        state.features_df["GAME_ID"].isin(gl_today["GAME_ID"])
    ]
    if feat_today.empty:
        return []

    probs = predict_proba(state.pipeline, feat_today)

    # Merge team metadata (not present in features_df) from game_list_df
    merged = feat_today.merge(
        gl_today[_GAME_LIST_COLS],
        on="GAME_ID",
        how="left",
    )

    results: list[GameSummary] = []
    for (_, row), prob in zip(merged.iterrows(), probs):
        results.append(
            GameSummary(
                game_id=str(row["GAME_ID"]),
                game_date=str(row["GAME_DATE"]),
                season=str(row["SEASON"]),
                home_team_id=int(row["HOME_TEAM_ID"]),
                home_team_abbr=str(row["HOME_TEAM_ABBR"]),
                away_team_id=int(row["AWAY_TEAM_ID"]),
                away_team_abbr=str(row["AWAY_TEAM_ABBR"]),
                home_pts=_nan_to_none(row.get("HOME_PTS")),
                away_pts=_nan_to_none(row.get("AWAY_PTS")),
                home_win=int(row["HOME_WIN"]),
                home_win_prob=float(prob),
                is_bubble_game=int(row["is_bubble_game"]),
                is_no_fans_season=int(row["is_no_fans_season"]),
            )
        )
    return results


@router.get("/game/{game_id}", response_model=GameDetail)
async def predict_game(
    game_id: str,
    state: AppState = Depends(get_state),
) -> GameDetail:
    """Return a single game with prediction and full feature breakdown."""
    feat_row = state.features_df[state.features_df["GAME_ID"] == game_id]
    if feat_row.empty:
        raise HTTPException(status_code=404, detail=f"Game {game_id!r} not found")

    gl_row = state.game_list_df[state.game_list_df["GAME_ID"] == game_id]
    if gl_row.empty:
        raise HTTPException(status_code=404, detail=f"Game {game_id!r} not found in game list")

    prob = float(predict_proba(state.pipeline, feat_row)[0])
    shap_values = _compute_game_shap(state.pipeline, feat_row)
    # Merge both rows into a single dict; features_df values override on key collision
    row = {**gl_row.iloc[0].to_dict(), **feat_row.iloc[0].to_dict()}

    return GameDetail(
        game_id=game_id,
        game_date=str(row["GAME_DATE"]),
        season=str(row["SEASON"]),
        home_team_id=int(row["HOME_TEAM_ID"]),
        home_team_abbr=str(row["HOME_TEAM_ABBR"]),
        away_team_id=int(row["AWAY_TEAM_ID"]),
        away_team_abbr=str(row["AWAY_TEAM_ABBR"]),
        home_pts=_nan_to_none(row.get("HOME_PTS")),
        away_pts=_nan_to_none(row.get("AWAY_PTS")),
        home_win=int(row["HOME_WIN"]),
        home_win_prob=prob,
        is_bubble_game=int(row.get("is_bubble_game", 0)),
        is_no_fans_season=int(row.get("is_no_fans_season", 0)),
        home_features=_extract_team_features(row, "home_"),
        away_features=_extract_team_features(row, "away_"),
        shap_values=shap_values,
    )
