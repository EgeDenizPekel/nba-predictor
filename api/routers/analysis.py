"""
NBA Predictor API - Analysis endpoints.

GET /analysis/feature-importance  SHAP-based feature ranking (pre-computed at startup).
GET /analysis/home-advantage       Season-by-season home win% with COVID annotations.
GET /games/seasons                 Available seasons + date ranges for the UI date picker.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import AppState, get_state
from api.schemas import CalibrationItem, FeatureImportanceItem, HomeAdvantageItem, SeasonInfo

router = APIRouter(tags=["analysis"])


@router.get("/analysis/feature-importance", response_model=list[FeatureImportanceItem])
async def feature_importance(
    state: AppState = Depends(get_state),
) -> list[dict]:
    """Return SHAP-based feature importance rankings, pre-computed at startup."""
    return state.shap_importance


@router.get("/analysis/home-advantage", response_model=list[HomeAdvantageItem])
async def home_advantage(
    state: AppState = Depends(get_state),
) -> list[dict]:
    """Return season-by-season home win rates with bubble/no-fans annotations."""
    return state.home_advantage


@router.get("/analysis/calibration", response_model=list[CalibrationItem])
async def calibration(
    state: AppState = Depends(get_state),
) -> list[dict]:
    """Return model calibration stats bucketed by confidence level (test set: 2020-21 + 2021-22)."""
    return state.calibration


@router.get("/games/seasons", response_model=list[SeasonInfo])
async def seasons(
    state: AppState = Depends(get_state),
) -> list[SeasonInfo]:
    """Return available seasons with date ranges and game counts for the UI date picker."""
    grouped = (
        state.features_df.groupby("SEASON")["GAME_DATE"]
        .agg(min_date="min", max_date="max", n_games="count")
        .reset_index()
        .sort_values("SEASON")
    )
    return [
        SeasonInfo(
            season=str(row["SEASON"]),
            min_date=str(row["min_date"]),
            max_date=str(row["max_date"]),
            n_games=int(row["n_games"]),
        )
        for _, row in grouped.iterrows()
    ]
