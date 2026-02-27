"""
NBA Predictor API - Team stats endpoint.

GET /teams/{team_id}/stats  Most recent rolling stats for a team.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_state
from api.routers.predict import _extract_team_features
from api.schemas import TeamStats

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/{team_id}/stats", response_model=TeamStats)
async def team_stats(
    team_id: int,
    state: AppState = Depends(get_state),
) -> TeamStats:
    """Return the most recent rolling stats for a team."""
    # Join features with game_list to get team IDs per row, then filter by team_id
    gl = state.game_list_df[["GAME_ID", "HOME_TEAM_ID", "HOME_TEAM_ABBR", "AWAY_TEAM_ID", "AWAY_TEAM_ABBR"]]
    merged = state.features_df.merge(gl, on="GAME_ID", how="left")

    team_rows = merged[
        (merged["HOME_TEAM_ID"] == team_id) | (merged["AWAY_TEAM_ID"] == team_id)
    ]
    if team_rows.empty:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    latest = team_rows.sort_values("GAME_DATE").iloc[-1]
    row = latest.to_dict()

    is_home = int(row.get("HOME_TEAM_ID", 0)) == team_id
    prefix = "home_" if is_home else "away_"
    abbr_col = "HOME_TEAM_ABBR" if is_home else "AWAY_TEAM_ABBR"

    return TeamStats(
        team_id=team_id,
        team_abbr=str(row[abbr_col]),
        last_game_date=str(row["GAME_DATE"]),
        season=str(row["SEASON"]),
        features=_extract_team_features(row, prefix),
    )
