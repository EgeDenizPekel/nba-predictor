"""
NBA Predictor API - Pydantic response schemas.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TeamFeatures(BaseModel):
    """14 rolling/context features for one team (home_ or away_ prefix stripped)."""

    win_pct_l5: Optional[float] = None
    win_pct_l10: Optional[float] = None
    win_pct_l20: Optional[float] = None
    off_rtg_l10: Optional[float] = None
    def_rtg_l10: Optional[float] = None
    pace_l10: Optional[float] = None
    ts_pct_l10: Optional[float] = None
    rest_days: Optional[float] = None
    is_b2b: int
    season_win_pct: Optional[float] = None
    home_win_pct_season: Optional[float] = None
    away_win_pct_season: Optional[float] = None
    streak: int
    is_early_season: int


class GameSummary(BaseModel):
    """One game with prediction + actual result. Used in list responses."""

    game_id: str
    game_date: str
    season: str
    home_team_id: int
    home_team_abbr: str
    away_team_id: int
    away_team_abbr: str
    home_pts: Optional[float] = None
    away_pts: Optional[float] = None
    home_win: int
    home_win_prob: float
    is_bubble_game: int
    is_no_fans_season: int


class GameDetail(GameSummary):
    """Single game with full feature breakdown for both teams."""

    home_features: TeamFeatures
    away_features: TeamFeatures


class TeamStats(BaseModel):
    """Most recent rolling stats for a team."""

    team_id: int
    team_abbr: str
    last_game_date: str
    season: str
    features: TeamFeatures


class FeatureImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float
    rank: int


class HomeAdvantageItem(BaseModel):
    season: str
    home_win_rate: float
    n_games: int
    is_bubble_season: bool
    is_no_fans_season: bool


class SeasonInfo(BaseModel):
    season: str
    min_date: str
    max_date: str
    n_games: int
