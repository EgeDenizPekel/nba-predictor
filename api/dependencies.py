"""
NBA Predictor API - Shared dependencies.

AppState holds all data loaded at startup. get_state() is a FastAPI
dependency that retrieves it from app.state on each request.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from fastapi import Request
from sklearn.pipeline import Pipeline


@dataclass
class AppState:
    features_df: pd.DataFrame
    game_list_df: pd.DataFrame
    pipeline: Pipeline
    shap_importance: list[dict]
    home_advantage: list[dict]


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
