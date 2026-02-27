"""
NBA Predictor - Analysis Insights

Utility functions for home advantage trend analysis and B2B penalty analysis.
Called from notebook 04.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def home_advantage_trend(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute season-by-season home win rate with COVID annotations.

    Args:
        features_df: The full features DataFrame (including bubble games for context).

    Returns:
        DataFrame with columns:
            season, home_win_rate, n_games, is_bubble_season, is_no_fans_season
    """
    records = []
    for season, grp in features_df.groupby("SEASON"):
        is_bubble = bool(grp["is_bubble_game"].any())
        is_no_fans = bool((grp["is_no_fans_season"] == 1).any())

        # Exclude bubble games from win rate computation (per CLAUDE.md convention)
        non_bubble = grp[grp["is_bubble_game"] != 1]

        if len(non_bubble) == 0:
            continue

        records.append({
            "season": season,
            "home_win_rate": non_bubble["HOME_WIN"].mean(),
            "n_games": len(non_bubble),
            "is_bubble_season": is_bubble,
            "is_no_fans_season": is_no_fans,
        })

    return pd.DataFrame(records).sort_values("season").reset_index(drop=True)


def back_to_back_analysis(
    features_df: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Conditional win probability: home/away B2B vs not, controlling for context.

    Four scenarios:
      1. Neither team on B2B
      2. Home team on B2B only
      3. Away team on B2B only
      4. Both teams on B2B

    Args:
        features_df: Features DataFrame (bubble games will be excluded internally).
        pipeline: Fitted sklearn Pipeline that exposes predict_proba.

    Returns:
        DataFrame with columns:
            scenario, home_is_b2b, away_is_b2b, mean_predicted_prob, std_predicted_prob, n_games
    """
    from src.models.train import FEATURE_COLS

    df = features_df[features_df["is_bubble_game"] != 1].copy()

    scenarios = [
        ("Neither on B2B",       0, 0),
        ("Home only on B2B",     1, 0),
        ("Away only on B2B",     0, 1),
        ("Both on B2B",          1, 1),
    ]

    records = []
    for label, home_b2b, away_b2b in scenarios:
        mask = (df["home_is_b2b"] == home_b2b) & (df["away_is_b2b"] == away_b2b)
        subset = df[mask]

        if len(subset) == 0:
            continue

        # Align columns - only pass features that exist
        X = subset[[c for c in FEATURE_COLS if c in subset.columns]]
        probs = pipeline.predict_proba(X)[:, 1]

        records.append({
            "scenario": label,
            "home_is_b2b": home_b2b,
            "away_is_b2b": away_b2b,
            "mean_predicted_prob": float(np.mean(probs)),
            "std_predicted_prob": float(np.std(probs)),
            "n_games": len(subset),
        })

    return pd.DataFrame(records)
