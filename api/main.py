"""
NBA Predictor API - FastAPI application entry point.

Lifespan loads all data once at startup (no per-request file I/O):
  - features.csv     (~22k rows, training-ready feature matrix)
  - game_list.csv    (game metadata: team IDs, abbreviations, scores)
  - ML pipeline      (best XGBoost model from MLflow)
  - SHAP importance  (pre-computed on 2000 test-season rows)
  - home advantage   (pre-computed season-by-season home win rates)

Start: uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Prevent dual-OpenMP segfault on macOS (XGBoost + PyTorch conflict)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import AppState
from api.routers import analysis as analysis_router_module
from api.routers import predict as predict_router_module
from api.routers import teams as teams_router_module
import xgboost as xgb

from src.analysis.insights import home_advantage_trend
from src.models.predict import load_best_pipeline, predict_proba
from src.models.train import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = _PROJECT_ROOT / "data" / "processed" / "features.csv"
GAME_LIST_PATH = _PROJECT_ROOT / "data" / "processed" / "game_list.csv"


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _build_game_list(path: Path) -> pd.DataFrame:
    """Load game_list.csv, deduplicate the 29 known Kaggle source dupes."""
    gl = pd.read_csv(path, dtype={"GAME_ID": str})
    before = len(gl)
    gl = gl.drop_duplicates(subset="GAME_ID")
    dropped = before - len(gl)
    if dropped:
        logger.warning(f"Dropped {dropped} duplicate GAME_IDs from game_list.csv")
    return gl


def _compute_shap_importance(pipeline, features_df: pd.DataFrame) -> list[dict]:
    """
    Compute mean absolute SHAP values on a 2000-row sample from the test seasons.

    Uses XGBoost's native pred_contribs=True instead of the shap library's
    TreeExplainer. This avoids a shap/xgboost version incompatibility where
    XGBoost 3.x serializes base_score as '[value]' (array string) which shap
    0.49.x fails to parse. pred_contribs produces identical SHAP values.
    """
    sample = features_df[
        features_df["SEASON"].isin({"2020-21", "2021-22"})
        & (features_df["is_bubble_game"] != 1)
    ].head(2000)

    logger.info(f"Computing SHAP importance on {len(sample)} rows...")

    # Apply preprocessing (SimpleImputer) but not the model step
    preproc = pipeline[:-1]
    X_transformed = preproc.transform(sample[FEATURE_COLS])

    booster = pipeline.named_steps["model"].get_booster()
    dmatrix = xgb.DMatrix(X_transformed)
    # pred_contribs returns (n_samples, n_features + 1); last col is the bias term
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    mean_abs = np.abs(shap_matrix[:, :-1]).mean(axis=0)

    ranked = sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: x[1], reverse=True)
    return [
        {"feature": f, "mean_abs_shap": float(v), "rank": i + 1}
        for i, (f, v) in enumerate(ranked)
    ]


def _compute_calibration(pipeline, features_df: pd.DataFrame) -> list[dict]:
    """
    Compute prediction accuracy bucketed by confidence level on the test set.

    Uses symmetric folding: confidence = max(prob, 1-prob), so each game
    is classified by how decisive the model was, regardless of direction.
    Accuracy is the fraction of games where the model's favored side won.
    """
    test = features_df[
        features_df["SEASON"].isin({"2020-21", "2021-22"})
        & (features_df["is_bubble_game"] != 1)
    ]
    probs = predict_proba(pipeline, test)
    y = test["HOME_WIN"].values

    confidence = np.maximum(probs, 1 - probs)
    # 1 if the side the model favored actually won
    predicted_correct = np.where(probs >= 0.5, y, 1 - y).astype(float)

    buckets = [
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 1.01, "80%+"),
    ]
    result = []
    for lo, hi, label in buckets:
        mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n > 0:
            result.append(
                {
                    "bucket": label,
                    "lo": lo,
                    "hi": hi,
                    "n_games": n,
                    "accuracy": float(predicted_correct[mask].mean()),
                }
            )
    return result


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading data and model...")

    features_df = pd.read_csv(FEATURES_PATH, dtype={"GAME_ID": str})
    logger.info(f"Loaded features.csv: {len(features_df)} rows")

    game_list_df = _build_game_list(GAME_LIST_PATH)
    logger.info(f"Loaded game_list.csv: {len(game_list_df)} rows")

    pipeline = load_best_pipeline()
    logger.info("Loaded best pipeline from MLflow")

    shap_importance = _compute_shap_importance(pipeline, features_df)
    logger.info(f"Computed SHAP importance for {len(shap_importance)} features")

    home_adv = home_advantage_trend(features_df).to_dict("records")
    logger.info(f"Computed home advantage trend: {len(home_adv)} seasons")

    calibration = _compute_calibration(pipeline, features_df)
    logger.info(f"Computed calibration: {len(calibration)} confidence buckets")

    app.state.app_state = AppState(
        features_df=features_df,
        game_list_df=game_list_df,
        pipeline=pipeline,
        shap_importance=shap_importance,
        home_advantage=home_adv,
        calibration=calibration,
    )

    logger.info("Startup complete.")
    yield
    logger.info("Shutdown.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="NBA Predictor API",
    description="Historical NBA game predictions (2003-04 through 2021-22). "
    "Browse by date to see model predictions alongside actual results.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to Vercel URL at deploy time
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router_module.router)
app.include_router(teams_router_module.router)
app.include_router(analysis_router_module.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
