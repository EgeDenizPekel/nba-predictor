"""
NBA feature engineering: builds data/processed/features.csv from Phase 1 outputs.

Inputs:
    data/processed/game_list.csv          - canonical game index (22,796 rows)
    data/raw/team_gamelogs/*.csv          - raw counting stats per team per game

Output:
    data/processed/features.csv           - 22,796 rows x 32 columns, training-ready

Design decisions:
    - Rolling stats use CROSS-SEASON continuity (global sort by team+date).
      Form doesn't reset at season boundary; is_early_season flag discounts
      partial windows in the model.
    - Season standing features (season_win_pct, home/away splits) reset at
      season boundary by design.
    - min_periods=1 for all rolling windows: keeps early-season games in the
      dataset, is_early_season flag handles the partial-window discount.
    - Bubble games (is_bubble_game=1) are included here; exclusion from
      training is train.py's responsibility.
    - rest_days=NaN for each team's first game ever; imputed in train.py.
    - Leakage prevention: all features use .shift(1) so only data from
      games strictly BEFORE the current game is used.

Usage:
    python -m src.data.features               # build features.csv
    python -m src.data.features --verify-only # leakage check on existing file
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMELOGS_DIR = Path("data/raw/team_gamelogs")
PROCESSED_DIR = Path("data/processed")
GAME_LIST_PATH = PROCESSED_DIR / "game_list.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"

EARLY_SEASON_THRESHOLD = 15  # first N games of a season get is_early_season=1

# Rolling window sizes
L5, L10, L20 = 5, 10, 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all_gamelogs(gamelogs_dir: Path = GAMELOGS_DIR) -> pd.DataFrame:
    """Concatenate all 19 season team_gamelogs CSVs into a single DataFrame."""
    files = sorted(gamelogs_dir.glob("team_gamelogs_*.csv"))
    if not files:
        raise FileNotFoundError(f"No team_gamelogs CSV files found in {gamelogs_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"GAME_ID": str})
        df["GAME_ID"] = df["GAME_ID"].str.zfill(10)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d team-game rows from %d files", len(combined), len(files))
    return combined


def load_game_list(path: Path = GAME_LIST_PATH) -> pd.DataFrame:
    """
    Load the canonical game index produced by Phase 1.

    Drops duplicate GAME_IDs (29 identical rows in early 2020-21 from the
    Kaggle source - known data quality issue).
    """
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    df["GAME_ID"] = df["GAME_ID"].str.zfill(10)
    before = len(df)
    df = df.drop_duplicates("GAME_ID").reset_index(drop=True)
    if len(df) < before:
        log.warning("Dropped %d duplicate GAME_ID rows from game_list", before - len(df))
    log.info("Loaded game_list: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 1: add WIN and IS_HOME columns
# ---------------------------------------------------------------------------

def add_win_and_home_columns(gamelogs: pd.DataFrame, game_list: pd.DataFrame) -> pd.DataFrame:
    """
    Join HOME_WIN and HOME_TEAM_ID from game_list onto gamelogs, then derive:
        IS_HOME: 1 if this team was the home team, else 0
        WIN:     1 if this team won, else 0
    """
    meta = game_list[["GAME_ID", "HOME_TEAM_ID", "HOME_WIN", "SEASON"]].copy()
    df = gamelogs.merge(meta, on="GAME_ID", how="inner")

    df["IS_HOME"] = (df["TEAM_ID"] == df["HOME_TEAM_ID"]).astype(int)
    df["WIN"] = df["IS_HOME"] * df["HOME_WIN"] + (1 - df["IS_HOME"]) * (1 - df["HOME_WIN"])
    df["WIN"] = df["WIN"].astype(int)

    log.info("After join: %d rows (%d unique games x 2 teams)", len(df), len(df) // 2)
    return df


# ---------------------------------------------------------------------------
# Step 2: per-game derived metrics
# ---------------------------------------------------------------------------

def compute_per_game_metrics(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Derive OffRtg, DefRtg, Pace, TS% from raw counting stats.

    Formulas (per CLAUDE.md):
        POSS    = FGA + 0.44 * FTA - OREB + TOV
        OFF_RTG = 100 * PTS / POSS          (NaN if POSS == 0)
        TS_PCT  = PTS / (2*(FGA + 0.44*FTA)) (NaN if denominator == 0)
        DEF_RTG = opponent's OFF_RTG for the same game
        PACE    = 48 * (POSS + OPP_POSS) / (2 * MIN / 5)
    """
    df = gamelogs.copy()

    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]

    ts_denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["OFF_RTG"] = np.where(df["POSS"] > 0, 100 * df["PTS"] / df["POSS"], np.nan)
    df["TS_PCT"] = np.where(ts_denom > 0, df["PTS"] / ts_denom, np.nan)

    # Self-join to get opponent stats in same game
    opp = df[["GAME_ID", "TEAM_ID", "POSS", "OFF_RTG", "MIN"]].rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "POSS": "OPP_POSS",
        "OFF_RTG": "DEF_RTG",   # opponent's offense = this team's defense allowed
        "MIN": "OPP_MIN",
    })
    df = df.merge(
        opp,
        on="GAME_ID",
        how="left",
    )
    # Drop the self-join row (team vs itself)
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()

    # Pace: 48 * (Poss + Opp_Poss) / (2 * MIN / 5)
    # MIN here is total player minutes (sum across all players, ~240 for regulation)
    # MIN/5 approximates game length in minutes (5 players on court)
    pace_denom = 2 * df["MIN"] / 5
    df["PACE"] = np.where(
        pace_denom > 0,
        48 * (df["POSS"] + df["OPP_POSS"]) / pace_denom,
        np.nan,
    )

    log.info("Per-game metrics computed: OFF_RTG, DEF_RTG, PACE, TS_PCT")
    return df


# ---------------------------------------------------------------------------
# Step 3: rolling features (cross-season continuity)
# ---------------------------------------------------------------------------

def _signed_streak(shifted_wins: pd.Series) -> pd.Series:
    """
    Compute signed win/loss streak from a pre-shifted WIN series.

    +N = N-game win streak going into current game
    -N = N-game losing streak going into current game
    """
    streaks = []
    current = 0
    for w in shifted_wins:
        if pd.isna(w):
            current = 0
        elif w == 1:
            current = max(1, current + 1)
        else:
            current = min(-1, current - 1)
        streaks.append(current)
    return pd.Series(streaks, index=shifted_wins.index)


def compute_rolling_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window features. Sort globally by (TEAM_ID, GAME_DATE) so
    rolling windows span season boundaries - form carries from prior season.

    All features are shifted by 1 (use only data from games before this one).
    """
    df = gamelogs.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()

    def _rolling_mean(series, window):
        return series.shift(1).rolling(window, min_periods=1).mean()

    g = df.groupby("TEAM_ID", sort=False)

    df["win_pct_l5"]    = g["WIN"].transform(lambda x: _rolling_mean(x, L5))
    df["win_pct_l10"]   = g["WIN"].transform(lambda x: _rolling_mean(x, L10))
    df["win_pct_l20"]   = g["WIN"].transform(lambda x: _rolling_mean(x, L20))
    df["off_rtg_l10"]   = g["OFF_RTG"].transform(lambda x: _rolling_mean(x, L10))
    df["def_rtg_l10"]   = g["DEF_RTG"].transform(lambda x: _rolling_mean(x, L10))
    df["pace_l10"]      = g["PACE"].transform(lambda x: _rolling_mean(x, L10))
    df["ts_pct_l10"]    = g["TS_PCT"].transform(lambda x: _rolling_mean(x, L10))

    # Streak: custom function, applied to pre-shifted WIN
    df["streak"] = g["WIN"].transform(lambda x: _signed_streak(x.shift(1)))

    log.info("Rolling features computed: win_pct l5/l10/l20, off/def_rtg, pace, ts_pct, streak")
    return df


# ---------------------------------------------------------------------------
# Step 4: rest features
# ---------------------------------------------------------------------------

def compute_rest_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest_days (days since previous game) and is_b2b (rest_days == 1).

    rest_days = NaN for each team's first game ever (imputed in train.py).
    """
    df = gamelogs.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    df["GAME_DATE_dt"] = pd.to_datetime(df["GAME_DATE"])

    df["rest_days"] = (
        df.groupby("TEAM_ID")["GAME_DATE_dt"]
        .transform(lambda x: x.diff().dt.days)
    )
    df["is_b2b"] = (df["rest_days"] == 1).astype(int)

    log.info("Rest features computed: rest_days, is_b2b")
    return df


# ---------------------------------------------------------------------------
# Step 5: season-reset features
# ---------------------------------------------------------------------------

def compute_season_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that reset at season boundaries:
        season_win_pct        - cumulative win% this season (shifted)
        home_win_pct_season   - cumulative home win% this season (shifted)
        away_win_pct_season   - cumulative away win% this season (shifted)
        is_early_season       - 1 for first 15 games of the season

    Uses transform throughout (not apply) to avoid multi-index alignment issues.
    Home/away splits mask the "wrong" side as NaN; expanding().mean() skips NaN,
    giving the correct per-venue cumulative win rate.
    """
    df = gamelogs.sort_values(["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"]).copy()

    g_season = df.groupby(["TEAM_ID", "SEASON"], sort=False)

    # season_win_pct: expanding mean of shifted WIN within (team, season)
    df["season_win_pct"] = g_season["WIN"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # is_early_season: first 15 games per (team, season)
    df["is_early_season"] = (g_season.cumcount() < EARLY_SEASON_THRESHOLD).astype(int)

    # home_win_pct_season: expanding home win% this season, shifted
    # Non-home games are NaN in the masked column; expanding().mean() skips them
    df["_home_win"] = df["WIN"].where(df["IS_HOME"] == 1)
    df["_away_win"] = df["WIN"].where(df["IS_HOME"] == 0)

    df["home_win_pct_season"] = g_season["_home_win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["away_win_pct_season"] = g_season["_away_win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    df = df.drop(columns=["_home_win", "_away_win"])

    log.info("Season features computed: season_win_pct, home/away_win_pct_season, is_early_season")
    return df


# ---------------------------------------------------------------------------
# Step 6: join onto game_list
# ---------------------------------------------------------------------------

def join_onto_game_list(game_list: pd.DataFrame, gamelogs_featured: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot team-level features into game-level rows by merging twice:
        1. home team features -> prefixed 'home_'
        2. away team features -> prefixed 'away_'
    """
    feature_cols = [
        "win_pct_l5", "win_pct_l10", "win_pct_l20",
        "off_rtg_l10", "def_rtg_l10",
        "pace_l10", "ts_pct_l10",
        "rest_days", "is_b2b",
        "season_win_pct",
        "home_win_pct_season", "away_win_pct_season",
        "streak",
        "is_early_season",
    ]

    team_features = gamelogs_featured[["GAME_ID", "TEAM_ID"] + feature_cols].copy()

    # Home side
    home_features = team_features.rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    df = game_list.merge(
        home_features.rename(columns={"TEAM_ID": "HOME_TEAM_ID"}),
        on=["GAME_ID", "HOME_TEAM_ID"],
        how="left",
    )

    # Away side
    away_features = team_features.rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )
    df = df.merge(
        away_features.rename(columns={"TEAM_ID": "AWAY_TEAM_ID"}),
        on=["GAME_ID", "AWAY_TEAM_ID"],
        how="left",
    )

    # Final column order per plan schema
    meta_cols = ["GAME_ID", "GAME_DATE", "SEASON", "HOME_WIN", "is_bubble_game", "is_no_fans_season"]
    home_cols = [f"home_{c}" for c in feature_cols]
    away_cols = [f"away_{c}" for c in feature_cols]
    df = df[meta_cols + home_cols + away_cols]

    log.info("Joined features onto game_list: %d rows x %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_leakage(
    features_df: pd.DataFrame,
    game_list: pd.DataFrame,
    n_games: int = 5,
) -> None:
    """
    Spot-check leakage prevention for home_win_pct_l10.

    For n_games mid-season games, manually compute win_pct_l10 from game_list
    (last 10 games strictly before GAME_DATE for the home team) and compare
    against features_df within 1e-6 tolerance.
    """
    print("\n" + "=" * 60)
    print("LEAKAGE VERIFICATION - home_win_pct_l10")
    print("=" * 60)

    candidates = features_df[
        features_df["home_win_pct_l10"].notna() &
        (features_df["home_is_early_season"] == 0)
    ].copy()

    if len(candidates) < n_games:
        print("WARN: not enough non-early-season games to verify")
        return

    sample = candidates.sample(n=n_games, random_state=42)

    # Build team-level win series from the canonical game_list
    gl = game_list[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_WIN"]].copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])

    home_rows = gl[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "HOME_WIN"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID", "HOME_WIN": "WIN"}
    )
    away_rows = gl[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID", "HOME_WIN"]].rename(
        columns={"AWAY_TEAM_ID": "TEAM_ID", "HOME_WIN": "WIN"}
    )
    away_rows["WIN"] = 1 - away_rows["WIN"]
    all_wins = pd.concat([home_rows, away_rows], ignore_index=True)
    all_wins = all_wins.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])

    # Build GAME_ID -> HOME_TEAM_ID lookup
    game_to_home = gl.set_index("GAME_ID")["HOME_TEAM_ID"].to_dict()

    all_pass = True
    for _, row in sample.iterrows():
        game_date = pd.to_datetime(row["GAME_DATE"])
        home_team_id = game_to_home.get(row["GAME_ID"])
        if home_team_id is None:
            print(f"  SKIP  GAME_ID={row['GAME_ID']} - not found in game_list")
            continue

        prior_wins = all_wins[
            (all_wins["TEAM_ID"] == home_team_id) &
            (all_wins["GAME_DATE"] < game_date)
        ]["WIN"].tail(10)

        expected = prior_wins.mean() if len(prior_wins) > 0 else np.nan
        actual = row["home_win_pct_l10"]

        match = (
            (pd.isna(expected) and pd.isna(actual)) or
            (not pd.isna(expected) and not pd.isna(actual) and abs(actual - expected) < 1e-6)
        )
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(
            f"  {status}  GAME_ID={row['GAME_ID']}  DATE={row['GAME_DATE']}  "
            f"expected={expected:.4f}  actual={actual:.4f}"
        )

    print(f"\n{'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def _print_quality_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT - features.csv")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"\nNull counts (non-zero only):\n{nulls.to_string()}")
    else:
        print("\nNull counts: none (all columns complete)")

    for col in ["home_win_pct_l10", "home_rest_days", "home_streak",
                "home_off_rtg_l10", "home_ts_pct_l10"]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"\n{col}: min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    gamelogs_dir: Path = GAMELOGS_DIR,
    game_list_path: Path = GAME_LIST_PATH,
    output_path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Produces features.csv.

    Steps:
        1. Load all team_gamelogs and game_list
        2. Add WIN / IS_HOME columns
        3. Compute per-game metrics (OffRtg, DefRtg, Pace, TS%)
        4. Compute rolling features (cross-season)
        5. Compute rest features
        6. Compute season-reset features
        7. Join home/away onto game_list
        8. Save and report
    """
    log.info("=== Phase 2: Feature Engineering ===")

    game_list = load_game_list(game_list_path)
    gamelogs = load_all_gamelogs(gamelogs_dir)

    log.info("Step 1: Adding WIN and IS_HOME columns ...")
    gamelogs = add_win_and_home_columns(gamelogs, game_list)

    log.info("Step 2: Computing per-game metrics ...")
    gamelogs = compute_per_game_metrics(gamelogs)

    log.info("Step 3: Computing rolling features ...")
    gamelogs = compute_rolling_features(gamelogs)

    log.info("Step 4: Computing rest features ...")
    gamelogs = compute_rest_features(gamelogs)

    log.info("Step 5: Computing season features ...")
    gamelogs = compute_season_features(gamelogs)

    log.info("Step 6: Joining features onto game_list ...")
    features = join_onto_game_list(game_list, gamelogs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    log.info("features.csv saved -> %s", output_path)

    _print_quality_report(features)
    verify_no_leakage(features, game_list)

    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA feature engineering pipeline"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run leakage check on existing features.csv without rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verify_only:
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"{FEATURES_PATH} not found; run without --verify-only first")
        features = pd.read_csv(FEATURES_PATH, dtype={"GAME_ID": str})
        game_list = load_game_list()
        verify_no_leakage(features, game_list)
        return

    build_features()


if __name__ == "__main__":
    main()
