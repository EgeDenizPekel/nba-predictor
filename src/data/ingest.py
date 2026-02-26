"""
NBA data ingestion: builds training data from Kaggle's NBA Games dataset.

All historical data (2003-04 through 2021-22) is sourced from Kaggle:
    https://www.kaggle.com/datasets/nathanlauga/nba-games

Required local files (place in data/):
    games_details.csv   - player-level box scores (aggregated to team level)
    games.csv           - game-level results (game index, scores, home/away)
    teams.csv           - team metadata (TEAM_ID -> ABBREVIATION mapping)

Usage:
    python -m src.data.ingest
    python -m src.data.ingest --build-game-list-only

Advanced stats (OffRtg, DefRtg, Pace, TS%) are derived in Phase 2 feature
engineering from raw columns saved here (FGA, FTA, OREB, etc.), avoiding
per-game API calls entirely.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASONS = [
    "2003-04", "2004-05", "2005-06", "2006-07", "2007-08",
    "2008-09", "2009-10", "2010-11", "2011-12", "2012-13",
    "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22",
]

# Kaggle SEASON column is the start year as an integer (e.g. 2021 -> "2021-22")
KAGGLE_SEASON_MAP = {
    2003: "2003-04", 2004: "2004-05", 2005: "2005-06", 2006: "2006-07",
    2007: "2007-08", 2008: "2008-09", 2009: "2009-10", 2010: "2010-11",
    2011: "2011-12", 2012: "2012-13", 2013: "2013-14", 2014: "2014-15",
    2015: "2015-16", 2016: "2016-17", 2017: "2017-18", 2018: "2018-19",
    2019: "2019-20", 2020: "2020-21", 2021: "2021-22",
}

BUBBLE_START = "2020-07-30"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEAM_GAMELOGS_DIR = RAW_DIR / "team_gamelogs"

# Columns required in cached team_gamelogs CSVs. Files missing these are
# regenerated (catches stale caches from earlier ingestion runs).
TEAM_GAMELOGS_REQUIRED_COLS = {"FGA", "FTA", "FGM", "FTM", "OREB", "MIN"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (TEAM_GAMELOGS_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _parse_min(value) -> float:
    """Convert player MIN from 'MM:SS' string to decimal minutes."""
    if pd.isna(value) or value == "":
        return 0.0
    try:
        s = str(value)
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(s)
    except Exception:
        return 0.0


def _load_kaggle_games(games_path: str) -> pd.DataFrame:
    """
    Load Kaggle's games.csv, filter to regular season only, normalize IDs and dates.

    Regular season is identified by the 3rd character of the zero-padded 10-digit
    GAME_ID being '2' (pre-season='1', regular='2', playoffs='4').
    """
    games = pd.read_csv(games_path)
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    regular = games[games["GAME_ID"].str[2] == "2"].copy()
    regular["SEASON"] = regular["SEASON"].map(KAGGLE_SEASON_MAP)
    regular = regular.dropna(subset=["SEASON"])
    regular["GAME_DATE"] = pd.to_datetime(regular["GAME_DATE_EST"]).dt.strftime("%Y-%m-%d")
    return regular


# ---------------------------------------------------------------------------
# Build: team_gamelogs_{season}.csv
# ---------------------------------------------------------------------------

def build_team_gamelogs_from_kaggle(
    details_path: str = "data/games_details.csv",
    games_path: str = "data/games.csv",
) -> None:
    """
    Build team_gamelogs_{season}.csv for all 19 seasons from Kaggle data.

    Aggregates player-level box scores (games_details.csv) to team level,
    then joins GAME_DATE from games.csv. Skips seasons where Kaggle coverage
    is below 95% of the expected game count.

    Output columns per file:
        TEAM_ID, GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MIN,
        FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
        FTM, FTA, FT_PCT, OREB, DREB, REB, AST, TOV, STL, BLK, PTS
    """
    log.info("Loading %s ...", details_path)
    details = pd.read_csv(details_path, low_memory=False)
    details["MIN_dec"] = details["MIN"].apply(_parse_min)

    log.info("Aggregating %d player rows to team level ...", len(details))
    agg = (
        details.groupby(["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"], as_index=False)
        .agg(
            MIN=("MIN_dec", "sum"),
            FGM=("FGM", "sum"),
            FGA=("FGA", "sum"),
            FG3M=("FG3M", "sum"),
            FG3A=("FG3A", "sum"),
            FTM=("FTM", "sum"),
            FTA=("FTA", "sum"),
            OREB=("OREB", "sum"),
            DREB=("DREB", "sum"),
            REB=("REB", "sum"),
            AST=("AST", "sum"),
            TOV=("TO", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            PTS=("PTS", "sum"),
        )
    )

    agg["FG_PCT"]  = (agg["FGM"]  / agg["FGA"] .replace(0, float("nan"))).round(3)
    agg["FG3_PCT"] = (agg["FG3M"] / agg["FG3A"].replace(0, float("nan"))).round(3)
    agg["FT_PCT"]  = (agg["FTM"]  / agg["FTA"] .replace(0, float("nan"))).round(3)
    agg["GAME_ID"] = agg["GAME_ID"].astype(str).str.zfill(10)

    log.info("Aggregated to %d team-game rows across all seasons", len(agg))

    log.info("Loading game index from %s ...", games_path)
    regular_games = _load_kaggle_games(games_path)
    game_meta = (
        regular_games[["GAME_ID", "GAME_DATE", "SEASON"]]
        .drop_duplicates("GAME_ID")
    )

    col_order = [
        "TEAM_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MIN",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "TOV", "STL", "BLK", "PTS",
    ]

    for season in SEASONS:
        cache_path = TEAM_GAMELOGS_DIR / f"team_gamelogs_{season}.csv"
        if cache_path.exists():
            existing = pd.read_csv(cache_path, dtype={"GAME_ID": str})
            if TEAM_GAMELOGS_REQUIRED_COLS.issubset(existing.columns):
                log.info("team_gamelogs %s: cache hit, skipping", season)
                continue
            log.info("team_gamelogs %s: cache missing required columns, regenerating", season)
            cache_path.unlink()

        season_meta = game_meta[game_meta["SEASON"] == season]
        if len(season_meta) == 0:
            log.warning("team_gamelogs %s: no games found in games.csv, skipping", season)
            continue

        season_ids = set(season_meta["GAME_ID"])
        season_data = agg[agg["GAME_ID"].isin(season_ids)].copy()

        coverage = season_data["GAME_ID"].nunique()
        expected = len(season_ids)
        pct = coverage / expected * 100

        if coverage == 0:
            log.warning("team_gamelogs %s: no data found in games_details.csv, skipping", season)
            continue
        if pct < 95:
            log.warning(
                "team_gamelogs %s: partial coverage %.0f%% (%d/%d games), skipping",
                season, pct, coverage, expected,
            )
            continue

        season_data = season_data.merge(
            season_meta[["GAME_ID", "GAME_DATE"]], on="GAME_ID", how="left"
        )
        season_data = season_data[[c for c in col_order if c in season_data.columns]]
        season_data = season_data.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        season_data.to_csv(cache_path, index=False)
        log.info(
            "team_gamelogs %s: saved %d rows (%.0f%% coverage) -> %s",
            season, len(season_data), pct, cache_path,
        )


# ---------------------------------------------------------------------------
# Build: game_list.csv
# ---------------------------------------------------------------------------

def build_game_list_from_kaggle(
    games_path: str = "data/games.csv",
    teams_path: str = "data/teams.csv",
) -> pd.DataFrame:
    """
    Build data/processed/game_list.csv from Kaggle's games.csv.

    games.csv is already one-row-per-game with home/away resolved. Team
    abbreviations are joined from teams.csv. Bubble and no-fans flags are
    derived from GAME_DATE and SEASON.

    Output columns:
        GAME_ID, GAME_DATE, SEASON, HOME_TEAM_ID, HOME_TEAM_ABBR,
        AWAY_TEAM_ID, AWAY_TEAM_ABBR, HOME_PTS, AWAY_PTS, HOME_WIN,
        is_bubble_game, is_no_fans_season
    """
    log.info("Loading game index from %s ...", games_path)
    regular_games = _load_kaggle_games(games_path)

    log.info("Loading team abbreviations from %s ...", teams_path)
    teams = pd.read_csv(teams_path)
    team_abbr = teams[["TEAM_ID", "ABBREVIATION"]].copy()

    game_list = regular_games[[
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "PTS_home", "PTS_away", "HOME_TEAM_WINS",
    ]].copy()

    game_list = game_list.rename(columns={
        "VISITOR_TEAM_ID": "AWAY_TEAM_ID",
        "PTS_home": "HOME_PTS",
        "PTS_away": "AWAY_PTS",
        "HOME_TEAM_WINS": "HOME_WIN",
    })

    game_list = game_list.merge(
        team_abbr.rename(columns={"ABBREVIATION": "HOME_TEAM_ABBR"}),
        left_on="HOME_TEAM_ID", right_on="TEAM_ID", how="left",
    ).drop(columns=["TEAM_ID"])

    game_list = game_list.merge(
        team_abbr.rename(columns={"ABBREVIATION": "AWAY_TEAM_ABBR"}),
        left_on="AWAY_TEAM_ID", right_on="TEAM_ID", how="left",
    ).drop(columns=["TEAM_ID"])

    game_list["is_bubble_game"] = (
        (game_list["SEASON"] == "2019-20") &
        (game_list["GAME_DATE"] >= BUBBLE_START)
    ).astype(int)

    game_list["is_no_fans_season"] = (game_list["SEASON"] == "2020-21").astype(int)

    col_order = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "HOME_TEAM_ABBR",
        "AWAY_TEAM_ID", "AWAY_TEAM_ABBR",
        "HOME_PTS", "AWAY_PTS", "HOME_WIN",
        "is_bubble_game", "is_no_fans_season",
    ]
    game_list = game_list[col_order]
    game_list = game_list.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    out_path = PROCESSED_DIR / "game_list.csv"
    game_list.to_csv(out_path, index=False)
    log.info("game_list saved: %d rows -> %s", len(game_list), out_path)

    _print_quality_report(game_list)
    return game_list


def _print_quality_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT - game_list.csv")
    print("=" * 60)

    print(f"\nTotal games: {len(df):,}")
    print(f"Null counts:\n{df.isnull().sum().to_string()}")

    print("\nGames per season:")
    for season, count in df.groupby("SEASON").size().items():
        print(f"  {season}: {count:,}")

    print("\nHOME_WIN rate per season (expected ~58-60% for non-bubble):")
    for season, rate in df.groupby("SEASON")["HOME_WIN"].mean().items():
        print(f"  {season}: {rate:.1%}")

    bubble_count = df["is_bubble_game"].sum()
    no_fans_count = df["is_no_fans_season"].sum()
    print(f"\nis_bubble_game count: {bubble_count} (expected ~88)")
    print(f"is_no_fans_season count: {no_fans_count} (full 2020-21 season)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA data ingestion pipeline (Kaggle source)"
    )
    parser.add_argument(
        "--build-game-list-only",
        action="store_true",
        help="Re-build game_list.csv from Kaggle games.csv without regenerating team_gamelogs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dirs()

    if args.build_game_list_only:
        build_game_list_from_kaggle()
        return

    build_team_gamelogs_from_kaggle()
    build_game_list_from_kaggle()


if __name__ == "__main__":
    main()
