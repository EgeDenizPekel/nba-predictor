"""
NBA data ingestion: fetches raw data from nba_api and caches to CSV.

Usage:
    python -m src.data.ingest --seasons all
    python -m src.data.ingest --seasons 2022-23
    python -m src.data.ingest --build-game-list-only

Advanced stats (OffRtg, DefRtg, Pace, TS%) are derived in Phase 2 feature
engineering from the raw columns saved here (FGA, FTA, OREB, etc.), avoiding
the ~11,000 per-game BoxScoreAdvanced API calls entirely.
"""

import argparse
import logging
import time
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
    # 2022-23 and 2023-24 have incomplete Kaggle coverage; fetch via nba_api when unblocked
]

# Kaggle dataset covers 2003-04 through 2021-22 completely.
# Start year (integer) -> season string mapping for Kaggle SEASON column.
KAGGLE_SEASON_MAP = {
    2003: "2003-04", 2004: "2004-05", 2005: "2005-06", 2006: "2006-07",
    2007: "2007-08", 2008: "2008-09", 2009: "2009-10", 2010: "2010-11",
    2011: "2011-12", 2012: "2012-13", 2013: "2013-14", 2014: "2014-15",
    2015: "2015-16", 2016: "2016-17", 2017: "2017-18", 2018: "2018-19",
    2019: "2019-20", 2020: "2020-21", 2021: "2021-22",
}

BUBBLE_START = "2020-07-30"

SLEEP_INTERVAL = 1.0  # seconds between API calls
MAX_RETRIES = 3
BACKOFF_FACTOR = 2    # sleeps: 1.0s, 2.0s, 4.0s on successive failures
API_TIMEOUT = 300     # seconds - bulk season calls can be slow to respond

# Headers that stats.nba.com expects
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://stats.nba.com/",
    "Origin": "https://www.nba.com",
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

GAMES_DIR = RAW_DIR / "games"
TEAM_GAMELOGS_DIR = RAW_DIR / "team_gamelogs"

# Columns required in team_gamelogs CSVs. If an existing cache is missing
# these (e.g. fetched before this column list was expanded), it is re-fetched.
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
    for d in (GAMES_DIR, TEAM_GAMELOGS_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _retry_call(fn, *args, game_id: str = "", **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on failure."""
    sleep = SLEEP_INTERVAL
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                log.error("FAILED after %d attempts (game_id=%s): %s", MAX_RETRIES, game_id, exc)
                raise
            log.warning(
                "Attempt %d/%d failed (game_id=%s): %s - retrying in %.1fs",
                attempt, MAX_RETRIES, game_id, exc, sleep,
            )
            time.sleep(sleep)
            sleep *= BACKOFF_FACTOR


# ---------------------------------------------------------------------------
# Fetch: LeagueGameFinder
# ---------------------------------------------------------------------------

def fetch_games(season: str) -> pd.DataFrame:
    """
    Fetch all regular-season games for a season via LeagueGameFinder.
    Returns two rows per game (one per team).
    Columns: GAME_ID, GAME_DATE, MATCHUP, TEAM_ID, TEAM_ABBREVIATION, WL, PTS
    """
    cache_path = GAMES_DIR / f"games_{season}.csv"
    if cache_path.exists():
        log.info("games %s: cache hit, skipping", season)
        return pd.read_csv(cache_path, dtype={"GAME_ID": str})

    from nba_api.stats.endpoints import LeagueGameFinder

    log.info("games %s: fetching...", season)

    def _fetch():
        finder = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",         # NBA
            season_type_nullable="Regular Season",
            headers=NBA_HEADERS,
            timeout=API_TIMEOUT,
        )
        return finder.get_data_frames()[0]

    df = _retry_call(_fetch)
    time.sleep(SLEEP_INTERVAL)

    keep = ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION", "WL", "PTS"]
    df = df[keep].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    df.to_csv(cache_path, index=False)
    log.info("games %s: saved %d rows -> %s", season, len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Fetch: TeamGameLogs
# ---------------------------------------------------------------------------

def fetch_team_gamelogs(season: str) -> pd.DataFrame:
    """
    Fetch team game logs for a season via TeamGameLogs.

    Saves raw counting stats needed to derive OffRtg, DefRtg, Pace, and TS%
    during Phase 2 feature engineering, avoiding per-game API calls.

    Columns: TEAM_ID, GAME_ID, GAME_DATE, MIN,
             FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
             OREB, DREB, REB, AST, TOV, STL, BLK, PTS, PLUS_MINUS
    """
    cache_path = TEAM_GAMELOGS_DIR / f"team_gamelogs_{season}.csv"
    if cache_path.exists():
        existing = pd.read_csv(cache_path, dtype={"GAME_ID": str})
        if TEAM_GAMELOGS_REQUIRED_COLS.issubset(existing.columns):
            log.info("team_gamelogs %s: cache hit, skipping", season)
            return existing
        log.info("team_gamelogs %s: cache missing columns, re-fetching", season)
        cache_path.unlink()

    from nba_api.stats.endpoints import TeamGameLogs

    log.info("team_gamelogs %s: fetching...", season)

    def _fetch():
        logs = TeamGameLogs(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            headers=NBA_HEADERS,
            timeout=API_TIMEOUT,
        )
        return logs.get_data_frames()[0]

    df = _retry_call(_fetch)
    time.sleep(SLEEP_INTERVAL)

    keep = [
        "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB",
        "AST", "TOV", "STL", "BLK", "PTS", "PLUS_MINUS",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    df.to_csv(cache_path, index=False)
    log.info("team_gamelogs %s: saved %d rows -> %s", season, len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Kaggle adapter: build team_gamelogs from Nathan Lauga's NBA Games dataset
# ---------------------------------------------------------------------------

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


def build_team_gamelogs_from_kaggle(
    details_path: str = "data/games_details.csv",
) -> None:
    """
    Build team_gamelogs_{season}.csv files from Kaggle's NBA Games dataset
    (Nathan Lauga) instead of the nba_api TeamGameLogs endpoint.

    Aggregates player-level box scores to team level and joins GAME_DATE from
    our cached LeagueGameFinder CSVs, which serve as the authoritative list of
    regular-season game IDs (filtering out playoffs automatically).

    Skips seasons where Kaggle coverage is below 95% of expected game count.

    Kaggle dataset: https://www.kaggle.com/datasets/nathanlauga/nba-games
    Expected file:  data/games_details.csv
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

    # Pad to 10-digit format to match our LeagueGameFinder GAME_IDs
    agg["GAME_ID"] = agg["GAME_ID"].astype(str).str.zfill(10)

    log.info("Aggregated to %d team-game rows", len(agg))

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

        games_csv = GAMES_DIR / f"games_{season}.csv"
        if not games_csv.exists():
            log.warning("team_gamelogs %s: no games CSV found, skipping", season)
            continue

        our_games = pd.read_csv(games_csv, dtype={"GAME_ID": str})
        season_ids = set(our_games["GAME_ID"].unique())
        game_dates = our_games[["GAME_ID", "GAME_DATE"]].drop_duplicates()

        season_data = agg[agg["GAME_ID"].isin(season_ids)].copy()

        coverage = season_data["GAME_ID"].nunique()
        expected = len(season_ids)
        pct = coverage / expected * 100

        if coverage == 0:
            log.warning("team_gamelogs %s: no Kaggle data found, skipping", season)
            continue
        if pct < 95:
            log.warning(
                "team_gamelogs %s: partial Kaggle coverage %.0f%% (%d/%d games), skipping",
                season, pct, coverage, expected,
            )
            continue

        season_data = season_data.merge(game_dates, on="GAME_ID", how="left")
        season_data = season_data[[c for c in col_order if c in season_data.columns]]
        season_data = season_data.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        season_data.to_csv(cache_path, index=False)
        log.info(
            "team_gamelogs %s: saved %d rows (%.0f%% coverage) -> %s",
            season, len(season_data), pct, cache_path,
        )


# ---------------------------------------------------------------------------
# Build game_list.csv
# ---------------------------------------------------------------------------

def build_game_list() -> pd.DataFrame:
    """
    Load all season CSVs from data/raw/games/, pivot from two-rows-per-game
    to one-row-per-game, add bubble/no-fans flags, save to data/processed/game_list.csv.
    Prints a data quality report.
    """
    season_dfs = []
    for season in SEASONS:
        path = GAMES_DIR / f"games_{season}.csv"
        if not path.exists():
            log.warning("Missing games file for %s, skipping", season)
            continue
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["SEASON"] = season
        season_dfs.append(df)

    if not season_dfs:
        raise FileNotFoundError("No games CSVs found. Run ingestion first.")

    all_games = pd.concat(season_dfs, ignore_index=True)

    # Separate home and away rows based on MATCHUP pattern
    # "BOS vs. LAL" -> BOS is home
    # "BOS @ LAL"   -> BOS is away
    home_mask = all_games["MATCHUP"].str.contains(r"\bvs\.", regex=True)
    away_mask = all_games["MATCHUP"].str.contains(r"\s@\s", regex=True)

    home = all_games[home_mask].copy()
    away = all_games[away_mask].copy()

    home = home.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM_ABBR",
        "WL": "HOME_WL",
        "PTS": "HOME_PTS",
    })
    away = away.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM_ABBR",
        "WL": "AWAY_WL",
        "PTS": "AWAY_PTS",
    })

    home = home[["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID", "HOME_TEAM_ABBR", "HOME_WL", "HOME_PTS"]]
    away = away[["GAME_ID", "AWAY_TEAM_ID", "AWAY_TEAM_ABBR", "AWAY_WL", "AWAY_PTS"]]

    game_list = home.merge(away, on="GAME_ID", how="inner")

    game_list["HOME_WIN"] = (game_list["HOME_WL"] == "W").astype(int)
    game_list = game_list.drop(columns=["HOME_WL", "AWAY_WL"])

    # Bubble and no-fans flags
    game_list["is_bubble_game"] = (
        (game_list["SEASON"] == "2019-20") &
        (game_list["GAME_DATE"] >= BUBBLE_START)
    ).astype(int)

    game_list["is_no_fans_season"] = (game_list["SEASON"] == "2020-21").astype(int)

    # Sort chronologically
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
    season_counts = df.groupby("SEASON").size()
    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")

    print("\nHOME_WIN rate per season (expected ~58-60% for non-bubble):")
    home_win_rate = df.groupby("SEASON")["HOME_WIN"].mean()
    for season, rate in home_win_rate.items():
        print(f"  {season}: {rate:.1%}")

    bubble_count = df["is_bubble_game"].sum()
    no_fans_count = df["is_no_fans_season"].sum()
    print(f"\nis_bubble_game count: {bubble_count} (expected ~88)")
    print(f"is_no_fans_season count: {no_fans_count} (expected ~1,080)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA data ingestion pipeline")
    parser.add_argument(
        "--seasons",
        default="all",
        help='Comma-separated seasons (e.g. "2022-23,2023-24") or "all"',
    )
    parser.add_argument(
        "--from-kaggle",
        action="store_true",
        help=(
            "Build team_gamelogs from local Kaggle CSV (data/games_details.csv) "
            "instead of fetching from nba_api. Games CSVs are still fetched via API."
        ),
    )
    parser.add_argument(
        "--build-game-list-only",
        action="store_true",
        help="Re-build game_list.csv from cached CSVs without re-fetching",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dirs()

    if args.build_game_list_only:
        build_game_list()
        return

    if args.seasons == "all":
        seasons = SEASONS
    else:
        seasons = [s.strip() for s in args.seasons.split(",")]
        invalid = [s for s in seasons if s not in SEASONS]
        if invalid:
            raise ValueError(f"Unknown seasons: {invalid}. Valid: {SEASONS}")

    log.info("Processing seasons: %s", seasons)

    # Step 1: LeagueGameFinder - fetch games CSVs for all seasons
    # These are the authoritative game ID lists; Kaggle path still needs them.
    for season in seasons:
        try:
            fetch_games(season)
        except Exception as exc:
            log.error("games %s: failed - %s", season, exc)

    # Step 2: Team game logs
    if args.from_kaggle:
        build_team_gamelogs_from_kaggle()
    else:
        failed_seasons = []
        for season in seasons:
            try:
                fetch_team_gamelogs(season)
            except Exception as exc:
                log.error("team_gamelogs %s: failed - %s", season, exc)
                failed_seasons.append(season)
        if failed_seasons:
            log.warning(
                "Could not fetch team_gamelogs for: %s. "
                "Likely rate-limited. Re-run with --from-kaggle or wait and retry.",
                failed_seasons,
            )

    # Step 3: Build game_list.csv from all seasons that have a games CSV
    build_game_list()


if __name__ == "__main__":
    main()
