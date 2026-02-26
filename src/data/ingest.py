"""
NBA data ingestion: fetches raw data from nba_api and caches to CSV.

Usage:
    python -m src.data.ingest --seasons all
    python -m src.data.ingest --seasons 2022-23
    python -m src.data.ingest --skip-boxscores
    python -m src.data.ingest --build-game-list-only
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]

BUBBLE_START = "2020-07-30"

SLEEP_INTERVAL = 0.6   # seconds between API calls
MAX_RETRIES = 3
BACKOFF_FACTOR = 2     # sleeps: 0.6s, 1.2s, 2.4s on successive failures

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

GAMES_DIR = RAW_DIR / "games"
TEAM_GAMELOGS_DIR = RAW_DIR / "team_gamelogs"
BOXSCORE_ADV_DIR = RAW_DIR / "boxscore_advanced"

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
    for d in (GAMES_DIR, TEAM_GAMELOGS_DIR, BOXSCORE_ADV_DIR, PROCESSED_DIR):
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
    Columns: TEAM_ID, GAME_ID, GAME_DATE, FG_PCT, FG3_PCT, REB, AST, TOV, STL, BLK, PTS, PLUS_MINUS
    """
    cache_path = TEAM_GAMELOGS_DIR / f"team_gamelogs_{season}.csv"
    if cache_path.exists():
        log.info("team_gamelogs %s: cache hit, skipping", season)
        return pd.read_csv(cache_path, dtype={"GAME_ID": str})

    from nba_api.stats.endpoints import TeamGameLogs

    log.info("team_gamelogs %s: fetching...", season)

    def _fetch():
        logs = TeamGameLogs(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        return logs.get_data_frames()[0]

    df = _retry_call(_fetch)
    time.sleep(SLEEP_INTERVAL)

    keep = [
        "TEAM_ID", "GAME_ID", "GAME_DATE",
        "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "STL", "BLK", "PTS", "PLUS_MINUS",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    df.to_csv(cache_path, index=False)
    log.info("team_gamelogs %s: saved %d rows -> %s", season, len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Fetch: BoxScoreAdvancedV2 (slow - one call per game)
# ---------------------------------------------------------------------------

def fetch_boxscore_advanced(season: str, game_ids: list[str]) -> pd.DataFrame:
    """
    Fetch team-level advanced box scores for every game in game_ids.
    Appends to cache CSV after each successful fetch (checkpointing).
    Returns the full season dataframe (previously cached + newly fetched).
    Columns: GAME_ID, TEAM_ID, OFF_RATING, DEF_RATING, PACE, TS_PCT, AST_PCT, REB_PCT
    """
    from nba_api.stats.endpoints import BoxScoreAdvancedV3

    # V3 uses camelCase columns; map to the uppercase names used downstream
    V3_RENAME = {
        "gameId": "GAME_ID",
        "teamId": "TEAM_ID",
        "offensiveRating": "OFF_RATING",
        "defensiveRating": "DEF_RATING",
        "pace": "PACE",
        "trueShootingPercentage": "TS_PCT",
        "assistPercentage": "AST_PCT",
        "reboundPercentage": "REB_PCT",
    }

    cache_path = BOXSCORE_ADV_DIR / f"boxscore_advanced_{season}.csv"

    # Load existing progress
    fetched_ids: set[str] = set()
    if cache_path.exists():
        existing = pd.read_csv(cache_path, dtype={"GAME_ID": str})
        fetched_ids = set(existing["GAME_ID"].unique())
        log.info(
            "boxscore_advanced %s: resuming - %d/%d games already cached",
            season, len(fetched_ids), len(game_ids),
        )

    remaining = [g for g in game_ids if g not in fetched_ids]
    if not remaining:
        log.info("boxscore_advanced %s: all games cached, skipping", season)
        return pd.read_csv(cache_path, dtype={"GAME_ID": str})

    log.info("boxscore_advanced %s: fetching %d games...", season, len(remaining))

    write_header = not cache_path.exists()

    for game_id in tqdm(remaining, desc=f"boxscore_adv {season}", unit="game"):
        def _fetch(gid=game_id):
            bs = BoxScoreAdvancedV3(game_id=gid)
            team_df = bs.team_stats.get_data_frame()
            return team_df

        try:
            team_df = _retry_call(_fetch, game_id=game_id)
        except Exception:
            log.error("Skipping game_id=%s after all retries exhausted", game_id)
            time.sleep(SLEEP_INTERVAL)
            continue

        row = team_df[list(V3_RENAME.keys())].rename(columns=V3_RENAME).copy()

        # Append to CSV immediately (checkpointing)
        row.to_csv(cache_path, mode="a", header=write_header, index=False)
        write_header = False

        time.sleep(SLEEP_INTERVAL)

    return pd.read_csv(cache_path, dtype={"GAME_ID": str})


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
        "--skip-boxscores",
        action="store_true",
        help="Skip the slow BoxScoreAdvancedV2 step",
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

    # Step 1: LeagueGameFinder (fast)
    for season in seasons:
        fetch_games(season)

    # Step 2: TeamGameLogs (fast)
    for season in seasons:
        fetch_team_gamelogs(season)

    # Step 3: BoxScoreAdvancedV2 (slow)
    if not args.skip_boxscores:
        for season in seasons:
            games_path = GAMES_DIR / f"games_{season}.csv"
            if not games_path.exists():
                log.warning("No games file for %s, skipping boxscores", season)
                continue
            games_df = pd.read_csv(games_path, dtype={"GAME_ID": str})
            # Deduplicate: each game appears twice (one row per team)
            game_ids = games_df["GAME_ID"].unique().tolist()
            fetch_boxscore_advanced(season, game_ids)
    else:
        log.info("--skip-boxscores set, skipping BoxScoreAdvancedV2")

    # Step 4: Build game_list.csv from all fetched seasons
    # Only build if all requested seasons are available
    available = [s for s in seasons if (GAMES_DIR / f"games_{s}.csv").exists()]
    if set(available) == set(seasons):
        build_game_list()
    else:
        log.warning(
            "Some seasons missing from cache, skipping game_list build. "
            "Run --build-game-list-only when all seasons are fetched."
        )


if __name__ == "__main__":
    main()
