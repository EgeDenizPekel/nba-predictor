"""
NBA Predictor API - Integration tests.

Uses FastAPI's TestClient (httpx-based, synchronous) which fires the lifespan,
loading the real model and data files. Tests verify HTTP contracts only.

Run: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Single TestClient for the module - lifespan runs once on enter."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /predict/games
# ---------------------------------------------------------------------------


def test_predict_games_valid_date(client):
    response = client.get("/predict/games", params={"date": "2021-11-15"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    first = data[0]
    assert "game_id" in first
    assert "game_date" in first
    assert "home_team_abbr" in first
    assert "away_team_abbr" in first
    assert "home_win_prob" in first
    assert 0.0 <= first["home_win_prob"] <= 1.0
    assert "home_win" in first
    assert first["home_win"] in (0, 1)


def test_predict_games_empty_date(client):
    # July is NBA off-season
    response = client.get("/predict/games", params={"date": "2021-07-01"})
    assert response.status_code == 200
    assert response.json() == []


def test_predict_games_bad_date(client):
    response = client.get("/predict/games", params={"date": "not-a-date"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict/game/{game_id}
# ---------------------------------------------------------------------------


def test_predict_game_valid_id(client):
    # Derive a valid game_id from a known date to avoid hardcoding
    r = client.get("/predict/games", params={"date": "2021-11-15"})
    games = r.json()
    assert len(games) > 0
    game_id = games[0]["game_id"]

    r2 = client.get(f"/predict/game/{game_id}")
    assert r2.status_code == 200
    data = r2.json()

    assert data["game_id"] == game_id
    assert "home_features" in data
    assert "away_features" in data

    for team_feat in (data["home_features"], data["away_features"]):
        assert "win_pct_l10" in team_feat
        assert "off_rtg_l10" in team_feat
        assert "streak" in team_feat
        assert "is_b2b" in team_feat


def test_predict_game_not_found(client):
    response = client.get("/predict/game/0000000000")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /teams/{team_id}/stats
# ---------------------------------------------------------------------------


def test_teams_stats_valid(client):
    # 1610612747 = Los Angeles Lakers
    response = client.get("/teams/1610612747/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["team_id"] == 1610612747
    assert "team_abbr" in data
    assert "last_game_date" in data
    assert "season" in data
    assert "features" in data
    assert "win_pct_l10" in data["features"]


def test_teams_stats_not_found(client):
    response = client.get("/teams/9999999999/stats")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /analysis/feature-importance
# ---------------------------------------------------------------------------


def test_feature_importance(client):
    response = client.get("/analysis/feature-importance")
    assert response.status_code == 200
    data = response.json()

    assert len(data) == 29  # one entry per feature in FEATURE_COLS
    assert all("feature" in item for item in data)
    assert all("mean_abs_shap" in item for item in data)
    assert all("rank" in item for item in data)

    ranks = [item["rank"] for item in data]
    assert sorted(ranks) == list(range(1, 30))


# ---------------------------------------------------------------------------
# /analysis/home-advantage
# ---------------------------------------------------------------------------


def test_home_advantage(client):
    response = client.get("/analysis/home-advantage")
    assert response.status_code == 200
    data = response.json()

    assert len(data) > 0
    for item in data:
        assert "season" in item
        assert "home_win_rate" in item
        assert "n_games" in item
        assert "is_bubble_season" in item
        assert "is_no_fans_season" in item
        assert 0.0 <= item["home_win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# /games/seasons
# ---------------------------------------------------------------------------


def test_seasons(client):
    response = client.get("/games/seasons")
    assert response.status_code == 200
    data = response.json()

    assert len(data) > 0
    for item in data:
        assert "season" in item
        assert "min_date" in item
        assert "max_date" in item
        assert "n_games" in item
        assert item["n_games"] > 0
        assert item["min_date"] <= item["max_date"]

    # Expect 19 seasons (2003-04 through 2021-22)
    assert len(data) == 19
