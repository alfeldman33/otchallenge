"""
fetch_data.py
Pulls player data from MoneyPuck (historical xG/shots) and
the NHL API (current game boxscore).
"""

import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NHL_API = "https://api-web.nhle.com/v1"
MONEYPUCK_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"


# ---------------------------------------------------------------------------
# MoneyPuck
# ---------------------------------------------------------------------------

def fetch_moneypuck_skaters(season: int, game_type: str = "playoffs") -> pd.DataFrame:
    """
    Download MoneyPuck skater summary for a season.
    season: e.g. 2024 for the 2024-25 season
    game_type: 'regular' or 'playoffs'
    """
    cache = DATA_DIR / f"mp_{season}_{game_type}.csv"
    if cache.exists():
        print(f"  [cache] {cache.name}")
        return pd.read_csv(cache)

    url = f"{MONEYPUCK_BASE}/{season}/{game_type}/skaters.csv"
    print(f"  [download] {url}")
    df = pd.read_csv(url)
    df.to_csv(cache, index=False)
    return df


def load_moneypuck_history(seasons: list[int]) -> pd.DataFrame:
    """
    Load and concat several seasons of playoff + regular-season data.
    Adds 'season' and 'game_type' columns.
    Returns a single DataFrame with all skaters.
    """
    frames = []
    for season in seasons:
        for gtype in ("playoffs", "regular"):
            try:
                df = fetch_moneypuck_skaters(season, gtype)
                df["season"] = season
                df["game_type"] = gtype
                frames.append(df)
            except Exception as e:
                print(f"  [warn] Could not fetch {season} {gtype}: {e}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# NHL API helpers
# ---------------------------------------------------------------------------

def _get(path: str) -> dict:
    resp = requests.get(f"{NHL_API}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_todays_playoff_games() -> list[dict]:
    """Return list of today's playoff games from the NHL schedule."""
    from datetime import date
    today = date.today().isoformat()
    data = _get(f"/schedule/{today}")
    games = []
    for day in data.get("gameWeek", []):
        for game in day.get("games", []):
            # gameType 3 = playoffs
            if game.get("gameType") == 3:
                games.append(game)
    return games


def fetch_boxscore(game_id: int) -> dict:
    """Full boxscore for a game."""
    return _get(f"/gamecenter/{game_id}/boxscore")


def fetch_play_by_play(game_id: int) -> dict:
    """Full play-by-play for a game."""
    return _get(f"/gamecenter/{game_id}/play-by-play")


def parse_game_skater_stats(boxscore: dict) -> pd.DataFrame:
    """
    Extract per-skater in-game stats from a boxscore response.
    Returns a DataFrame with columns:
        playerId, name, teamAbbrev, toi, shots, goals, assists, plusMinus
    """
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = boxscore.get(side, {})
        abbrev = team.get("abbrev", "")
        for skater in team.get("skaters", []):
            rows.append({
                "playerId": skater.get("playerId"),
                "name": f"{skater.get('firstName', {}).get('default', '')} "
                        f"{skater.get('lastName', {}).get('default', '')}".strip(),
                "teamAbbrev": abbrev,
                "position": skater.get("position", ""),
                "toi": _parse_toi(skater.get("toi", "0:00")),
                "shots": skater.get("sog", 0),
                "goals": skater.get("goals", 0),
                "assists": skater.get("assists", 0),
                "plusMinus": skater.get("plusMinus", 0),
            })
    return pd.DataFrame(rows)


def _parse_toi(toi_str: str) -> float:
    """Convert 'MM:SS' string to total minutes (float)."""
    try:
        parts = toi_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return 0.0


def list_games_for_team(team_abbrev: str, season: int = 2025) -> list[dict]:
    """
    Return the current-season playoff schedule for a team.
    Useful for pulling the game_id of the most recent game.
    season: the end year, e.g. 2025 for 2024-25.
    """
    season_str = f"{season - 1}{season}"
    data = _get(f"/club-schedule-season/{team_abbrev}/{season_str}")
    return [g for g in data.get("games", []) if g.get("gameType") == 3]


# ---------------------------------------------------------------------------
# Series context
# ---------------------------------------------------------------------------

def get_series_game_ids(game_id: int) -> tuple[list[int], int]:
    """
    Parse an NHL playoff game ID to return prior game IDs in the same series.

    NHL playoff game ID format: YYYYTTSSSg
      YYYY = season start year (4 digits)
      TT   = game type, 03 for playoffs (2 digits)
      SSS  = series code, encodes round + matchup (3 digits)
      g    = game number within series, 1-7 (1 digit)

    Returns (list_of_prior_game_ids, current_game_number).
    """
    s = str(game_id)
    series_prefix = s[:9]       # YYYYTTSSS
    current_game = int(s[9])    # 1-7
    prior_ids = [int(series_prefix + str(g)) for g in range(1, current_game)]
    return prior_ids, current_game


def fetch_series_skater_stats(game_id: int) -> pd.DataFrame:
    """
    Aggregate per-skater stats across all prior games in the current series.
    Returns a DataFrame with shots/goals per game in this series, or empty
    DataFrame if this is game 1.
    """
    prior_ids, _ = get_series_game_ids(game_id)
    if not prior_ids:
        return pd.DataFrame()

    frames = []
    for gid in prior_ids:
        try:
            bs = fetch_boxscore(gid)
            df = parse_game_skater_stats(bs)
            frames.append(df)
        except Exception as e:
            print(f"  [warn] Could not fetch series game {gid}: {e}")

    if not frames:
        return pd.DataFrame()

    all_games = pd.concat(frames, ignore_index=True)
    agg = (
        all_games.groupby(["playerId", "name", "teamAbbrev", "position"], as_index=False)
        .agg(
            series_games=("shots", "count"),
            series_shots=("shots", "sum"),
            series_goals=("goals", "sum"),
            series_toi=("toi", "sum"),
        )
    )
    agg["shots_per_game_series"] = agg["series_shots"] / agg["series_games"].clip(lower=1)
    agg["goals_per_game_series"] = agg["series_goals"] / agg["series_games"].clip(lower=1)
    agg["playerId"] = agg["playerId"].astype(str)
    return agg


# ---------------------------------------------------------------------------
# Goalie stats
# ---------------------------------------------------------------------------

def parse_goalie_stats(boxscore: dict) -> pd.DataFrame:
    """
    Extract in-game goalie stats from a boxscore.
    Returns one row per goalie with: teamAbbrev, name, toi, shots_against,
    saves, goals_against, in_game_sv_pct.
    Only includes goalies who actually played (toi > 0).
    """
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = boxscore.get(side, {})
        abbrev = team.get("abbrev", "")
        for goalie in team.get("goalies", []):
            toi = _parse_toi(goalie.get("toi", "0:00"))
            if toi == 0:
                continue
            shots_against = goalie.get("shotsAgainst", 0)
            saves = goalie.get("saves", 0)
            rows.append({
                "teamAbbrev": abbrev,
                "playerId": goalie.get("playerId"),
                "name": f"{goalie.get('firstName', {}).get('default', '')} "
                        f"{goalie.get('lastName', {}).get('default', '')}".strip(),
                "toi": toi,
                "shots_against": shots_against,
                "saves": saves,
                "goals_against": goalie.get("goalsAgainst", 0),
                "in_game_sv_pct": saves / max(shots_against, 1),
            })
    return pd.DataFrame(rows)
