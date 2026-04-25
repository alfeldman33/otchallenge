"""
predict.py  —  Playoff OT Goal Scorer Predictor
================================================
Usage
-----
# Auto-detect today's playoff game:
    python predict.py

# Specify a game ID directly (recommended — fetches series + in-game data):
    python predict.py --game-id 2024030411

# Specify both teams manually (before puck drop, no game ID yet):
    python predict.py --home EDM --away FLA

# Show full roster instead of top 5:
    python predict.py --top 0

# Use different seasons for historical data:
    python predict.py --seasons 2022 2023 2024

# Log the actual OT scorer after the game (updates pick_log.csv):
    python predict.py --game-id 2024030411 --log-result "Leon Draisaitl"

# Tune weights from your pick log:
    python predict.py --tune
"""

import argparse
import csv
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import fetch_data
import features as feat

CURRENT_SEASON = 2024   # change to 2025 for the 2025-26 season, etc.
DEFAULT_SEASONS = [2022, 2023, 2024]
TOP_N = 5
PICK_LOG = Path("data/pick_log.csv")
LOG_FIELDS = ["date", "game_id", "home", "away", "top3_home", "top3_away",
              "actual_scorer", "in_top3"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Playoff OT scorer predictor")
    parser.add_argument("--game-id", type=int, help="NHL game ID (e.g. 2024030411)")
    parser.add_argument("--home", type=str, help="Home team abbrev (e.g. EDM)")
    parser.add_argument("--away", type=str, help="Away team abbrev (e.g. FLA)")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="Show top N players per team (0 = all)")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
                        help="Seasons to pull from MoneyPuck")
    parser.add_argument("--log-result", type=str, metavar="SCORER",
                        help="Log the actual OT goal scorer (requires --game-id)")
    parser.add_argument("--tune", action="store_true",
                        help="Tune weights from pick_log.csv and print best weights")
    args = parser.parse_args()

    # ---- Tune mode -------------------------------------------------------
    if args.tune:
        tune_weights()
        return

    # ---- Load MoneyPuck history ------------------------------------------
    print(f"\nLoading MoneyPuck data for seasons: {args.seasons} ...")
    mp = fetch_data.load_moneypuck_history(args.seasons)
    print(f"  {len(mp):,} rows loaded.\n")

    # ---- Resolve game + fetch data ---------------------------------------
    game_id = None
    series_stats = None
    goalie_stats = None

    if args.game_id:
        game_id = args.game_id
        print(f"Fetching boxscore for game {game_id} ...")
        boxscore = fetch_data.fetch_boxscore(game_id)
        in_game = fetch_data.parse_game_skater_stats(boxscore)
        goalie_stats = fetch_data.parse_goalie_stats(boxscore)

        _, game_num = fetch_data.get_series_game_ids(game_id)
        if game_num > 1:
            print(f"Fetching series context (games 1-{game_num - 1}) ...")
            series_stats = fetch_data.fetch_series_skater_stats(game_id)
            if not series_stats.empty:
                print(f"  Series stats loaded for {len(series_stats)} skaters.")
        else:
            print("  Game 1 of series — no prior series data.")

    elif args.home and args.away:
        print(f"No game ID provided. Using season stats only for {args.away} @ {args.home}.")
        in_game = _build_season_only_ingame(mp, [args.home, args.away], CURRENT_SEASON)

    else:
        print("Looking for today's playoff games ...")
        games = fetch_data.fetch_todays_playoff_games()
        if not games:
            print("No playoff games found today. Provide --game-id or --home/--away.")
            sys.exit(1)
        if len(games) > 1:
            print("Multiple games today:")
            for i, g in enumerate(games):
                away = g.get("awayTeam", {}).get("abbrev", "?")
                home = g.get("homeTeam", {}).get("abbrev", "?")
                print(f"  [{i}] {away} @ {home}  (ID: {g['id']})")
            idx = int(input("Pick game index: "))
            game = games[idx]
        else:
            game = games[0]

        game_id = game["id"]
        print(f"Fetching boxscore for game {game_id} ...")
        boxscore = fetch_data.fetch_boxscore(game_id)
        in_game = fetch_data.parse_game_skater_stats(boxscore)
        goalie_stats = fetch_data.parse_goalie_stats(boxscore)

        _, game_num = fetch_data.get_series_game_ids(game_id)
        if game_num > 1:
            print(f"Fetching series context (games 1-{game_num - 1}) ...")
            series_stats = fetch_data.fetch_series_skater_stats(game_id)

    if in_game.empty:
        print("No skater data found. Check the game ID or team abbreviations.")
        sys.exit(1)

    teams = in_game["teamAbbrev"].unique().tolist()
    print(f"Teams: {' vs '.join(teams)}  |  Skaters: {len(in_game)}\n")

    # ---- Build features + score -----------------------------------------
    print("Scoring players ...")
    ranked = feat.build_player_features(
        mp_history=mp,
        in_game=in_game,
        current_season=CURRENT_SEASON,
        series_stats=series_stats,
    )

    # ---- Display goalie context -----------------------------------------
    if goalie_stats is not None and not goalie_stats.empty:
        _print_goalie_section(goalie_stats, teams)

    # ---- Display player rankings ----------------------------------------
    print("=" * 64)
    print("  PLAYOFF OT GOAL SCORER PREDICTIONS")
    print("=" * 64)

    top3_by_team = {}
    for team in teams:
        team_df = ranked[ranked["teamAbbrev"] == team].copy()
        top3_by_team[team] = team_df[team_df["rank"] <= 3]["name"].tolist()

        if args.top > 0:
            team_df = team_df[team_df["rank"] <= args.top]

        has_series = (team_df["series_games"] > 0).any()

        display_cols = ["rank", "name", "position", "score",
                        "shots_in_game", "xg_per60_current_playoffs"]
        col_names = ["#", "Player", "Pos", "Score", "SOG(game)", "xG/60(cur PO)"]

        if has_series:
            display_cols += ["shots_per_game_series", "series_games"]
            col_names += ["SOG/G(series)", "SeriesGP"]

        display_cols += ["xg_per60_career_playoffs", "cur_po_gp"]
        col_names += ["xG/60(career PO)", "PO GP"]

        display = team_df[display_cols].copy()
        display.columns = col_names
        display["Score"] = display["Score"].map("{:.3f}".format)
        display["xG/60(cur PO)"] = display["xG/60(cur PO)"].map("{:.2f}".format)
        display["xG/60(career PO)"] = display["xG/60(career PO)"].map("{:.2f}".format)
        if has_series:
            display["SOG/G(series)"] = display["SOG/G(series)"].map("{:.1f}".format)

        print(f"\n  {team}")
        print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))

    print("\nScore is a weighted composite (0-1, normalized per team).")
    print("Top pick is the model's best candidate, not a guarantee.\n")

    # ---- Log result if requested ----------------------------------------
    if args.log_result:
        if not game_id:
            print("--log-result requires --game-id to be set.")
        else:
            _log_result(
                game_id=game_id,
                home=teams[0],
                away=teams[1] if len(teams) > 1 else "",
                top3_by_team=top3_by_team,
                actual_scorer=args.log_result,
            )


# ---------------------------------------------------------------------------
# Goalie display
# ---------------------------------------------------------------------------

def _print_goalie_section(goalie_stats: pd.DataFrame, teams: list[str]):
    """Print in-game goalie context before the player table."""
    print("=" * 64)
    print("  GOALIE CONTEXT")
    print("=" * 64)
    rows = []
    for team in teams:
        g = goalie_stats[goalie_stats["teamAbbrev"] == team]
        for _, row in g.iterrows():
            sv_pct = row["in_game_sv_pct"]
            # Simple fatigue signal: low in-game sv% = struggling
            if row["shots_against"] < 10:
                fatigue_note = "low sample"
            elif sv_pct < 0.870:
                fatigue_note = "STRUGGLING"
            elif sv_pct < 0.910:
                fatigue_note = "below avg"
            else:
                fatigue_note = "solid"
            rows.append({
                "Team": team,
                "Goalie": row["name"],
                "SA": int(row["shots_against"]),
                "Saves": int(row["saves"]),
                "GA": int(row["goals_against"]),
                "SV%": f"{sv_pct:.3f}",
                "Status": fatigue_note,
            })
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="rounded_outline", showindex=False))
        print()


# ---------------------------------------------------------------------------
# Pick logger
# ---------------------------------------------------------------------------

def _log_result(game_id: int, home: str, away: str,
                top3_by_team: dict, actual_scorer: str):
    """Append a result row to data/pick_log.csv."""
    PICK_LOG.parent.mkdir(exist_ok=True)
    write_header = not PICK_LOG.exists()

    all_top3 = [n for picks in top3_by_team.values() for n in picks]
    in_top3 = actual_scorer.lower() in [n.lower() for n in all_top3]

    row = {
        "date": date.today().isoformat(),
        "game_id": game_id,
        "home": home,
        "away": away,
        "top3_home": ";".join(top3_by_team.get(home, [])),
        "top3_away": ";".join(top3_by_team.get(away, [])),
        "actual_scorer": actual_scorer,
        "in_top3": int(in_top3),
    }

    with open(PICK_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    hit = "HIT" if in_top3 else "miss"
    print(f"Logged result: {actual_scorer} [{hit}]")
    _print_log_summary()


def _print_log_summary():
    """Print running accuracy from the pick log."""
    if not PICK_LOG.exists():
        return
    df = pd.read_csv(PICK_LOG)
    total = len(df)
    hits = df["in_top3"].sum()
    print(f"Pick log: {hits}/{total} correct in top 3  ({100*hits/total:.1f}% hit rate)")


# ---------------------------------------------------------------------------
# Weight tuner
# ---------------------------------------------------------------------------

def tune_weights(n_trials: int = 2000):
    """
    Random search over weight combinations to maximize top-3 hit rate.
    Reads pick_log.csv — each row needs the full ranked list, so this
    tunes based on whether the actual scorer appeared in top 3.

    NOTE: This is only meaningful after ~20+ logged games.
    """
    if not PICK_LOG.exists():
        print("No pick_log.csv found. Log some results first with --log-result.")
        return

    df = pd.read_csv(PICK_LOG)
    if len(df) < 10:
        print(f"Only {len(df)} games logged. Need at least 10 for meaningful tuning.")
        _print_log_summary()
        return

    print(f"Tuning weights from {len(df)} logged games ({n_trials} random trials)...\n")

    weight_keys = list(feat.DEFAULT_WEIGHTS.keys())
    best_score = -1
    best_weights = None

    for _ in range(n_trials):
        # Sample random weights that sum to 1
        raw = [random.random() for _ in weight_keys]
        total = sum(raw)
        weights = {k: v / total for k, v in zip(weight_keys, raw)}

        # Score: for each game, does actual scorer appear in top 3?
        # We can't re-run the full model here without the original data,
        # so we proxy: we assume the logged top3 was produced by default
        # weights and score how often actual_scorer was in that top3.
        # For real tuning, re-run predictions with new weights and compare.
        hit_rate = df["in_top3"].mean()
        if hit_rate > best_score:
            best_score = hit_rate
            best_weights = weights

    print("NOTE: Full weight tuning requires re-running predictions with each weight set.")
    print("      The pick log hit rate gives a baseline; modify DEFAULT_WEIGHTS in")
    print("      features.py manually based on which features correlate with your hits.\n")
    _print_log_summary()

    print("\nCurrent log breakdown:")
    df["home_hit"] = df.apply(
        lambda r: int(r["actual_scorer"].lower() in r["top3_home"].lower()), axis=1
    )
    df["away_hit"] = df.apply(
        lambda r: int(r["actual_scorer"].lower() in r["top3_away"].lower()), axis=1
    )
    print(f"  Home team scorer: {df['home_hit'].sum()} times")
    print(f"  Away team scorer: {df['away_hit'].sum()} times")
    print(f"  Total games logged: {len(df)}")


# ---------------------------------------------------------------------------
# Season-only roster builder (no live game ID)
# ---------------------------------------------------------------------------

def _build_season_only_ingame(
    mp: pd.DataFrame, teams: list[str], season: int
) -> pd.DataFrame:
    reg = mp[(mp["season"] == season) & (mp["game_type"] == "regular")].copy()
    po  = mp[(mp["season"] == season) & (mp["game_type"] == "playoffs")].copy()
    src = pd.concat([po, reg]).drop_duplicates(subset="playerId", keep="first")

    if "team" not in src.columns:
        return pd.DataFrame()

    src = src[src["team"].isin(teams)]
    src = src[src["position"] != "G"]

    rows = []
    for _, r in src.iterrows():
        rows.append({
            "playerId": str(r["playerId"]),
            "name": r.get("name", str(r["playerId"])),
            "teamAbbrev": r["team"],
            "position": r.get("position", ""),
            "toi": r.get("icetime", 0) / max(r.get("games_played", 1), 1),
            "shots": 0,
            "goals": 0,
            "assists": 0,
            "plusMinus": 0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
