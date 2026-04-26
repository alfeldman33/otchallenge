"""
features.py
Builds a scored/ranked player table from MoneyPuck history + in-game stats.

Scoring formula (all components normalized 0-1 before weighting):

  score = w1 * xg_per60_current_playoffs   (who's generating this postseason)
        + w2 * shots_per_game_series        (activity vs. THIS goalie this series)
        + w3 * shots_in_game                (hot right now)
        + w4 * xg_per60_career_playoffs     (proven playoff performer)
        + w5 * xg_per60_reg_season          (baseline skill, current season)
        + w6 * toi_share_in_game            (coach trust / on ice)
        + w7 * career_playoff_sh_pct        (finisher bonus)

Weights are tunable. Defaults lean heavily on current-playoff xG and
series-specific activity since OT is a direct continuation of this matchup.
"""

import numpy as np
import pandas as pd

# Default weights — must sum to 1.0; tweak these as you gather results
DEFAULT_WEIGHTS = {
    "xg_per60_current_playoffs": 0.25,
    "shots_per_game_series":     0.22,
    "shots_in_game":             0.15,
    "xg_per60_career_playoffs":  0.15,
    "xg_per60_reg_season":       0.10,
    "toi_share_in_game":         0.08,
    "career_playoff_sh_pct":     0.05,
}

MIN_PLAYOFF_GP = 3    # min games to trust current-playoff xG rate
MIN_CAREER_GP  = 10   # min career playoff games for career stats


def build_player_features(
    mp_history: pd.DataFrame,
    in_game: pd.DataFrame,
    current_season: int,
    series_stats: pd.DataFrame = None,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    mp_history    : Output of fetch_data.load_moneypuck_history()
    in_game       : Output of fetch_data.parse_game_skater_stats() for current game
    current_season: e.g. 2024 for the 2024-25 season
    series_stats  : Output of fetch_data.fetch_series_skater_stats() (optional)
    weights       : Override DEFAULT_WEIGHTS if desired

    Returns
    -------
    DataFrame of skaters with feature columns + final 'score', sorted best-first.
    Each team is ranked separately.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # ---- 0. Dedupe in_game and rename shots -> shots_in_game ------------
    in_game = in_game.copy().drop_duplicates(subset="playerId")
    if "shots" in in_game.columns and "shots_in_game" not in in_game.columns:
        in_game = in_game.rename(columns={"shots": "shots_in_game"})

    cur_po = _filter(mp_history, season=current_season, game_type="playoffs")
    cur_po = _add_per60(cur_po, prefix="cur_po")

    # ---- 2. Career playoff stats (all seasons loaded) -------------------
    all_po = mp_history[mp_history["game_type"] == "playoffs"].copy()
    if "situation" in all_po.columns:
        all_po = all_po[all_po["situation"] == "all"]
    career_po = (
        all_po.groupby("playerId", as_index=False)
        .agg(
            career_po_gp=("games_played", "sum"),
            career_po_goals=("I_F_goals", "sum"),
            career_po_xg=("I_F_xGoals", "sum"),
            career_po_toi=("icetime", "sum"),
            career_po_shots=("I_F_shotsOnGoal", "sum"),
        )
    )
    career_po["xg_per60_career_playoffs"] = np.where(
        career_po["career_po_toi"] > 0,
        career_po["career_po_xg"] / (career_po["career_po_toi"] / 60),
        0,
    )
    career_po["career_playoff_sh_pct"] = np.where(
        career_po["career_po_shots"] > 0,
        career_po["career_po_goals"] / career_po["career_po_shots"],
        0,
    )
    # Zero out players with too-small sample
    career_po.loc[career_po["career_po_gp"] < MIN_CAREER_GP, "xg_per60_career_playoffs"] = 0
    career_po.loc[career_po["career_po_gp"] < MIN_CAREER_GP, "career_playoff_sh_pct"] = 0

    # ---- 3. Current regular-season baseline -----------------------------
    cur_reg = _filter(mp_history, season=current_season, game_type="regular")
    cur_reg = _add_per60(cur_reg, prefix="cur_reg")

    # ---- 4. Merge everything onto in-game player list -------------------
    df = in_game.copy()

    # MoneyPuck uses numeric playerId — ensure same dtype
    df["playerId"] = df["playerId"].astype(str)
    for frame in (cur_po, career_po, cur_reg):
        frame["playerId"] = frame["playerId"].astype(str)

    df = df.merge(
        cur_po[["playerId", "cur_po_gp", "xg_per60_current_playoffs"]],
        on="playerId", how="left"
    )
    df = df.merge(
        career_po[["playerId", "career_po_gp", "xg_per60_career_playoffs", "career_playoff_sh_pct"]],
        on="playerId", how="left"
    )
    df = df.merge(
        cur_reg[["playerId", "xg_per60_reg_season"]],
        on="playerId", how="left"
    )
    # ---- 4b. Series stats (this matchup vs. this goalie) ----------------
    if series_stats is not None and not series_stats.empty:
        series_stats = series_stats.copy()
        series_stats["playerId"] = series_stats["playerId"].astype(str)
        df = df.merge(
            series_stats[["playerId", "shots_per_game_series", "goals_per_game_series", "series_games"]],
            on="playerId", how="left"
        )
    else:
        df["shots_per_game_series"] = 0.0
        df["goals_per_game_series"] = 0.0
        df["series_games"] = 0

    df = df.fillna(0)

    # Zero out current-playoff rate if too small a sample
    df.loc[df["cur_po_gp"] < MIN_PLAYOFF_GP, "xg_per60_current_playoffs"] = 0

    # ---- 5. In-game derived features ------------------------------------
    df["toi_share_in_game"] = df.groupby("teamAbbrev")["toi"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )

    # ---- 6. Normalize each feature column 0-1 per team -----------------
    feature_cols = list(weights.keys())
    for team, grp in df.groupby("teamAbbrev"):
        for col in feature_cols:
            col_min = grp[col].min()
            col_max = grp[col].max()
            denom = col_max - col_min
            if denom > 0:
                df.loc[grp.index, f"norm_{col}"] = (grp[col] - col_min) / denom
            else:
                df.loc[grp.index, f"norm_{col}"] = 0.0

    # ---- 7. Weighted score ----------------------------------------------
    df["score"] = sum(
        weights[col] * df[f"norm_{col}"] for col in feature_cols
    )

    # ---- 8. OT goal probability (Poisson, 1-100 scale) ------------------
    # Use best available xG/60: current playoffs > career playoffs > reg season
    xg = df["xg_per60_current_playoffs"].where(df["cur_po_gp"] >= MIN_PLAYOFF_GP, 0)
    xg = xg.where(xg > 0, df["xg_per60_career_playoffs"])
    xg = xg.where(xg > 0, df["xg_per60_reg_season"])
    # Expected xG in a 20-minute OT period
    df["p_ot"] = (1 - np.exp(-(xg * (20 / 60)).clip(lower=0))) * 100
    df["p_ot"] = df["p_ot"].round(1)

    # ---- 9. Rank within each team (1 = best) ----------------------------
    df["rank"] = df.groupby("teamAbbrev")["score"].rank(ascending=False, method="min").astype(int)

    output_cols = [
        "rank", "name", "teamAbbrev", "position",
        "score", "p_ot",
        "xg_per60_current_playoffs",
        "shots_per_game_series",
        "series_games",
        "shots_in_game",
        "xg_per60_career_playoffs",
        "xg_per60_reg_season",
        "toi_share_in_game",
        "career_playoff_sh_pct",
        "cur_po_gp", "career_po_gp",
    ]
    return df[output_cols].sort_values(["teamAbbrev", "rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full-game scoring probabilities (Poisson on xG)
# ---------------------------------------------------------------------------

def build_fullgame_probabilities(
    mp_history: pd.DataFrame,
    teams: list,
    current_season: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    For each skater on the given teams, estimate P(score >= 1 goal in this game)
    using a Poisson model: P = 1 - exp(-xg_per_game).

    xg_per_game is taken from current-season playoffs first, then regular season
    as a fallback for players with no playoff sample.

    Returns a DataFrame with columns:
        rank, name, teamAbbrev, position, p_score (%), xg_per_game, games_played, goals
    sorted best-first within each team.
    """
    import math

    if mp_history.empty or "team" not in mp_history.columns:
        return pd.DataFrame()

    po = mp_history[
        (mp_history["season"] == current_season) & (mp_history["game_type"] == "playoffs")
    ].copy()
    reg = mp_history[
        (mp_history["season"] == current_season) & (mp_history["game_type"] == "regular")
    ].copy()

    # Prefer playoff data; fall back to regular for players with no playoff rows
    combined = pd.concat([po, reg]).drop_duplicates(subset="playerId", keep="first")
    combined = combined[combined["team"].isin(teams)].copy()
    combined = combined[combined["position"] != "G"]

    if combined.empty:
        return pd.DataFrame()

    combined["xg_per_game"] = np.where(
        combined["games_played"] > 0,
        combined["I_F_xGoals"] / combined["games_played"],
        0,
    )
    combined["p_score"] = combined["xg_per_game"].apply(
        lambda xg: (1 - math.exp(-max(float(xg), 0))) * 100
    )

    combined["rank"] = (
        combined.groupby("team")["p_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    out = combined[["rank", "name", "team", "position",
                    "p_score", "xg_per_game", "games_played", "I_F_goals"]].copy()
    out.columns = ["rank", "name", "teamAbbrev", "position",
                   "p_score", "xg_per_game", "games_played", "goals"]

    return out.sort_values(["teamAbbrev", "rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter(df: pd.DataFrame, season: int, game_type: str) -> pd.DataFrame:
    result = df[(df["season"] == season) & (df["game_type"] == game_type)].copy()
    if "situation" in result.columns:
        result = result[result["situation"] == "all"]
    return result.drop_duplicates(subset="playerId")


def _add_per60(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add xG/60 and rename games_played for the given slice."""
    key = f"{prefix}_gp"
    df = df.rename(columns={"games_played": key})
    df[f"xg_per60_{prefix}"] = np.where(
        df["icetime"] > 0,
        df["I_F_xGoals"] / (df["icetime"] / 60),
        0,
    )
    # Normalize the column name to a stable output name
    if prefix == "cur_po":
        df = df.rename(columns={f"xg_per60_{prefix}": "xg_per60_current_playoffs"})
    elif prefix == "cur_reg":
        df = df.rename(columns={f"xg_per60_{prefix}": "xg_per60_reg_season"})
    return df
