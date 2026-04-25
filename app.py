"""
app.py — Playoff OT Goal Scorer Predictor (Streamlit web app)

Run locally:   streamlit run app.py
Deploy:        push to GitHub → share.streamlit.io → connect repo → deploy
"""

import streamlit as st
import pandas as pd
from datetime import date

import fetch_data
import features as feat

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Playoff OT Predictor",
    page_icon="🏒",
    layout="wide",
)

st.title("🏒 Playoff OT Goal Scorer Predictor")
st.caption("Ranks players by likelihood of scoring in sudden-death playoff overtime.")

CURRENT_SEASON = 2024
DEFAULT_SEASONS = [2022, 2023, 2024]

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    top_n = st.slider("Players to show per team", 3, 20, 5)
    seasons = st.multiselect(
        "Historical seasons (MoneyPuck)",
        options=[2019, 2020, 2021, 2022, 2023, 2024],
        default=DEFAULT_SEASONS,
    )
    if not seasons:
        seasons = DEFAULT_SEASONS

    st.divider()
    st.header("Log a Result")
    log_scorer = st.text_input("Actual OT scorer (first + last name)")
    log_game_id_input = st.text_input("Game ID for this result")
    if st.button("Log result"):
        if log_scorer and log_game_id_input:
            _log_result_sidebar(int(log_game_id_input), log_scorer)
        else:
            st.warning("Enter both scorer name and game ID to log.")

    st.divider()
    if st.button("Show pick log accuracy"):
        _show_log_summary()


# ---------------------------------------------------------------------------
# MoneyPuck data (cached so it only downloads once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading historical data from MoneyPuck...")
def load_history(seasons: tuple[int, ...]) -> pd.DataFrame:
    return fetch_data.load_moneypuck_history(list(seasons))


mp = load_history(tuple(sorted(seasons)))

# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------

st.subheader("Select a game")

mode = st.radio(
    "How do you want to pick the game?",
    ["Auto-detect today's games", "Enter game ID", "Enter team abbreviations"],
    horizontal=True,
)

game_id = None
in_game = None
goalie_stats = None
series_stats = None

if mode == "Auto-detect today's games":
    if st.button("Find today's playoff games"):
        with st.spinner("Checking NHL schedule..."):
            games = fetch_data.fetch_todays_playoff_games()

        if not games:
            st.warning("No playoff games found today.")
        else:
            options = {
                f"{g.get('awayTeam',{}).get('abbrev','?')} @ "
                f"{g.get('homeTeam',{}).get('abbrev','?')}  (ID: {g['id']})": g["id"]
                for g in games
            }
            choice = st.selectbox("Pick a game", list(options.keys()))
            game_id = options[choice]

elif mode == "Enter game ID":
    gid_str = st.text_input("NHL game ID", placeholder="e.g. 2024030411")
    if gid_str.strip().isdigit():
        game_id = int(gid_str.strip())

elif mode == "Enter team abbreviations":
    col1, col2 = st.columns(2)
    with col1:
        away = st.text_input("Away team", placeholder="FLA").upper().strip()
    with col2:
        home = st.text_input("Home team", placeholder="EDM").upper().strip()

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------

run = st.button("Run prediction", type="primary")

if run:
    with st.spinner("Fetching game data..."):

        if game_id:
            boxscore = fetch_data.fetch_boxscore(game_id)
            in_game = fetch_data.parse_game_skater_stats(boxscore)
            goalie_stats = fetch_data.parse_goalie_stats(boxscore)

            _, game_num = fetch_data.get_series_game_ids(game_id)
            if game_num > 1:
                series_stats = fetch_data.fetch_series_skater_stats(game_id)

        elif mode == "Enter team abbreviations" and home and away:
            in_game = _build_season_only_ingame(mp, [home, away], CURRENT_SEASON)

        else:
            st.error("Please select or enter a game first.")
            st.stop()

    if in_game is None or in_game.empty:
        st.error("No player data found. Check the game ID or team abbreviations.")
        st.stop()

    with st.spinner("Scoring players..."):
        ranked = feat.build_player_features(
            mp_history=mp,
            in_game=in_game,
            current_season=CURRENT_SEASON,
            series_stats=series_stats,
        )

    teams = in_game["teamAbbrev"].unique().tolist()

    # ---- Goalie context --------------------------------------------------
    if goalie_stats is not None and not goalie_stats.empty:
        with st.expander("Goalie context", expanded=True):
            rows = []
            for team in teams:
                g = goalie_stats[goalie_stats["teamAbbrev"] == team]
                for _, row in g.iterrows():
                    sv = row["in_game_sv_pct"]
                    if row["shots_against"] < 10:
                        status = "low sample"
                    elif sv < 0.870:
                        status = "STRUGGLING"
                    elif sv < 0.910:
                        status = "below avg"
                    else:
                        status = "solid"
                    rows.append({
                        "Team": team,
                        "Goalie": row["name"],
                        "SA": int(row["shots_against"]),
                        "Saves": int(row["saves"]),
                        "GA": int(row["goals_against"]),
                        "SV%": f"{sv:.3f}",
                        "Status": status,
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- Series context note ---------------------------------------------
    has_series = series_stats is not None and not series_stats.empty
    if has_series:
        _, game_num = fetch_data.get_series_game_ids(game_id)
        st.info(f"Series stats included — data from {game_num - 1} prior game(s) in this matchup.")

    # ---- Player rankings -------------------------------------------------
    st.subheader("Player rankings")
    cols = st.columns(len(teams))

    for i, team in enumerate(teams):
        with cols[i]:
            team_df = ranked[ranked["teamAbbrev"] == team].copy()
            team_df = team_df[team_df["rank"] <= top_n]

            display_cols = {
                "#": team_df["rank"],
                "Player": team_df["name"],
                "Pos": team_df["position"],
                "Score": team_df["score"].map("{:.3f}".format),
                "SOG (game)": team_df["shots_in_game"].astype(int),
                "xG/60 (cur PO)": team_df["xg_per60_current_playoffs"].map("{:.2f}".format),
            }
            if has_series:
                display_cols["SOG/G (series)"] = team_df["shots_per_game_series"].map("{:.1f}".format)
            display_cols["xG/60 (career PO)"] = team_df["xg_per60_career_playoffs"].map("{:.2f}".format)
            display_cols["PO GP"] = team_df["cur_po_gp"].astype(int)

            st.markdown(f"**{team}**")
            st.dataframe(pd.DataFrame(display_cols), hide_index=True, use_container_width=True)

    st.caption(
        "Score is a weighted composite normalized 0–1 per team. "
        "Top pick is the model's best candidate, not a guarantee."
    )

    # Store top 3 per team in session state for logging
    st.session_state["last_game_id"] = game_id
    st.session_state["last_top3"] = {
        team: ranked[ranked["teamAbbrev"] == team].head(3)["name"].tolist()
        for team in teams
    }
    st.session_state["last_teams"] = teams


# ---------------------------------------------------------------------------
# Helpers referenced in sidebar (defined after mp is loaded)
# ---------------------------------------------------------------------------

def _log_result_sidebar(gid: int, scorer: str):
    import csv
    from pathlib import Path

    log_path = Path("data/pick_log.csv")
    log_path.parent.mkdir(exist_ok=True)
    fields = ["date", "game_id", "home", "away", "top3_home", "top3_away",
              "actual_scorer", "in_top3"]

    top3 = st.session_state.get("last_top3", {})
    teams = st.session_state.get("last_teams", ["", ""])
    all_names = [n for picks in top3.values() for n in picks]
    in_top3 = int(scorer.lower() in [n.lower() for n in all_names])

    row = {
        "date": date.today().isoformat(),
        "game_id": gid,
        "home": teams[0] if teams else "",
        "away": teams[1] if len(teams) > 1 else "",
        "top3_home": ";".join(top3.get(teams[0], []) if teams else []),
        "top3_away": ";".join(top3.get(teams[1], []) if len(teams) > 1 else []),
        "actual_scorer": scorer,
        "in_top3": in_top3,
    }

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    label = "HIT" if in_top3 else "miss"
    st.sidebar.success(f"Logged: {scorer} [{label}]")


def _show_log_summary():
    from pathlib import Path
    log_path = Path("data/pick_log.csv")
    if not log_path.exists():
        st.sidebar.info("No pick log yet.")
        return
    df = pd.read_csv(log_path)
    total = len(df)
    hits = int(df["in_top3"].sum())
    st.sidebar.metric("Top-3 hit rate", f"{100*hits/total:.1f}%", f"{hits}/{total} games")


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
            "shots": 0, "goals": 0, "assists": 0, "plusMinus": 0,
        })
    return pd.DataFrame(rows)
