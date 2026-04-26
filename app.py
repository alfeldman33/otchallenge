"""
app.py — Playoff OT Goal Scorer Predictor (Streamlit web app)

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> connect repo -> deploy
"""

import csv
import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path

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
# Helper functions (must be defined before sidebar references them)
# ---------------------------------------------------------------------------

def _log_result_sidebar(gid: int, scorer: str):
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
    log_path = Path("data/pick_log.csv")
    if not log_path.exists():
        st.sidebar.info("No pick log yet.")
        return
    df = pd.read_csv(log_path)
    total = len(df)
    hits = int(df["in_top3"].sum())
    st.sidebar.metric("Top-3 hit rate", f"{100*hits/total:.1f}%", f"{hits}/{total} games")


def _build_season_only_ingame(mp: pd.DataFrame, teams: list, season: int) -> pd.DataFrame:
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
def load_history(seasons: tuple) -> pd.DataFrame:
    return fetch_data.load_moneypuck_history(list(seasons))


try:
    mp = load_history(tuple(sorted(seasons)))
except Exception as e:
    st.error(
        f"Could not load MoneyPuck data: {e}\n\n"
        "MoneyPuck may be temporarily unavailable. Try again in a few minutes, "
        "or adjust the selected seasons in the sidebar."
    )
    st.stop()

if mp.empty:
    st.error(
        "No data was returned from MoneyPuck for the selected seasons. "
        "This can happen if the site is blocking automated requests from cloud servers. "
        "Try removing older seasons from the sidebar, or try again later."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Game selection
# ---------------------------------------------------------------------------

st.subheader("Select a game")

mode = st.radio(
    "How do you want to pick the game?",
    ["Auto-detect today's games", "Enter game ID", "Enter team abbreviations"],
    horizontal=True,
)

in_game = None
goalie_stats = None
series_stats = None
home = ""
away = ""

if mode == "Auto-detect today's games":
    if st.button("Find today's playoff games"):
        with st.spinner("Checking NHL schedule..."):
            games = fetch_data.fetch_todays_playoff_games()
        if not games:
            st.warning("No playoff games found today.")
            st.session_state.pop("today_games", None)
        else:
            st.session_state["today_games"] = games

    if "today_games" in st.session_state:
        games = st.session_state["today_games"]
        options = {
            f"{g.get('awayTeam',{}).get('abbrev','?')} @ "
            f"{g.get('homeTeam',{}).get('abbrev','?')}  (ID: {g['id']})": g["id"]
            for g in games
        }
        choice = st.selectbox("Pick a game", list(options.keys()))
        st.session_state["selected_game_id"] = options[choice]

elif mode == "Enter game ID":
    gid_str = st.text_input("NHL game ID", placeholder="e.g. 2024030411")
    if gid_str.strip().isdigit():
        st.session_state["selected_game_id"] = int(gid_str.strip())

elif mode == "Enter team abbreviations":
    col1, col2 = st.columns(2)
    with col1:
        away = st.text_input("Away team", placeholder="FLA").upper().strip()
    with col2:
        home = st.text_input("Home team", placeholder="EDM").upper().strip()

game_id = st.session_state.get("selected_game_id") if mode != "Enter team abbreviations" else None

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
        if game_id:
            # Game likely hasn't started yet — fall back to season stats
            st.info("Live player data not available yet (game may not have started). Using season stats instead.")
            bs = fetch_data.fetch_boxscore(game_id)
            away_abbrev = bs.get("awayTeam", {}).get("abbrev", "")
            home_abbrev = bs.get("homeTeam", {}).get("abbrev", "")
            if away_abbrev and home_abbrev:
                in_game = _build_season_only_ingame(mp, [away_abbrev, home_abbrev], CURRENT_SEASON)
            if in_game is None or in_game.empty:
                st.error("Could not build roster. Try 'Enter team abbreviations' manually.")
                st.stop()
        else:
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

    # ---- Tabs ------------------------------------------------------------
    tab_ot, tab_fullgame = st.tabs(["OT Scorer Predictor", "Full Game Scorer"])

    # ---- Tab 1: OT Scorer ------------------------------------------------
    with tab_ot:
        st.subheader("Player rankings — OT")
        cols = st.columns(len(teams))

        for i, team in enumerate(teams):
            with cols[i]:
                team_df = ranked[ranked["teamAbbrev"] == team].copy()
                team_df = team_df[team_df["rank"] <= top_n]

                display_cols = {
                    "#": team_df["rank"],
                    "Player": team_df["name"],
                    "Pos": team_df["position"],
                    "OT Goal Prob": team_df["p_ot"].map("{:.1f}".format),
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
            "OT Goal Prob: Poisson probability of scoring in a 20-min OT period (0-100 scale). "
            "Based on best available xG/60 — current playoffs, career playoffs, or regular season."
        )

    # ---- Tab 2: Full Game Scorer -----------------------------------------
    with tab_fullgame:
        st.subheader("Full Game Scorer Probabilities")
        st.caption(
            "P(score) uses a Poisson model on xG per game from MoneyPuck (current playoff season, "
            "regular season as fallback). Enter DraftKings odds below to highlight value plays."
        )

        fg = feat.build_fullgame_probabilities(
            mp_history=mp,
            teams=teams,
            current_season=CURRENT_SEASON,
            top_n=top_n,
        )

        if fg.empty:
            st.warning("No full-game probability data available for these teams.")
        else:
            # ---- DraftKings odds input -----------------------------------
            with st.expander("Enter DraftKings anytime goal scorer odds (optional)", expanded=False):
                st.caption("Enter American odds (e.g. +150, -110). Leave blank to skip.")
                dk_odds = {}
                all_players = fg[fg["rank"] <= top_n]["name"].tolist()
                dk_cols = st.columns(2)
                for j, player in enumerate(all_players):
                    with dk_cols[j % 2]:
                        val = st.text_input(player, key=f"dk_{player}", placeholder="+150")
                        if val.strip():
                            dk_odds[player] = val.strip()

            def american_to_implied(odds_str: str) -> float:
                try:
                    o = int(odds_str.replace(" ", ""))
                    if o > 0:
                        return 100 / (o + 100)
                    else:
                        return abs(o) / (abs(o) + 100)
                except Exception:
                    return None

            fg_cols = st.columns(len(teams))
            for i, team in enumerate(teams):
                with fg_cols[i]:
                    team_fg = fg[fg["teamAbbrev"] == team].copy()
                    team_fg = team_fg[team_fg["rank"] <= top_n]
                    display = {
                        "#": team_fg["rank"],
                        "Player": team_fg["name"],
                        "Pos": team_fg["position"],
                        "Model P%": team_fg["p_score"].map("{:.1f}%".format),
                        "xG/game": team_fg["xg_per_game"].map("{:.3f}".format),
                        "Goals": team_fg["goals"].astype(int),
                        "GP": team_fg["games_played"].astype(int),
                    }
                    if dk_odds:
                        implied = []
                        value = []
                        for _, row in team_fg.iterrows():
                            imp = american_to_implied(dk_odds.get(row["name"], ""))
                            if imp is not None:
                                implied.append(f"{imp*100:.1f}%")
                                edge = row["p_score"] / 100 - imp
                                value.append(f"+{edge*100:.1f}%" if edge > 0 else f"{edge*100:.1f}%")
                            else:
                                implied.append("—")
                                value.append("—")
                        display["DK Implied%"] = implied
                        display["Edge"] = value

                    df_display = pd.DataFrame(display)

                    if dk_odds:
                        def highlight_value(row):
                            edge = row.get("Edge", "—")
                            if isinstance(edge, str) and edge.startswith("+"):
                                return ["background-color: #1a3a1a"] * len(row)
                            return [""] * len(row)
                        st.markdown(f"**{team}**")
                        st.dataframe(df_display.style.apply(highlight_value, axis=1),
                                     hide_index=True, use_container_width=True)
                    else:
                        st.markdown(f"**{team}**")
                        st.dataframe(df_display, hide_index=True, use_container_width=True)

    # Store top 3 per team in session state for logging
    st.session_state["last_game_id"] = game_id
    st.session_state["last_top3"] = {
        team: ranked[ranked["teamAbbrev"] == team].head(3)["name"].tolist()
        for team in teams
    }
    st.session_state["last_teams"] = teams
