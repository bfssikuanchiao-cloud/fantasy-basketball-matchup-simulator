from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from fantasy_engine import (
    SEASONS,
    load_yahoo_matchup_context,
    select_yahoo_league_matchup,
    build_yahoo_historical_rosters,
    get_yahoo_authorization_url,
    exchange_yahoo_verifier,
    load_schedule,
    compare_add_drop_fast,
)

# ============================================================
# PAGE SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).parent
LOGO_PATH = APP_FOLDER / "秘密結社.png"
LOGO1_PATH = APP_FOLDER / "秘密結社1.png"
SCHEDULE_PATH = APP_FOLDER / "schedule.csv"




col1, col2, col3, col4, col5, col6 = st.columns([0.3,2,1,1,1,1])

with col2:
    st.image(
        str(LOGO1_PATH),
        width=428
    )

st.title(
    "🟣🟡 秘密結社 🟡🟣"
)

st.title(
    "🏆🏀 夢幻籃球模擬器 🏀🏆"
)


st.caption(
    "蒙地卡羅 H2H 傳統九項機率引擎"
)

st.markdown(
    """
- 多賽季歷史數據池
- 模型抽樣比重隨賽程動態調整
- Add / Drop 機率分析
- Yahoo Fantasy API 聯盟設定導入


"""
)

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "my_team": [],
    "opp_team": [],
    "my_roster_data": None,
    "opp_roster_data": None,
    "rosters_loaded": False,
    "last_result": None,
    "last_scenario_message": None,
    "yahoo_context": None,
    "yahoo_loaded": False,
    "yahoo_league_input": "",
    "yahoo_token": None,
    "yahoo_verifier_code": "",
    "yahoo_current_totals_loaded": False,
    "yahoo_matchup_loaded": False,
    "yahoo_history_loaded": False,
    "selected_my_team_key": None,
    "selected_opp_team_key": None,
    "my_game_selection": {},
    "opp_game_selection": {},
    "game_selection_signature": None,
    "game_selection_fingerprint": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================
# HELPERS
# ============================================================
def clean_player_name(name):
    return " ".join(str(name).strip().split()).title()


def invalidate_loaded_rosters():
    st.session_state.my_roster_data = None
    st.session_state.opp_roster_data = None
    st.session_state.rosters_loaded = False
    st.session_state.my_game_selection = {}
    st.session_state.opp_game_selection = {}
    st.session_state.game_selection_signature = None
    st.session_state.game_selection_fingerprint = None


def roster_to_dataframe(team_list, yahoo_players=None):
    yahoo_by_name = {
        clean_player_name(player.get("name", "")): player
        for player in (yahoo_players or [])
        if player.get("name")
    }
    rows = []
    for index, player_name in enumerate(team_list, start=1):
        metadata = yahoo_by_name.get(clean_player_name(player_name), {})
        eligible = metadata.get("eligible_positions") or []
        if isinstance(eligible, str):
            eligible = [eligible]
        rows.append({
            "No.": index,
            "Player": player_name,
            "Pos": metadata.get("selected_position") or "",
            "Eligible": ",".join(str(value) for value in eligible if value),
        })
    return pd.DataFrame(rows, columns=["No.", "Player", "Pos", "Eligible"])

def yahoo_roster_metadata(team_key):
    context = st.session_state.get("yahoo_context")
    if not isinstance(context, dict):
        return []
    if team_key == "my_team":
        return context.get("my_roster", [])
    if team_key == "opp_team":
        return context.get("opponent_roster", [])
    return []


def synchronize_selected_matchup(selected_my_key, selected_opp_key):
    context = st.session_state.get("yahoo_context")
    if not isinstance(context, dict):
        return
    if selected_my_key == selected_opp_key:
        st.warning("本陣與敵陣必須選擇不同隊伍。")
        return
    if (
        selected_my_key == context.get("my_team_key")
        and selected_opp_key == context.get("opponent_team_key")
    ):
        return
    with st.spinner("正在切換 matchup、載入雙方陣容與本週累積數據..."):
        selected = select_yahoo_league_matchup(
            context,
            selected_my_key,
            selected_opp_key,
        )
    st.session_state.yahoo_context = selected
    st.session_state.my_team = [
        player["name"] for player in selected["my_roster"]
    ]
    st.session_state.opp_team = [
        player["name"] for player in selected["opponent_roster"]
    ]
    current_stat_fields = [
        "FGA", "FGM", "FTA", "FTM", "3PM",
        "Pts", "Reb", "Ast", "Stl", "Blk", "TO",
    ]
    for stat in current_stat_fields:
        st.session_state[f"current_my_{stat}"] = int(
            selected["current_my"].get(stat, 0)
        )
        st.session_state[f"current_opp_{stat}"] = int(
            selected["current_opp"].get(stat, 0)
        )
    invalidate_loaded_rosters()
    st.session_state.yahoo_history_loaded = False
    st.session_state.yahoo_current_totals_loaded = True
    st.session_state.last_result = None
    st.session_state.last_scenario_message = None
    st.rerun()


def show_team_editor(team_key, title):
    team = st.session_state[team_key]
    st.subheader(f"{title} ({len(team)})")
    select_col, erase_col, clear_col = st.columns([1.35, 1, 1])
    remove_player = None
    with select_col:
        if team:
            remove_player = st.selectbox(
                "選擇球員",
                options=team,
                key=f"remove_select_{team_key}",
                label_visibility="collapsed",
            )
        else:
            st.selectbox(
                "選擇球員",
                options=["目前沒有球員"],
                key=f"remove_empty_{team_key}",
                disabled=True,
                label_visibility="collapsed",
            )
    with erase_col:
        if st.button(
            "Erase Selected",
            key=f"erase_selected_{team_key}",
            disabled=not bool(team),
            use_container_width=True,
        ):
            st.session_state[team_key].remove(remove_player)
            context = st.session_state.get("yahoo_context")
            if isinstance(context, dict):
                roster_field = (
                    "my_roster" if team_key == "my_team" else "opponent_roster"
                )
                context[roster_field] = [
                    player for player in context.get(roster_field, [])
                    if player.get("name") != remove_player
                ]
            invalidate_loaded_rosters()
            st.session_state.yahoo_history_loaded = False
            st.session_state.last_result = None
            st.rerun()
    with clear_col:
        if st.button(
            f"Clear {title}",
            key=f"clear_{team_key}",
            disabled=not bool(team),
            use_container_width=True,
        ):
            st.session_state[team_key] = []
            context = st.session_state.get("yahoo_context")
            if isinstance(context, dict):
                roster_field = (
                    "my_roster" if team_key == "my_team" else "opponent_roster"
                )
                context[roster_field] = []
            invalidate_loaded_rosters()
            st.session_state.yahoo_history_loaded = False
            st.session_state.last_result = None
            st.rerun()
    st.markdown(f"#### 目前 {title}")
    if team:
        st.dataframe(
            roster_to_dataframe(team, yahoo_roster_metadata(team_key)),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(f"{title} 目前沒有球員。")

def yahoo_injury_status(player):
    """Return Yahoo's raw injury status for application logic."""
    return str(player.get("status") or "").strip().upper()


def yahoo_injury_designation(player):
    """Return an injury label for display only; keep raw Yahoo status untouched."""
    status = yahoo_injury_status(player)
    display_map = {
        "O": "O ❌",
        "D": "D 🟠",
        "Q": "Q 🔵",
        "P": "P 🟢",
    }
    return display_map.get(status, status)


def style_injury_status(value):
    """Bold/color the read-only Injury cell without changing its underlying logic."""
    display = str(value or "").strip()
    status = display.split(maxsplit=1)[0].upper() if display else ""
    color_map = {
        "O": "#DC2626",  # red
        "D": "#F97316",  # orange
        "Q": "#2563EB",  # blue
        "P": "#16A34A",  # green
    }
    color = color_map.get(status)
    return f"color: {color}; font-weight: 700;" if color else ""


def injury_default_enabled(player):
    """Use only raw Yahoo status for schedule-selection defaults."""
    raw_status = yahoo_injury_status(player)
    slot = str(player.get("selected_position") or "").strip().upper()
    if slot in {"IL", "IL+", "IR", "IR+"}:
        return False
    return raw_status not in {"O", "OUT", "INJ", "IL", "IL+", "IR", "IR+"}


def build_game_matrix(players, schedule_df, start_date, end_date, selection):
    """Build one row per player with a Play/Game pair for every date."""
    schedule = schedule_df.copy()
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    date_range = list(pd.date_range(start_ts, end_ts, freq="D"))

    rows = []
    game_lookup = {}
    for player in players:
        name = player.get("name")
        team = str(player.get("team") or "").upper()
        if not name or not team:
            continue

        games = schedule.loc[
            schedule["team"].astype(str).str.upper().eq(team)
            & schedule["date"].between(start_ts, end_ts)
        ].sort_values("date")
        game_by_date = {
            pd.Timestamp(game["date"]).normalize(): game
            for _, game in games.iterrows()
        }
        default_enabled = injury_default_enabled(player)
        player_selection = selection.setdefault(name, {})

        row = {
            "Player": name,
            "Team": team,
            "Pos": player.get("selected_position") or "",
            "Injury": yahoo_injury_designation(player),
            "G": len(games),
            "G#": 0,
        }

        for game_date in date_range:
            date_label = f"{game_date.strftime('%a')} {game_date.month}/{game_date.day}"
            play_col = f"{date_label} ✓"
            game_col = f"{date_label} Game"
            game = game_by_date.get(game_date)

            # No game means no control. None renders as a blank cell instead of
            # suggesting that an unavailable game can be selected.
            if game is None:
                row[play_col] = None
                row[game_col] = ""
                continue

            date_key = game_date.strftime("%Y-%m-%d")
            if date_key not in player_selection:
                player_selection[date_key] = default_enabled
            away = str(game.get("home_away", "")).strip().lower() == "away"
            opponent = str(game.get("opponent", "")).upper()
            enabled = bool(player_selection[date_key])
            row[play_col] = enabled
            row[game_col] = f"@{opponent}" if away else opponent
            row["G#"] += int(enabled)
            game_lookup[(name, play_col)] = date_key

        rows.append(row)

    base_columns = ["Player", "Team", "Pos", "Injury", "G", "G#"]
    date_columns = []
    for game_date in date_range:
        date_label = f"{game_date.strftime('%a')} {game_date.month}/{game_date.day}"
        date_columns.extend([f"{date_label} ✓", f"{date_label} Game"])

    return pd.DataFrame(rows, columns=base_columns + date_columns), game_lookup, date_range


def render_game_selector(label, side_key, players, schedule_df, start_date, end_date):
    selection_key = f"{side_key}_game_selection"
    selection = st.session_state[selection_key]
    matrix, game_lookup, date_range = build_game_matrix(
        players, schedule_df, start_date, end_date, selection
    )

    st.markdown(f"#### {label}")
    button_cols = st.columns(3)
    if button_cols[0].button("Select All Games", key=f"select_all_{side_key}", use_container_width=True):
        for player_dates in selection.values():
            for date_key in player_dates:
                player_dates[date_key] = True
        st.session_state.last_result = None
        st.rerun()
    if button_cols[1].button("Clear All Games", key=f"clear_all_{side_key}", use_container_width=True):
        for player_dates in selection.values():
            for date_key in player_dates:
                player_dates[date_key] = False
        st.session_state.last_result = None
        st.rerun()
    if button_cols[2].button("Restore Injury Defaults", key=f"injury_defaults_{side_key}", use_container_width=True):
        player_by_name = {player.get("name"): player for player in players}
        for player_name, player_dates in selection.items():
            enabled = injury_default_enabled(player_by_name.get(player_name, {}))
            for date_key in player_dates:
                player_dates[date_key] = enabled
        st.session_state.last_result = None
        st.rerun()

    if matrix.empty:
        st.info("此日期範圍沒有可選賽程。")
        return selection

    play_columns = [column for column in matrix.columns if column.endswith(" ✓")]
    game_columns = [column for column in matrix.columns if column.endswith(" Game")]
    existing_game_cells = set(game_lookup)

    column_config = {
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="small"),
        "Pos": st.column_config.TextColumn("Pos", width="small"),
        "Injury": st.column_config.TextColumn(
            "Injury", width="small", help="Yahoo player injury designation; blank means Yahoo has no injury designation."
        ),
        "G": st.column_config.NumberColumn("G", width="small"),
        "G#": st.column_config.NumberColumn("G#", width="small"),
    }
    for play_col in play_columns:
        column_config[play_col] = st.column_config.CheckboxColumn(
            play_col,
            help="Only scheduled games can be selected for Monte Carlo.",
            width="small",
        )
    for game_col in game_columns:
        column_config[game_col] = st.column_config.TextColumn(game_col, width="small")

    editor_key = f"game_matrix_{side_key}_{pd.Timestamp(start_date):%Y%m%d}_{pd.Timestamp(end_date):%Y%m%d}"
    styled_matrix = matrix.style.map(
        style_injury_status,
        subset=["Injury"],
    )
    edited = st.data_editor(
        styled_matrix,
        key=editor_key,
        hide_index=True,
        use_container_width=True,
        height=min(650, 38 * (len(matrix) + 2)),
        disabled=["Player", "Team", "Pos", "Injury", "G", "G#"] + game_columns,
        column_config=column_config,
    )

    # Only actual games are persisted. Blank Game cells have no game_lookup
    # entry, so even if a browser renders a null checkbox affordance it can
    # never become an active simulation game.
    for row_index, original in matrix.reset_index(drop=True).iterrows():
        player_name = original["Player"]
        for play_col in play_columns:
            lookup_key = (player_name, play_col)
            if lookup_key not in existing_game_cells:
                continue
            date_key = game_lookup[lookup_key]
            selection.setdefault(player_name, {})[date_key] = bool(
                edited.iloc[row_index][play_col]
            )

    daily_parts = []
    for game_date in date_range:
        date_label = f"{game_date.strftime('%a')} {game_date.month}/{game_date.day}"
        play_col = f"{date_label} ✓"
        scheduled = sum(
            1 for player_name in matrix["Player"]
            if (player_name, play_col) in existing_game_cells
        )
        selected = sum(
            int(selection.get(player_name, {}).get(game_lookup[(player_name, play_col)], False))
            for player_name in matrix["Player"]
            if (player_name, play_col) in existing_game_cells
        )
        daily_parts.append(f"{date_label}: {selected}/{scheduled}")

    st.caption("Daily selected / scheduled player-games: " + "  |  ".join(daily_parts))
    st.caption(
        "@XXX = away game; XXX = home game. Injury shows Yahoo's designation (Q / D / O etc.); blank means no Yahoo injury designation."
    )
    return selection

# ============================================================
# YAHOO AUTHORIZATION + MATCHUP AUTO LOAD
# ============================================================
st.divider()
st.header("Yahoo Matchup 自動載入")
st.caption("先取得 Yahoo Verify Code。完成後，本次網頁 session 的整個 Yahoo pipeline 共用同一 OAuth token。")
league_input = st.text_input(
    "Yahoo League URL 或 League ID", key="yahoo_league_input",
    placeholder="例如：1486 或 Yahoo league URL",
)
if st.session_state.yahoo_token is None:
    try:
        st.link_button("1. 開啟 Yahoo 授權頁", get_yahoo_authorization_url(),type="primary", use_container_width=True)
    except Exception as error:
        st.error("無法建立 Yahoo 授權連結。")
        st.exception(error)
    verifier_code = st.text_input(
        "2. Yahoo Verify Code", key="yahoo_verifier_code",
        type="password", placeholder="貼上 Yahoo 顯示的 code",
    )
    if st.button("完成 Yahoo 授權", type="primary", use_container_width=True,
                 disabled=not bool(verifier_code.strip())):
        try:
            st.session_state.yahoo_token = exchange_yahoo_verifier(
                verifier_code
            )
            
            st.rerun()
        except Exception as error:
            st.session_state.yahoo_token = None
            st.error("Verify Code 交換 token 失敗，請重新取得 code。")
            st.exception(error)
else:
    auth_col, logout_col = st.columns([3, 1])
    with auth_col:
        st.success("Yahoo 已授權。所有 Yahoo API 呼叫共用同一 token，過期時自動 refresh。")
    with logout_col:
        if st.button("登出 Yahoo", use_container_width=True):
            st.session_state.yahoo_token = None
            st.session_state.yahoo_context = None
            st.session_state.yahoo_loaded = False
            st.session_state.yahoo_matchup_loaded = False
            st.session_state.yahoo_history_loaded = False
            st.session_state.yahoo_current_totals_loaded = False
            st.session_state.selected_my_team_key = None
            st.session_state.selected_opp_team_key = None
            invalidate_loaded_rosters()
            st.rerun()
    if st.button(
        "1. 載入 Yahoo League",
        type="primary",
        use_container_width=True,
        disabled=not bool(league_input.strip()),
    ):
        try:
            with st.spinner("正在載入 League teams、預設 matchup、雙方 rosters 與目前累積數據..."):
                yc = load_yahoo_matchup_context(league_input.strip())

            st.session_state.yahoo_context = yc
            st.session_state.yahoo_loaded = True
            st.session_state.yahoo_matchup_loaded = True
            st.session_state.yahoo_history_loaded = False
            st.session_state.selected_my_team_key = yc["my_team_key"]
            st.session_state.selected_opp_team_key = yc["opponent_team_key"]
            st.session_state.my_team = [
                player["name"] for player in yc["my_roster"]
            ]
            st.session_state.opp_team = [
                player["name"] for player in yc["opponent_roster"]
            ]

            # New matchup means any previously built historical roster is stale.
            invalidate_loaded_rosters()

            current_stat_fields = [
                "FGA", "FGM", "FTA", "FTM", "3PM",
                "Pts", "Reb", "Ast", "Stl", "Blk", "TO",
            ]
            for stat in current_stat_fields:
                st.session_state[f"current_my_{stat}"] = int(
                    yc["current_my"].get(stat, 0)
                )
                st.session_state[f"current_opp_{stat}"] = int(
                    yc["current_opp"].get(stat, 0)
                )

            st.session_state.yahoo_current_totals_loaded = True
            st.session_state.last_result = None
            st.rerun()
        except Exception as error:
            st.session_state.yahoo_loaded = False
            st.session_state.yahoo_matchup_loaded = False
            st.error("Yahoo matchup 輕量載入失敗。")
            st.exception(error)

    if st.session_state.yahoo_matchup_loaded:
        st.success("Yahoo League、預設 matchup、雙方陣容與目前累積數據已載入。")

        history_button_label = (
            "2. 重新載入雙方球員歷史資料"
            if st.session_state.yahoo_history_loaded
            else "2. 載入雙方球員歷史資料"
        )
        if st.button(
            history_button_label,
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.yahoo_matchup_loaded,
        ):
            try:
                with st.spinner(
                    "正在載入 2022-23 至 2026-27 歷史資料；"
                    "已快取資料會直接讀取，只補齊缺少部分..."
                ):
                    historical = build_yahoo_historical_rosters(
                        st.session_state.yahoo_context,
                        seasons=tuple(SEASONS),
                    )

                st.session_state.my_roster_data = (
                    historical["my_roster_engine"]
                )
                st.session_state.opp_roster_data = (
                    historical["opponent_roster_engine"]
                )
                st.session_state.rosters_loaded = True
                st.session_state.yahoo_history_loaded = True
                st.session_state.last_result = None
                st.rerun()
            except Exception as error:
                st.session_state.yahoo_history_loaded = False
                invalidate_loaded_rosters()
                st.error("Yahoo 歷史資料載入失敗。")
                st.exception(error)

if st.session_state.yahoo_loaded and st.session_state.yahoo_context:
    yc = st.session_state.yahoo_context
    st.info(f"已連接 League {yc['public_league_id']} | API key: {yc['league_key']} | Week {yc['current_week']}")

# ============================================================
# LEAGUE MATCHUP SELECTOR + ROSTER EDITOR
# ============================================================
st.divider()
st.header("Matchup 雙方陣容")
st.caption(
    "從 Yahoo League 選擇任意兩支隊伍。切換隊伍後會更新 roster 與本週累積數據，"
    "並清除舊的 historical roster，避免沿用上一組 matchup。"
)
context = st.session_state.get("yahoo_context")
if isinstance(context, dict) and context.get("league_teams"):
    league_teams = context["league_teams"]
    team_name_by_key = {
        team["team_key"]: team["name"] for team in league_teams
    }
    team_keys = list(team_name_by_key)
    if st.session_state.selected_my_team_key not in team_keys:
        st.session_state.selected_my_team_key = context.get("my_team_key")
    if st.session_state.selected_opp_team_key not in team_keys:
        st.session_state.selected_opp_team_key = context.get("opponent_team_key")

    matchup_left, vs_col, matchup_right = st.columns([4, 0.7, 4])
    with matchup_left:
        selected_my_key = st.selectbox(
            "本陣",
            options=team_keys,
            format_func=lambda key: team_name_by_key[key],
            key="selected_my_team_key",
        )
    with vs_col:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-weight:700;'>VS</div>", unsafe_allow_html=True)
    with matchup_right:
        selected_opp_key = st.selectbox(
            "敵陣",
            options=team_keys,
            format_func=lambda key: team_name_by_key[key],
            key="selected_opp_team_key",
        )

    if selected_my_key == selected_opp_key:
        st.warning("本陣與敵陣不能選擇同一支 Yahoo Team。")
    elif (
        selected_my_key != context.get("my_team_key")
        or selected_opp_key != context.get("opponent_team_key")
    ):
        synchronize_selected_matchup(selected_my_key, selected_opp_key)

    my_col, opp_col = st.columns(2)
    with my_col:
        show_team_editor("my_team", "本陣")
    with opp_col:
        show_team_editor("opp_team", "敵陣")
else:
    st.info("請先完成 Yahoo 授權並載入 League。")

# ============================================================
# ROSTER SUMMARY
# ============================================================
st.divider()

Oneal_PATH = APP_FOLDER / "O_neal.png"
Garnett_PATH = APP_FOLDER / "Garnett.png"
James_PATH = APP_FOLDER / "James.png"
Jordan_PATH = APP_FOLDER / "Jordan.png"
Bryant_PATH = APP_FOLDER / "Bryant.png"

col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])

with col2:
    st.image(
        str(Bryant_PATH),
        width=92
    )
with col3:
    st.image(
        str(Jordan_PATH),
        width=95
    )
with col4:
    st.image(
        str(James_PATH),
        width=99
    )
with col5:
    st.image(
        str(Garnett_PATH),
        width=104
    )
with col6:
    st.image(
        str(Oneal_PATH),
        width=107
    )

st.header("陣型")
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.metric("My Team Players", len(st.session_state.my_team))
with metric_col2:
    st.metric("Opponent Players", len(st.session_state.opp_team))

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.markdown("#### 我方")
    st.dataframe(
        roster_to_dataframe(
            st.session_state.my_team,
            yahoo_roster_metadata("my_team"),
        ),
        hide_index=True,
        use_container_width=True,
    )
with summary_col2:
    st.markdown("#### 敵方")
    st.dataframe(
        roster_to_dataframe(
            st.session_state.opp_team,
            yahoo_roster_metadata("opp_team"),
        ),
        hide_index=True,
        use_container_width=True,
    )

# ============================================================
# HISTORICAL DATA STATUS (YAHOO ONLY)
# ============================================================
st.divider()
st.header("Yahoo 歷史資料狀態")
if st.session_state.rosters_loaded:
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"My Team Yahoo history: {len(st.session_state.my_roster_data)} players")
    with c2:
        st.success(f"Opponent Yahoo history: {len(st.session_state.opp_roster_data)} players")
elif st.session_state.yahoo_matchup_loaded:
    st.info("Matchup 已載入；請按「2. 載入雙方球員歷史資料」以啟用模擬。")
else:
    st.info("請先連接 Yahoo 並載入 League。")

# ============================================================
# MATCHUP SETTINGS
# ============================================================
st.divider()
st.header("Matchup Settings")
start_col, end_col, sim_col = st.columns(3)
with start_col:
    start_date = st.date_input("Fantasy Week Start Date")
with end_col:
    end_date = st.date_input("Fantasy Week End Date")
with sim_col:
    n_sim = st.number_input(
        "Monte Carlo Simulations",
        min_value=100,
        max_value=100000,
        value=3000,
        step=100,
    )

st.markdown("### Scheduled Games Selection")
st.caption(
    "依所選日期與球員目前球隊列出賽程。只有勾選的 player-games 會進入 Monte Carlo；"
    "Yahoo Injury 狀態只作參考與初始預設。"
)
if SCHEDULE_PATH.exists() and isinstance(st.session_state.get("yahoo_context"), dict):
    selection_schedule = load_schedule(str(SCHEDULE_PATH))
    context = st.session_state.yahoo_context
    signature = (
        str(start_date), str(end_date),
        context.get("my_team_key"), context.get("opponent_team_key"),
        tuple(st.session_state.my_team), tuple(st.session_state.opp_team),
    )
    if st.session_state.game_selection_signature != signature:
        st.session_state.my_game_selection = {}
        st.session_state.opp_game_selection = {}
        st.session_state.game_selection_signature = signature
        st.session_state.last_result = None
    game_tab_my, game_tab_opp = st.tabs(["本陣賽程", "敵陣賽程"])
    with game_tab_my:
        my_selected_games = render_game_selector(
            "本陣", "my", context.get("my_roster", []),
            selection_schedule, start_date, end_date,
        )
    with game_tab_opp:
        opp_selected_games = render_game_selector(
            "敵陣", "opp", context.get("opponent_roster", []),
            selection_schedule, start_date, end_date,
        )
    fingerprint = repr((my_selected_games, opp_selected_games))
    if (
        st.session_state.game_selection_fingerprint is not None
        and st.session_state.game_selection_fingerprint != fingerprint
    ):
        st.session_state.last_result = None
    st.session_state.game_selection_fingerprint = fingerprint
else:
    my_selected_games = {}
    opp_selected_games = {}
    st.info("載入 Yahoo League 後會顯示雙方球員賽程選擇。")

st.markdown("### 當前各項累積數據")
st.caption("Yahoo 載入後會自動填入目前累積值；FG% 與 FT% 由 FGM/FGA、FTM/FTA 計算。")
stat_columns = ["FGA", "FGM", "FTA", "FTM", "3PM", "Pts", "Reb", "Ast", "Stl", "Blk", "TO"]


if st.session_state.get("yahoo_current_totals_loaded"):
    st.success("Yahoo 本週目前累積數據已自動填入。你仍可手動修改。")

my_totals_col, opp_totals_col = st.columns(2)

with my_totals_col:
    st.markdown("#### 目前我方各項成績")
    current_my = {}
    for stat in stat_columns:
        current_my[stat] = st.number_input(
            f"My {stat}", min_value=0, step=1, key=f"current_my_{stat}"
        )

with opp_totals_col:
    st.markdown("#### 目前對方各項成績")
    current_opp = {}
    for stat in stat_columns:
        current_opp[stat] = st.number_input(
            f"Opponent {stat}", min_value=0, step=1, key=f"current_opp_{stat}"
        )

# ============================================================
# ADD / DROP SCENARIO
# ============================================================

st.divider()
st.header("Add / Drop Scenario")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    add_player = st.text_input(
        "Add Player",
        placeholder="例如：Michael Jordan; 若沒有添加球員,釋出球員依然成立。"
    )

with scenario_col2:
    if st.session_state.my_team:
        drop_player = st.selectbox(
            "Drop Player",
            options=st.session_state.my_team
        )
    else:
        drop_player = ""
        st.info("請先選擇並載入本陣。")


# ============================================================
# SESSION STATE FOR SAVED MONTE CARLO RESULT
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_scenario_message" not in st.session_state:
    st.session_state.last_scenario_message = None


# ============================================================
# RUN BUTTON
# ============================================================

run_scenario = st.button(
    "對球員變動狀態進行蒙地卡羅模擬",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN MONTE CARLO ONLY WHEN BUTTON IS PRESSED
# ============================================================

if run_scenario:

    cleaned_add_player = clean_player_name(
        add_player
    )

    errors = []

    if not st.session_state.rosters_loaded:
        errors.append(
            "請先按下「2. 載入雙方球員歷史資料」。"
        )

    # Add Player 可以留空

    if not drop_player:
        errors.append(
            "請選擇 Drop Player。"
        )

    if start_date > end_date:
        errors.append(
            "Fantasy Week Start Date 不能晚於 End Date。"
        )

    if not SCHEDULE_PATH.exists():
        errors.append(
            "找不到 schedule.csv。"
        )

    if cleaned_add_player:

        if cleaned_add_player in st.session_state.my_team:
            errors.append(
                f"{cleaned_add_player} 已經在 My Team 裡。"
            )

        if cleaned_add_player in st.session_state.opp_team:
            errors.append(
                f"{cleaned_add_player} 目前在 Opponent Team 裡。"
            )
        
    if errors:

        for message in errors:
            st.error(message)

    else:

        try:

            with st.spinner(
                "Reading schedule and running Monte Carlo..."
            ):

                schedule_df = load_schedule(
                    str(SCHEDULE_PATH)
                )

                new_result = compare_add_drop_fast(
                    baseline_roster=(
                        st.session_state.my_roster_data
                    ),
                    opp_roster=(
                        st.session_state.opp_roster_data
                    ),
                    schedule_df=schedule_df,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    current_my=current_my,
                    current_opp=current_opp,
                    add_player=cleaned_add_player,
                    drop_player=drop_player,
                    seasons=SEASONS,
                    n_sim=int(n_sim),
                    random_seed=12345,
                    yahoo_league_key=st.session_state.yahoo_context["league_key"],
                    my_selected_games=st.session_state.my_game_selection,
                    opp_selected_games=st.session_state.opp_game_selection,
                )

            # Only replace the saved result after a successful run.
            st.session_state.last_result = new_result

            st.session_state.last_scenario_message = (
                f"Simulation completed: "
                f"Add {cleaned_add_player}, "
                f"Drop {drop_player}"
            )

        except Exception as error:

            st.error(
                "Monte Carlo scenario 執行失敗。"
            )

            st.exception(error)


# ============================================================
# DISPLAY SAVED RESULT
#
# This is intentionally OUTSIDE if run_scenario.
# Therefore selectbox changes do not erase the result or
# rerun the Monte Carlo simulation.
# ============================================================

if st.session_state.last_result is not None:

    result = st.session_state.last_result

    if st.session_state.last_scenario_message:

        st.success(
            st.session_state.last_scenario_message
        )

    # ========================================================
    # OVERALL PROBABILITY METRICS
    # ========================================================

    overall = (
        result["overall_comparison"]
        .set_index("Target")
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:

        st.metric(
            "P(Score ≥ 5) after Add/Drop",
            (
                f"{overall.loc['P(>=5)', 'AddDrop'] * 100:.1f}%"
            ),
            delta=(
                f"{overall.loc['P(>=5)', 'Delta'] * 100:+.1f} pp"
            ),
        )

    with result_col2:

        st.metric(
            "P(Score ≥ 6) after Add/Drop",
            (
                f"{overall.loc['P(>=6)', 'AddDrop'] * 100:.1f}%"
            ),
            delta=(
                f"{overall.loc['P(>=6)', 'Delta'] * 100:+.1f} pp"
            ),
        )

    with result_col3:

        st.metric(
            "P(Score ≥ 7) after Add/Drop",
            (
                f"{overall.loc['P(>=7)', 'AddDrop'] * 100:.1f}%"
            ),
            delta=(
                f"{overall.loc['P(>=7)', 'Delta'] * 100:+.1f} pp"
            ),
        )

    # ========================================================
    # CATEGORY TABLE
    # ========================================================

    st.markdown(
        "### 各項目 勝率變化"
    )

    category_display = (
        result["category_comparison"]
        .copy()
    )
    category_order = [
        "FG%",
        "FT%",
        "3PM",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "TO",
    ]
    
    category_display["Category"] = pd.Categorical(
        category_display["Category"],
        categories=category_order,
        ordered=True,
    )
    
    category_display = (
        category_display
        .sort_values("Category")
        .reset_index(drop=True)
    )
    for column in [
        "Baseline",
        "AddDrop",
        "Delta",
    ]:

        category_display[column] = (
            category_display[column] * 100
        ).round(1)

    st.dataframe(
        category_display,
        hide_index=True,
        use_container_width=True,
    )

    # ========================================================
    # EXPECTED PLAYER PRODUCTION
    # ========================================================
    st.markdown("### 預估球員各項目輸出")

    production_view = st.radio(
        "Production View",
        options=["Matchup Total", "Per Game Average"],
        horizontal=True,
        key="expected_production_view",
        label_visibility="collapsed",
    )

    if production_view == "Per Game Average":
        st.caption(
            "Average projected per-game production across all simulations for the remaining matchup."
        )
    else:
        st.caption(
            "Average projected total production across all simulations for the remaining matchup."
        )

    expected_production = result.get("expected_player_production")
    if expected_production is not None and not expected_production.empty:
        production_display = expected_production[
            ["Player", "Scheduled Games", "Active Games", "FG%", "FT%", "3PM", "Pts", "Reb", "Ast", "Stl", "Blk", "TO"]
        ].copy()

        counting_columns = ["3PM", "Pts", "Reb", "Ast", "Stl", "Blk", "TO"]
        if production_view == "Per Game Average":
            active_games = pd.to_numeric(
                production_display["Active Games"], errors="coerce"
            ).fillna(0)
            safe_active_games = active_games.where(active_games > 0)
            production_display[counting_columns] = (
                production_display[counting_columns]
                .div(safe_active_games, axis=0)
                .fillna(0.0)
            )

        # Shooting percentages are ratios and never divided by Active Games.
        production_display["FG%"] = (production_display["FG%"] * 100).round(1)
        production_display["FT%"] = (production_display["FT%"] * 100).round(1)
        for column in counting_columns:
            production_display[column] = production_display[column].round(2)

        st.dataframe(
            production_display,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Show Shooting Volume Details", expanded=False):
            shooting_display = expected_production[
                ["Player", "FGM", "FGA", "FTM", "FTA", "Active Games"]
            ].copy()
            shooting_columns = ["FGM", "FGA", "FTM", "FTA"]
            if production_view == "Per Game Average":
                active_games = pd.to_numeric(
                    shooting_display["Active Games"], errors="coerce"
                ).fillna(0)
                safe_active_games = active_games.where(active_games > 0)
                shooting_display[shooting_columns] = (
                    shooting_display[shooting_columns]
                    .div(safe_active_games, axis=0)
                    .fillna(0.0)
                )
            shooting_display = shooting_display.drop(columns=["Active Games"])
            for column in shooting_columns:
                shooting_display[column] = shooting_display[column].round(2)
            st.dataframe(
                shooting_display,
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Expected player production was not generated.")

    # ========================================================
    # OVERALL TABLE
    # ========================================================

    st.markdown(
        "### 得到N分以上的機率"
    )

    overall_display = (
        result["overall_comparison"]
        .copy()
    )

    for column in [
        "Baseline",
        "AddDrop",
        "Delta",
    ]:

        overall_display[column] = (
            overall_display[column] * 100
        ).round(1)

    st.dataframe(
        overall_display,
        hide_index=True,
        use_container_width=True,
    )

    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    st.markdown(
        "### H2H 得分機率分佈"
    )

    score_dist = result.get(
        "score_distribution_comparison"
    )


    if (
        score_dist is not None
        and not score_dist.empty
    ):

        score_dist = score_dist.copy()

        score_order = [
            f"{wins}:{9 - wins}"
            for wins in range(10)
        ]

        score_dist["Score"] = pd.Categorical(
            score_dist["Score"],
            categories=score_order,
            ordered=True
        )

        score_dist = (
            score_dist
            .sort_values(
                [
                    "Score",
                    "Scenario"
                ]
            )
            .reset_index(drop=True)
        )

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        sns.barplot(
            data=score_dist,
            x="Score",
            y="Probability",
            hue="Scenario",
            hue_order=[
                "Baseline",
                "Add / Drop"
            ],
            palette={
                "Baseline": "#9333EA",
                "Add / Drop": "#FBBF24"
            },
            ax=ax
        )

        for container in ax.containers:

            labels = [
                f"{value*100:.1f}%"
                if value > 0.001
                else ""
                for value in container.datavalues
            ] 

            ax.bar_label(
                container,
                labels=labels,
                padding=2,
                fontsize=6
            )
        
        ax.set_title(
            "Final Matchup H2H Score Distribution",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Final Category Score"
        )

        ax.set_ylabel(
            "Probability"
        )

        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(
                xmax=1.0
            )
        )

        ax.legend(
            title=""
        )

        ax.grid(
            axis="y",
            alpha=0.20
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # Optional numerical table
        score_table = (
            score_dist[
                [
                    "Score",
                    "Scenario",
                    "Count",
                    "Probability"
                ]
            ]
            .copy()
        )

        score_table[
            "Probability (%)"
        ] = (
            score_table["Probability"]
            * 100
        ).round(2)

        score_table = score_table.drop(
            columns=[
                "Probability"
            ]
        )

        with st.expander(
            "Show Score Distribution Table",
            expanded=False
        ):

            st.dataframe(
                score_table,
                hide_index=True,
                use_container_width=True
            )

    else:

        st.warning(
            "Score distribution was not generated."
        )

    # ========================================================
    # HEATMAP
    # ========================================================

    st.markdown(
        "### 蒙地卡羅熱圖分析"
    )

    heatmap_fig = result.get(
        "heatmap_fig"
    )

    if heatmap_fig is not None:

        st.pyplot(
            heatmap_fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Heatmap figure was not generated."
        )

    # ========================================================
    # MONTE CARLO SIMULATION INSPECTION
    # ========================================================
    
    st.markdown(
        "### 蒙地卡羅模擬觀測"
    )
    
    st.caption(
        "隨機顯示 7 次模擬。團隊表格包含目前已累積數字；"
        "球員 breakdown 只包含未來尚未進行的比賽。"
    )
    
    baseline_debug = result.get(
        "baseline_debug"
    )
    
    scenario_debug = result.get(
        "scenario_debug"
    )
    
    opp_debug = result.get(
        "opp_debug"
    )
    
    if (
        baseline_debug is not None
        and scenario_debug is not None
        and opp_debug is not None
    ):
    
        # ====================================================
        # MY TEAM TOTALS
        # ====================================================
    
        st.markdown(
            "#### My Team Totals: Baseline vs Add / Drop"
        )
    
        baseline_team_sample = (
            baseline_debug["team_sample"]
            .copy()
        )
    
        baseline_team_sample.insert(
            1,
            "Scenario",
            "Baseline"
        )
    
        scenario_team_sample = (
            scenario_debug["team_sample"]
            .copy()
        )
    
        scenario_team_sample.insert(
            1,
            "Scenario",
            "Add / Drop"
        )
    
        team_debug_display = pd.concat(
            [
                baseline_team_sample,
                scenario_team_sample
            ],
            ignore_index=True
        )
    
        team_debug_display = (
            team_debug_display
            .sort_values(
                [
                    "Simulation",
                    "Scenario"
                ]
            )
            .reset_index(drop=True)
        )
    
        st.dataframe(
            team_debug_display,
            hide_index=True,
            use_container_width=True
        )
    
        # ====================================================
        # OPPONENT TOTALS
        # ====================================================
    
        st.markdown(
            "#### Opponent Team Totals"
        )
    
        st.dataframe(
            opp_debug["team_sample"],
            hide_index=True,
            use_container_width=True
        )
    
        # ====================================================
        # PLAYER DATA PREPARATION
        # ====================================================
    
        player_debug_display = (
            baseline_debug["player_sample"]
            .copy()
        )
        
        player_debug_display.insert(
            1,
            "Tag",
            ""
        )
        player_debug_display.loc[
            player_debug_display["Player"] == drop_player,
            "Tag"
        ] = "DROP"

        if add_player:

            add_rows = (
                scenario_debug["player_sample"]
                .loc[
                    scenario_debug["player_sample"]["Player"]
                    == add_player
                ]
                .copy()
            )
        
            add_rows.insert(
                1,
                "Tag",
                "ADD"
            )
        
            player_debug_display = pd.concat(
                [
                    player_debug_display,
                    add_rows
                ],
                ignore_index=True
            )
        player_debug_display["TagOrder"] = (
            player_debug_display["Tag"]
            .map(
                {
                    "DROP": 0,
                    "ADD": 1,
                    "": 2
                }
            )
        )
        
        player_debug_display = (
            player_debug_display
            .sort_values(
                [
                    "Simulation",
                    "TagOrder",
                    "Player"
                ]
            )
            .drop(
                columns="TagOrder"
            )
            .reset_index(drop=True)
        )
    
        opp_player_debug_display = (
            opp_debug["player_sample"]
            .copy()
        )
    
        opp_player_debug_display = (
            opp_player_debug_display
            .sort_values(
                [
                    "Simulation",
                    "Player"
                ]
            )
            .reset_index(drop=True)
        )
    
        # ====================================================
        # SIMULATION EXPLORER
        # ====================================================
    
        with st.expander(
            "🔬 Simulation Explorer | My Team vs Opponent",
            expanded=False
        ):
    
            simulation_options = sorted(
                player_debug_display[
                    "Simulation"
                ].unique()
            )
    
            selected_simulation = st.selectbox(
                "Select Simulation ID",
                options=simulation_options,
                key="debug_simulation_selection"
            )
    
            my_selected_breakdown = (
                player_debug_display[
                    player_debug_display[
                        "Simulation"
                    ]
                    == selected_simulation
                ]
                .copy()
            )
    
            opp_selected_breakdown = (
                opp_player_debug_display[
                    opp_player_debug_display[
                        "Simulation"
                    ]
                    == selected_simulation
                ]
                .copy()
            )
    
            left_col, right_col = st.columns(2)
    
            with left_col:
    
                st.markdown(
                    "### 🟣 My Team"
                )
    
                st.dataframe(
                    my_selected_breakdown,
                    hide_index=True,
                    use_container_width=True
                )
    
            with right_col:
    
                st.markdown(
                    "### 🔴 Opponent Team"
                )
    
                st.dataframe(
                    opp_selected_breakdown,
                    hide_index=True,
                    use_container_width=True
                )
    
            st.markdown(
                "#### 七次模擬抽樣：My Team"
            )
    
            st.dataframe(
                player_debug_display,
                hide_index=True,
                use_container_width=True,
                height=760
            )
    
            st.markdown(
                "#### 七次模擬抽樣：Opponent Team"
            )
    
            st.dataframe(
                opp_player_debug_display,
                hide_index=True,
                use_container_width=True,
                height=760
            )
    
    else:
    
        st.warning(
            "Monte Carlo debug sample was not generated."
        )

Jersey_PATH = APP_FOLDER / "日下清十郎.png"
Shoe_PATH = APP_FOLDER / "Air_Jordan_1.png"
Basketball_PATH = APP_FOLDER / "Basketball.png"

col1, col2, col3, col4, col5, col6, col7 = st.columns([3,1,0.8,1.7,1,1,3])

with col4:
    st.image(
        str(Jersey_PATH),
        width=360
    )
with col3:
    st.image(
        str(Basketball_PATH),
        width=90
    )
with col5:
    st.image(
        str(Shoe_PATH),
        width=120
    )


st.markdown(
    """

௹ѪΩ€Ξ     ҉҉҉    『Δικαιοσύνη』    •    『Προμηθεύς』    •    『Ολτρεμάρε』    •    『Φαντασία』    •    『Καλαθοσφαίρισης』    •    『Μυστική』    •    『Εταιρεία』       •     『Βερίτας』    ҉҉҉    £ΩѦѤӁ


"""
)


