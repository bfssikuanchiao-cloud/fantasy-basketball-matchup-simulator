from pathlib import Path
import re
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from fantasy_engine import SEASONS, build_roster, load_schedule, compare_add_drop_fast

# ============================================================
# PAGE SETTINGS
# ============================================================

APP_FOLDER = Path(__file__).parent
LOGO_PATH = APP_FOLDER / "秘密結社.png"
LOGO1_PATH = APP_FOLDER / "秘密結社1.png"
SCHEDULE_PATH = APP_FOLDER / "schedule.csv"



st.set_page_config(
    page_title="秘密結社夢幻籃球模擬器",
    page_icon=str(LOGO_PATH),
    layout="wide",
)

col1, col2, col3, col4, col5, col6 = st.columns([0.3,2,1,1,1,1])

with col2:
    st.image(
        str(LOGO1_PATH),
        width=400
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
- Yahoo Fantasy API integration（coming soon）




௹Ѫ€Ξ    •    『Βερίτας』    •    『Προμηθεύς』    •    『Ολτρεμάρε』    •    『Δικαιοσύνη』    •    『Φαντασία』    •    『Καλαθοσφαίρισης』    •    『Μυστική』    •    『Εταιρεία』    •    £ѦѤӁ

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
    "my_paste_text": "",
    "opp_paste_text": "",
    "last_result": None,
    "last_scenario_message": None,
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


def roster_to_dataframe(team_list):
    return pd.DataFrame(
        {
            "No.": range(1, len(team_list) + 1),
            "Player": team_list,
        }
    )


ROSTER_SLOTS = {
    "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTL",
    "BN", "BENCH", "IL", "IL+", "IR", "IR+"
}

NOISE_EXACT = {
    "PLAYER NOTE", "NEW PLAYER NOTE", "NO NEW PLAYER NOTES",
    "EDIT ROSTER", "START ACTIVE PLAYERS", "BENCH",
}

TEAM_DETAIL_PATTERN = re.compile(
    r"^[A-Z]{2,3}\s*-\s*(?:PG|SG|SF|PF|C|G|F)(?:\s*,\s*(?:PG|SG|SF|PF|C|G|F))*$",
    re.IGNORECASE,
)

GAME_TIME_PATTERN = re.compile(
    r"^(?:\d{1,2}:\d{2}\s*(?:am|pm)|@\s*[A-Z]{2,3}|vs\.?\s*[A-Z]{2,3})",
    re.IGNORECASE,
)

STATUS_PATTERN = re.compile(r"^(?:Q|O|DTD|GTD|INJ|SUSP|NA)$", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


def is_probable_player_name(line):
    text = line.strip()
    upper = text.upper()

    if not text or upper in ROSTER_SLOTS or upper in NOISE_EXACT:
        return False
    if URL_PATTERN.search(text):
        return False
    if TEAM_DETAIL_PATTERN.match(text):
        return False
    if GAME_TIME_PATTERN.match(text):
        return False
    if STATUS_PATTERN.match(text):
        return False
    if "PLAYER NOTE" in upper or "FANTASY" in upper:
        return False
    if re.search(r"\d", text):
        return False

    # NBA player names are generally two to five alphabetic name parts.
    parts = text.replace("'", "").replace("-", " ").split()
    if not 2 <= len(parts) <= 5:
        return False
    if not all(part.replace(".", "").isalpha() for part in parts):
        return False

    return True


def parse_roster_text(raw_text):
    """Parse either one-player-per-line text or copied Yahoo roster text."""
    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Also support comma-separated player names when no Yahoo slot structure exists.
    if len(raw_lines) == 1 and "," in raw_lines[0]:
        raw_lines = [item.strip() for item in raw_lines[0].split(",") if item.strip()]

    players = []
    seen = set()
    waiting_for_player = False

    for line in raw_lines:
        upper = line.upper().rstrip(":")

        if upper in ROSTER_SLOTS:
            waiting_for_player = True
            continue

        if waiting_for_player:
            if is_probable_player_name(line):
                player = clean_player_name(line)
                if player not in seen:
                    players.append(player)
                    seen.add(player)
                waiting_for_player = False
            continue

        # Simple mode: permit clean one-player-per-line lists.
        if is_probable_player_name(line):
            player = clean_player_name(line)
            if player not in seen:
                players.append(player)
                seen.add(player)

    return players


@st.cache_data(show_spinner=False)
def load_roster_from_nba(player_names_tuple, seasons_tuple):
    return build_roster(list(player_names_tuple), seasons=seasons_tuple)


def apply_pasted_roster(text_key, team_key, other_team_key):
    parsed_players = parse_roster_text(st.session_state[text_key])

    if not parsed_players:
        st.warning(
            "沒有辨識到球員。請嘗試每行只放一名球員，或保留 Yahoo roster 的位置與球員名稱。"
        )
        return

    other_team = set(st.session_state[other_team_key])
    duplicates = [player for player in parsed_players if player in other_team]
    accepted = [player for player in parsed_players if player not in other_team]

    st.session_state[team_key] = accepted
    invalidate_loaded_rosters()

    if duplicates:
        st.warning(
            "下列球員已在另一支球隊，因此未匯入：" + ", ".join(duplicates)
        )

    st.success(f"成功匯入 {len(accepted)} 名球員。")
    st.rerun()


def show_paste_team(team_key, title, text_key, other_team_key):
    team = st.session_state[team_key]
    st.subheader(f"{title} ({len(team)})")

    st.text_area(
        f"Paste {title}",
        key=text_key,
        height=280,
        placeholder=(
            "可貼 Yahoo roster，例如：\n"
            "SG\nKobe Bryant\nLAL - SG\n"
            "C\nShaquille O'Neal\nLAL - C\n\n"
            "也可以每行只貼一名球員。"
        ),
    )

    import_col, erase_col, clear_col = st.columns(3)

    with import_col:
        if st.button(
            f"Import {title}",
            key=f"import_{team_key}",
            type="primary",
            use_container_width=True,
        ):
            apply_pasted_roster(text_key, team_key, other_team_key)

    with erase_col:
        if team:
            remove_player = st.selectbox(
                "Erase player",
                options=team,
                key=f"remove_select_{team_key}",
                label_visibility="collapsed",
            )
            if st.button(
                "Erase Selected",
                key=f"erase_selected_{team_key}",
                use_container_width=True,
            ):
                st.session_state[team_key].remove(remove_player)
                invalidate_loaded_rosters()
                st.rerun()
        else:
            st.button(
                "Erase Selected",
                key=f"erase_disabled_{team_key}",
                disabled=True,
                use_container_width=True,
            )

    with clear_col:
        if st.button(
            f"Clear {title}",
            key=f"clear_{team_key}",
            disabled=not bool(team),
            use_container_width=True,
        ):
            st.session_state[team_key] = []
            st.session_state[text_key] = ""
            invalidate_loaded_rosters()
            st.rerun()

    st.markdown(f"#### 目前 {title}")
    if team:
        st.dataframe(
            roster_to_dataframe(team),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(f"{title} 目前尚未匯入球員。")

# ============================================================
# PASTE ROSTERS
# ============================================================
st.divider()
st.header("輸入球員陣容")
st.caption(
    "可直接貼 Yahoo Roster，也可以使用每行一名球員的簡單格式。匯入會取代該隊目前名單。"
)

my_col, opp_col = st.columns(2)

with my_col:
    show_paste_team(
        team_key="my_team",
        title="本陣",
        text_key="my_paste_text",
        other_team_key="opp_team",
    )

with opp_col:
    show_paste_team(
        team_key="opp_team",
        title="敵陣",
        text_key="opp_paste_text",
        other_team_key="my_team",
    )

# ============================================================
# ROSTER SUMMARY
# ============================================================
st.divider()
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
        roster_to_dataframe(st.session_state.my_team),
        hide_index=True,
        use_container_width=True,
    )
with summary_col2:
    st.markdown("#### 敵方")
    st.dataframe(
        roster_to_dataframe(st.session_state.opp_team),
        hide_index=True,
        use_container_width=True,
    )

# ============================================================
# LOAD NBA DATA
# ============================================================
st.divider()
st.header("載入NBA自2018至2027球季比賽數據")
st.write("按下按鈕後，程式會透過 NBA API 載入兩隊球員的賽季逐場比賽紀錄。")

if st.button("輸入雙方陣容", type="primary", use_container_width=True):
    if not st.session_state.my_team:
        st.error("本陣尚未匯入任何球員。")
    elif not st.session_state.opp_team:
        st.error("敵陣尚未匯入任何球員。")
    else:
        try:
            with st.spinner("Loading My Team multi-season data..."):
                st.session_state.my_roster_data = load_roster_from_nba(
                    tuple(st.session_state.my_team), tuple(SEASONS)
                )
            with st.spinner("Loading Opponent Team multi-season data..."):
                st.session_state.opp_roster_data = load_roster_from_nba(
                    tuple(st.session_state.opp_team), tuple(SEASONS)
                )
            st.session_state.rosters_loaded = True
            st.success("Both rosters were loaded successfully.")
        except Exception as error:
            invalidate_loaded_rosters()
            st.error("載入 roster 時發生錯誤。")
            st.exception(error)

if st.session_state.rosters_loaded:
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.success(f"My Team loaded: {len(st.session_state.my_roster_data)} players")
    with status_col2:
        st.success(f"Opponent loaded: {len(st.session_state.opp_roster_data)} players")
else:
    st.info("陣容尚未載入 NBA multi-season data。")

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

st.markdown("### 當前各項累積數據")
st.caption("FG% 與 FT% 不直接輸入百分比，請輸入 FGM、FGA、FTM、FTA。")
stat_columns = ["Pts", "Reb", "Ast", "Stl", "Blk", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
my_totals_col, opp_totals_col = st.columns(2)

with my_totals_col:
    st.markdown("#### 目前我方各項成績")
    current_my = {}
    for stat in stat_columns:
        current_my[stat] = st.number_input(
            f"My {stat}", min_value=0, value=0, step=1, key=f"current_my_{stat}"
        )

with opp_totals_col:
    st.markdown("#### 目前對方各項成績")
    current_opp = {}
    for stat in stat_columns:
        current_opp[stat] = st.number_input(
            f"Opponent {stat}", min_value=0, value=0, step=1, key=f"current_opp_{stat}"
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
        st.info("請先匯入 My Team。")


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
            "請先按下 Load Both Rosters。"
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
        "### Category Win Probability Comparison"
    )

    category_display = (
        result["category_comparison"]
        .copy()
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
    # OVERALL TABLE
    # ========================================================

    st.markdown(
        "### Matchup Target Probability"
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
        "### H2H Score Distribution"
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
        "### Monte Carlo Heatmap Analysis"
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


