from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import streamlit as st

from fantasy_engine import compare_bench_scenarios_fast, load_schedule

APP_FOLDER = Path(__file__).parent
SCHEDULE_PATH = APP_FOLDER / "schedule.csv"
CATEGORY_ORDER = ["FG%", "FT%", "3PM", "Pts", "Reb", "Ast", "Stl", "Blk", "TO"]
CATEGORY_DISPLAY = {"FG%":"FG%", "FT%":"FT%", "3PM":"3PT", "Pts":"PT", "Reb":"REB", "Ast":"AST", "Stl":"STL", "Blk":"BLK", "TO":"TO"}
DISPLAY_ORDER = [CATEGORY_DISPLAY[x] for x in CATEGORY_ORDER]
STAT_COLUMNS = ["FGA", "FGM", "FTA", "FTM", "3PM", "Pts", "Reb", "Ast", "Stl", "Blk", "TO"]


def yahoo_injury_status(player):
    """Raw Yahoo status for logic only."""
    return str(player.get("status") or "").strip().upper()


def yahoo_injury_designation(player):
    """Display label identical to matchup_simulator.py."""
    status = yahoo_injury_status(player)
    return {"O":"O ❌", "D":"D 🟠", "Q":"Q 🔵", "P":"P 🟢"}.get(status, status)


def style_injury_status(value):
    """Display styling identical to matchup_simulator.py."""
    display = str(value or "").strip()
    status = display.split(maxsplit=1)[0].upper() if display else ""
    color = {"O":"#DC2626", "D":"#F97316", "Q":"#2563EB", "P":"#16A34A"}.get(status)
    return f"color: {color}; font-weight: 700;" if color else ""


def injury_default_enabled(player):
    raw_status = yahoo_injury_status(player)
    slot = str(player.get("selected_position") or "").strip().upper()
    if slot in {"IL", "IL+", "IR", "IR+"}:
        return False
    return raw_status not in {"O", "OUT", "INJ", "IL", "IL+", "IR", "IR+"}


def scheduled_rows(players, schedule, start_date, end_date):
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    rows = []
    for player in players:
        name = player.get("name")
        team = str(player.get("team") or "").upper()
        games = schedule.loc[
            schedule["team"].astype(str).str.upper().eq(team)
            & schedule["date"].between(start_ts, end_ts)
        ].sort_values("date")
        for _, game in games.iterrows():
            game_date = pd.Timestamp(game["date"]).normalize()
            away = str(game.get("home_away", "")).lower() == "away"
            opp = str(game.get("opponent", "")).upper()
            rows.append({
                "Player": name,
                "Team": team,
                "Date": game_date.strftime("%Y-%m-%d"),
                "Game": f"@{opp}" if away else opp,
                "Injury": yahoo_injury_designation(player),
                "Play All": injury_default_enabled(player),
                "Bench": False,
            })
    return pd.DataFrame(rows)


def selections_from_table(table):
    play_all, bench = {}, {}
    for row in table.to_dict("records"):
        player, game_date = row["Player"], row["Date"]
        enabled = bool(row["Play All"])
        play_all.setdefault(player, {})[game_date] = enabled
        bench.setdefault(player, {})[game_date] = enabled and not bool(row["Bench"])
    return play_all, bench


def opponent_selection(players, schedule, start_date, end_date):
    table = scheduled_rows(players, schedule, start_date, end_date)
    selection = {}
    for row in table.to_dict("records"):
        selection.setdefault(row["Player"], {})[row["Date"]] = bool(row["Play All"])
    return selection, table


def fmt_projected(category, payload):
    if category in {"FG%", "FT%"}:
        return f"{payload['mean'] * 100:.1f}% ({payload['p5'] * 100:.1f}-{payload['p95'] * 100:.1f}%)"
    return f"{payload['mean']:.1f} ({payload['p5']:.1f}-{payload['p95']:.1f})"


st.title("🪑 Bench Decision")
st.caption("比較 Play All 與 Bench Selected Games。兩個情境共用同一批 Monte Carlo 抽樣，差異只來自被取消的本陣 player-games。")
WallPaint_PATH = APP_FOLDER / "WallPaint.png"
col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])
with col7:
    st.image(
        str(WallPaint_PATH),
        width=100
    )
context = st.session_state.get("yahoo_context")
if not isinstance(context, dict):
    st.warning("請先到 Matchup 模擬頁載入 Yahoo League、matchup 與本週累積數據。")
    st.stop()
if not st.session_state.get("rosters_loaded"):
    st.warning("請先到 Matchup 模擬頁載入雙方球員歷史資料。")
    st.stop()
if not SCHEDULE_PATH.exists():
    st.error("找不到 schedule.csv。")
    st.stop()

schedule = load_schedule(str(SCHEDULE_PATH))
left, middle, right = st.columns(3)
with left: start_date = st.date_input("Start Date", key="bench_start_date")
with middle: end_date = st.date_input("End Date", key="bench_end_date")
with right: n_sim = st.number_input("Monte Carlo Simulations", 100, 100000, 3000, 100, key="bench_n_sim")
if start_date > end_date:
    st.error("Start Date 不能晚於 End Date。")
    st.stop()

signature = (str(start_date), str(end_date), context.get("my_team_key"), context.get("opponent_team_key"))
base_table = scheduled_rows(context.get("my_roster", []), schedule, start_date, end_date)
if st.session_state.get("bench_table_signature") != signature:
    st.session_state.bench_table_signature = signature
    st.session_state.bench_game_table = base_table
    st.session_state.bench_last_result = None

st.subheader("Scenario 1 與 Scenario 2 選擇")
st.caption("Play All 決定基準情境要模擬的比賽；Bench 只會從 Scenario 2 移除該場。")
if base_table.empty:
    st.info("所選日期範圍內，本陣沒有比賽。")
    st.stop()

styled_bench_table = st.session_state.bench_game_table.style.map(style_injury_status, subset=["Injury"])
edited = st.data_editor(
    styled_bench_table,
    hide_index=True,
    use_container_width=True,
    disabled=["Player", "Team", "Date", "Game", "Injury"],
    column_config={
        "Injury": st.column_config.TextColumn("Injury", width="small", help="Yahoo player injury designation; blank means Yahoo has no injury designation."),
        "Play All": st.column_config.CheckboxColumn("Scenario 1: Play"),
        "Bench": st.column_config.CheckboxColumn("Scenario 2: Bench"),
    },
    key=f"bench_editor_{signature}",
)
edited.loc[~edited["Play All"].astype(bool), "Bench"] = False
st.session_state.bench_game_table = edited
bench_count = int(edited["Bench"].sum())
st.caption(f"Scenario 2 將板凳 {bench_count} 個 player-game。")

opp_selected, opp_table = opponent_selection(context.get("opponent_roster", []), schedule, start_date, end_date)
with st.expander("敵方預設上場比賽", expanded=False):
    styled_opp = opp_table.drop(columns=["Bench"]).style.map(style_injury_status, subset=["Injury"])
    st.dataframe(styled_opp, hide_index=True, use_container_width=True)

st.subheader("目前九項累積成績")
my_col, opp_col = st.columns(2)
current_my, current_opp = {}, {}
with my_col:
    st.markdown("#### 本陣")
    for stat in STAT_COLUMNS:
        default = int(context.get("current_my", {}).get(stat, 0))
        current_my[stat] = st.number_input(f"My {stat}", min_value=0, value=default, step=1, key=f"bench_my_{stat}")
with opp_col:
    st.markdown("#### 敵陣")
    for stat in STAT_COLUMNS:
        default = int(context.get("current_opp", {}).get(stat, 0))
        current_opp[stat] = st.number_input(f"Opponent {stat}", min_value=0, value=default, step=1, key=f"bench_opp_{stat}")

run = st.button("比較 Play All 與 Bench", type="primary", use_container_width=True)
if run:
    play_all_selection, bench_selection = selections_from_table(edited)
    if bench_count == 0: st.warning("尚未勾選任何 Bench player-game。兩個 Scenario 會完全相同。")
    try:
        with st.spinner("Running paired Monte Carlo..."):
            st.session_state.bench_last_result = compare_bench_scenarios_fast(
                my_roster=st.session_state.my_roster_data, opp_roster=st.session_state.opp_roster_data,
                schedule_df=schedule, start_date=str(start_date), end_date=str(end_date),
                current_my=current_my, current_opp=current_opp,
                play_all_my_selected_games=play_all_selection, bench_my_selected_games=bench_selection,
                opp_selected_games=opp_selected, n_sim=int(n_sim), random_seed=12345,
            )
    except Exception as error:
        st.error("Bench Decision 模擬失敗。")
        st.exception(error)

result = st.session_state.get("bench_last_result")
if result is None: st.stop()
overall = result["overall_comparison"].set_index("Metric")
delta = float(overall.loc["Matchup win", "Delta"])
if delta >= 0.03: st.success(f"建議板凳：Matchup 勝率提高 {delta * 100:.1f} 個百分點。")
elif delta <= -0.03: st.error(f"建議上場：板凳會讓 Matchup 勝率下降 {abs(delta) * 100:.1f} 個百分點。")
else: st.warning(f"差異不明顯：Matchup 勝率變化 {delta * 100:+.1f} 個百分點。建議提高模擬次數再確認。")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Play All matchup win", f"{overall.loc['Matchup win','Play All']*100:.1f}%")
m2.metric("Bench matchup win", f"{overall.loc['Matchup win','Bench']*100:.1f}%", f"{delta*100:+.1f} pp")
m3.metric("Play All expected wins", f"{result['play_all']['expected_category_wins']:.2f}")
m4.metric("Bench expected wins", f"{result['bench']['expected_category_wins']:.2f}", f"{result['bench']['expected_category_wins']-result['play_all']['expected_category_wins']:+.2f}")

st.markdown("### FG%、FT%、TO 保護效果")
protection = result["category_comparison"].query("Category in ['FG%', 'FT%', 'TO']").copy()
protection["Category"] = protection["Category"].map(CATEGORY_DISPLAY)
for col in ["Play All Win","Bench Win","Delta","Play All Tie","Bench Tie"]: protection[col] = (protection[col]*100).round(1)
st.dataframe(protection, hide_index=True, use_container_width=True)

st.markdown("### 九項勝率變化")
comparison = result["category_comparison"].copy()
comparison["Display"] = comparison["Category"].map(CATEGORY_DISPLAY)
comparison["Display"] = pd.Categorical(comparison["Display"], categories=DISPLAY_ORDER, ordered=True)
plot_data = comparison.sort_values("Display")[["Display","Delta"]].copy()
plot_data["Delta pp"] = plot_data["Delta"] * 100
fig, ax = plt.subplots(figsize=(9,4.5))
colors = ["#16A34A" if value >= 0 else "#DC2626" for value in plot_data["Delta pp"]]
ax.barh(plot_data["Display"], plot_data["Delta pp"], color=colors)
ax.invert_yaxis()
ax.axvline(0,color="#374151",linewidth=1)
ax.set_xlabel("Bench minus Play All win probability (percentage points)")
ax.set_ylabel("")
ax.grid(axis="x",alpha=.2)
fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

st.markdown("### 預測最終值與 90% 模擬區間")
projection_rows=[]
for category in CATEGORY_ORDER:
    projection_rows.append({"Category":CATEGORY_DISPLAY[category], "Play All":fmt_projected(category,result["play_all"]["projected"][category]), "Bench":fmt_projected(category,result["bench"]["projected"][category])})
st.dataframe(pd.DataFrame(projection_rows),hide_index=True,use_container_width=True)
st.caption("FG% / FT% 的中心值採 mean(made) / mean(attempts)；括號為每次模擬最終百分比或累積值的 P5-P95。")
MobileSuit_PATH = APP_FOLDER / "MobileSuit.png"
col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])
with col1:
    st.image(
        str(MobileSuit_PATH),
        width=120
    )
st.markdown("### H2H Category Wins 分布")
score=result["score_distribution_comparison"].copy()
fig,ax=plt.subplots(figsize=(9,4.5))
sns.lineplot(data=score,x="Wins",y="Probability",hue="Scenario",marker="o",ax=ax,palette={"Play All":"#7E22CE","Bench":"#F59E0B"})
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.set_xticks(range(10)); ax.set_xlabel("Categories won"); ax.set_ylabel("Probability"); ax.grid(alpha=.2)
fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)



