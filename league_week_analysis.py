from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fantasy_engine import select_yahoo_league_matchup


# ============================================================
# PAGE CONSTANTS
# ============================================================
APP_FOLDER = Path(__file__).parent
POSITIVE_CATEGORIES = ["FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK"]
ALL_CATEGORIES = POSITIVE_CATEGORIES + ["TO"]
COUNTING_COLUMNS = ["3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]

STAT_ALIASES = {
    "FGM": ("FGM", "Fgm", "fgm"),
    "FGA": ("FGA", "Fga", "fga"),
    "FTM": ("FTM", "Ftm", "ftm"),
    "FTA": ("FTA", "Fta", "fta"),
    "3PM": ("3PM", "3Pm", "3ptm", "3PTM"),
    "PTS": ("PTS", "Pts", "pts"),
    "REB": ("REB", "Reb", "reb"),
    "AST": ("AST", "Ast", "ast"),
    "STL": ("STL", "Stl", "stl"),
    "BLK": ("BLK", "Blk", "blk"),
    "TO": ("TO", "To", "to"),
}


# ============================================================
# DATA HELPERS
# ============================================================
def _first_number(source: dict, aliases: tuple[str, ...]) -> float:
    """Return the first numeric value found among possible Yahoo stat keys."""
    for key in aliases:
        if key not in source:
            continue
        value = source.get(key)
        if value in (None, "", "-"):
            continue
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
    return 0.0


def _extract_team_row(team: dict, totals: dict) -> dict:
    values = {
        stat: _first_number(totals, aliases)
        for stat, aliases in STAT_ALIASES.items()
    }
    fg_pct = values["FGM"] / values["FGA"] if values["FGA"] > 0 else np.nan
    ft_pct = values["FTM"] / values["FTA"] if values["FTA"] > 0 else np.nan

    return {
        "Team Key": team.get("team_key", ""),
        "Team": team.get("name", team.get("team_key", "Unknown Team")),
        "FGM": values["FGM"],
        "FGA": values["FGA"],
        "FG%": fg_pct,
        "FTM": values["FTM"],
        "FTA": values["FTA"],
        "FT%": ft_pct,
        "3PM": values["3PM"],
        "PTS": values["PTS"],
        "REB": values["REB"],
        "AST": values["AST"],
        "STL": values["STL"],
        "BLK": values["BLK"],
        "TO": values["TO"],
    }


def load_all_league_week_totals(context: dict) -> pd.DataFrame:
    """
    Reuse the existing lightweight Yahoo matchup selector to retrieve the
    current-week cumulative totals for every fantasy team.

    This intentionally does not load player history and does not call the
    Monte Carlo pipeline.
    """
    teams = context.get("league_teams") or []
    if len(teams) < 2:
        raise ValueError("Yahoo league context does not contain at least two teams.")

    rows: list[dict] = []
    for index, team in enumerate(teams):
        team_key = team.get("team_key")
        if not team_key:
            continue

        comparison_team = next(
            candidate for candidate in teams
            if candidate.get("team_key") != team_key
        )
        selected = select_yahoo_league_matchup(
            context,
            team_key,
            comparison_team["team_key"],
        )
        totals = selected.get("current_my") or {}
        rows.append(_extract_team_row(team, totals))

    if not rows:
        raise ValueError("No Yahoo team totals were returned.")

    return pd.DataFrame(rows)


def _min_max_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Excel-style 0-100 Min-Max score; neutral 50 when every team is tied."""
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min(skipna=True)
    maximum = numeric.max(skipna=True)

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(50.0, index=series.index)
    if np.isclose(maximum, minimum):
        return pd.Series(50.0, index=series.index)

    if reverse:
        score = (maximum - numeric) / (maximum - minimum) * 100
    else:
        score = (numeric - minimum) / (maximum - minimum) * 100
    return score.fillna(50.0).clip(0, 100)


def calculate_league_strength(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the workbook's nine-category Min-Max strength calculation."""
    result = raw[["Team Key", "Team"]].copy()

    for category in POSITIVE_CATEGORIES:
        result[category] = _min_max_score(raw[category])
    result["TO"] = _min_max_score(raw["TO"], reverse=True)

    result["Weekly Strength"] = result[ALL_CATEGORIES].mean(axis=1)
    result["Rank"] = (
        result["Weekly Strength"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result.sort_values(
        ["Weekly Strength", "Team"], ascending=[False, True]
    ).reset_index(drop=True)


def raw_display_frame(raw: pd.DataFrame) -> pd.DataFrame:
    display = raw.drop(columns=["Team Key"]).copy()
    for column in ["FGM", "FGA", "FTM", "FTA"] + COUNTING_COLUMNS:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0).round(0).astype(int)
    display["FG%"] = (display["FG%"] * 100).round(1)
    display["FT%"] = (display["FT%"] * 100).round(1)
    return display


def strength_display_frame(strength: pd.DataFrame) -> pd.DataFrame:
    columns = ["Rank", "Team", "Weekly Strength"] + ALL_CATEGORIES
    display = strength[columns].copy()
    for column in ["Weekly Strength"] + ALL_CATEGORIES:
        display[column] = display[column].round(1)
    return display


def render_radar(team_row: pd.Series):
    values = team_row[ALL_CATEGORIES].astype(float).tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(ALL_CATEGORIES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, values, color="#7E22CE", linewidth=2)
    ax.fill(angles, values, color="#A855F7", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ALL_CATEGORIES)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title(str(team_row["Team"]), pad=22, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================
# PAGE UI
# ============================================================
st.title("📊 聯盟本週實力分析")
st.caption(
    "讀取 Yahoo League 所有 Fantasy Teams 的本週累積數據，依 Excel 方法將九項數據轉為 0–100，並計算 Weekly Strength。"
)

context = st.session_state.get("yahoo_context")
if not isinstance(context, dict) or not context.get("league_teams"):
    st.warning("尚未載入 Yahoo League。請先回到 Matchup 頁完成 Yahoo 授權並按下「載入 Yahoo League」。")
    st.stop()

league_id = context.get("public_league_id", "")
league_key = context.get("league_key", "")
current_week = context.get("current_week", "")
teams = context.get("league_teams") or []

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("League", league_id or league_key)
info_col2.metric("Current Week", current_week)
info_col3.metric("Fantasy Teams", len(teams))

st.info(
    "目前版本重用既有的輕量 Yahoo matchup loader，只讀取本週累積數據；不下載歷史球員資料，也不執行 Monte Carlo。"
)

load_label = (
    "重新載入本週全聯盟數據"
    if st.session_state.get("league_week_raw") is not None
    else "載入本週全聯盟數據"
)

if st.button(load_label, type="primary", use_container_width=True):
    try:
        progress = st.progress(0, text="正在讀取 Yahoo League teams...")
        raw = load_all_league_week_totals(context)
        progress.progress(75, text="正在計算九項 Min-Max 分數...")
        strength = calculate_league_strength(raw)
        st.session_state["league_week_raw"] = raw
        st.session_state["league_week_strength"] = strength
        st.session_state["league_week_signature"] = (
            league_key,
            str(current_week),
        )
        st.session_state["league_week_loaded_at"] = datetime.now().isoformat(timespec="seconds")
        progress.progress(100, text="完成")
        progress.empty()
        st.success("全聯盟本週累積數據與 Weekly Strength 已更新。")
    except Exception as error:
        st.error("載入全聯盟本週數據失敗。現有 Matchup 與 Monte Carlo 資料沒有被修改。")
        st.exception(error)

expected_signature = (league_key, str(current_week))
loaded_signature = st.session_state.get("league_week_signature")
raw = st.session_state.get("league_week_raw")
strength = st.session_state.get("league_week_strength")

if loaded_signature != expected_signature:
    if raw is not None:
        st.warning("目前保存的分析來自另一個 League 或 Week，請重新載入。")
    st.stop()

if raw is None or strength is None:
    st.info("按下「載入本週全聯盟數據」開始分析。")
    st.stop()

loaded_at = st.session_state.get("league_week_loaded_at", "")
st.caption(f"Last refreshed: {loaded_at}")

# ============================================================
# OVERVIEW
# ============================================================
st.divider()
st.header("本週聯盟概況")
leader = strength.iloc[0]
most_balanced = strength.iloc[(strength[ALL_CATEGORIES].std(axis=1)).argmin()]
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Weekly Strength Leader", leader["Team"], f"{leader['Weekly Strength']:.1f}")
metric2.metric("League Average", f"{strength['Weekly Strength'].mean():.1f}")
metric3.metric("Most Balanced Profile", most_balanced["Team"])

# ============================================================
# RAW TOTALS
# ============================================================
st.divider()
st.header("本週原始累積數據")
st.caption("FG% = FGM ÷ FGA；FT% = FTM ÷ FTA。顯示值為百分比，例如 48.5 代表 48.5%。")
raw_display = raw_display_frame(raw)
st.dataframe(
    raw_display.style.format({"FG%": "{:.1f}", "FT%": "{:.1f}"}),
    hide_index=True,
    use_container_width=True,
)

# ============================================================
# STRENGTH TABLE
# ============================================================
st.divider()
st.header("九項標準化與 Weekly Strength")
st.caption(
    "FG%、FT%、3PM、PTS、REB、AST、STL、BLK 採正向 Min-Max；TO 採反向 Min-Max。九項平均即 Weekly Strength。若全聯盟某項完全相同，該項一律給 50。"
)
strength_display = strength_display_frame(strength)
st.dataframe(
    strength_display.style
    .background_gradient(subset=["Weekly Strength"] + ALL_CATEGORIES, cmap="RdYlGn", vmin=0, vmax=100)
    .format({column: "{:.1f}" for column in ["Weekly Strength"] + ALL_CATEGORIES}),
    hide_index=True,
    use_container_width=True,
)

# ============================================================
# TEAM RADAR
# ============================================================
st.divider()
st.header("單隊九項輪廓")
selected_team = st.selectbox("Fantasy Team", strength["Team"].tolist())
team_row = strength.loc[strength["Team"] == selected_team].iloc[0]
radar_col, detail_col = st.columns([1.2, 1])
with radar_col:
    radar_fig = render_radar(team_row)
    st.pyplot(radar_fig, use_container_width=True)
    plt.close(radar_fig)
with detail_col:
    st.metric("League Rank", f"#{int(team_row['Rank'])}")
    st.metric("Weekly Strength", f"{team_row['Weekly Strength']:.1f}")
    detail = pd.DataFrame({
        "Category": ALL_CATEGORIES,
        "Strength": [round(float(team_row[c]), 1) for c in ALL_CATEGORIES],
    }).sort_values("Strength", ascending=False)
    st.dataframe(detail, hide_index=True, use_container_width=True)

# ============================================================
# EXPORT
# ============================================================
st.divider()
st.header("匯出分析")
export = raw.merge(
    strength[["Team Key", "Rank", "Weekly Strength"] + ALL_CATEGORIES],
    on="Team Key",
    suffixes=(" Raw", " Strength"),
)
export_csv = export.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下載本週聯盟分析 CSV",
    data=export_csv,
    file_name=f"league_week_{current_week}_analysis.csv",
    mime="text/csv",
    use_container_width=True,
)
