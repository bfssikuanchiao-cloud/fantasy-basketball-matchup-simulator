from pathlib import Path
import streamlit as st

APP_FOLDER = Path(__file__).parent
LOGO_PATH = APP_FOLDER / "秘密結社.png"
page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🏀"

st.set_page_config(
    page_title="秘密結社夢幻籃球模擬器",
    page_icon=page_icon,
    layout="wide",
)

matchup_page = st.Page(
    "matchup_simulator.py", title="每週對決模擬", icon="🏀", default=True,
)
bench_page = st.Page(
    "bench_decision.py", title="板凳決策水晶球", icon="🪑",
)
league_page = st.Page(
    "league_week_analysis.py", title="聯盟週分析", icon="📊",
)

navigation = st.navigation(
    [matchup_page, bench_page, league_page],
    position="top",
)
navigation.run()
