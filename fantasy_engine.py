# Imports

import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.endpoints.commonallplayers import CommonAllPlayers
from nba_api.stats.static import players
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
)


# ============================================================
# 0. MODEL SETTINGS
# ============================================================

DECAY = 0.99

HOME_BOOST = 1.3
AWAY_BOOST = 1.3
OPP_BOOST = 1.8


# ============================================================
# MULTI-SEASON SETTINGS
# ============================================================

SEASONS = (
    "2026-27",
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
    "2019-20",
    "2018-19"
)

CURRENT_SEASON = "2026-27"

STAT_COLS = [
    "Pts",
    "Reb",
    "Ast",
    "Stl",
    "Blk",
    "3PM",
    "TO",
    "FGM",
    "FGA",
    "FTM",
    "FTA"
]

# ============================================================
# PLAYER CACHE SETTINGS
# ============================================================

CACHE_DIR = (
    Path(__file__).resolve().parent
    / "player_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. FIND PLAYER
# ============================================================

@st.cache_data
def get_current_players():

    return CommonAllPlayers(
        is_only_current_season=1
    ).get_data_frames()[0]
    
CURRENT_PLAYERS = get_current_players()

def find_player(name):

    matches = CURRENT_PLAYERS[
        CURRENT_PLAYERS["DISPLAY_FIRST_LAST"]
        .str.lower()
        .eq(name.lower())
    ]

    if matches.empty:

        matches = CURRENT_PLAYERS[
            CURRENT_PLAYERS["DISPLAY_FIRST_LAST"]
            .str.contains(
                name,
                case=False,
                na=False
            )
        ]

    if matches.empty:
        raise ValueError(
            f"找不到球員：{name}"
        )

    if len(matches) > 1:

        print(
            f"\n找到多位符合 '{name}' 的球員："
        )

        for _, row in matches.iterrows():
            print(
                row["PERSON_ID"],
                row["DISPLAY_FIRST_LAST"]
            )

        raise ValueError(
            "請輸入更完整的球員名字"
        )

    player = matches.iloc[0]

    print(
        f"Matched: "
        f"{player['DISPLAY_FIRST_LAST']} "
        f"(NBA ID: {player['PERSON_ID']})"
    )

    return {
        "id": int(player["PERSON_ID"]),
        "full_name": player["DISPLAY_FIRST_LAST"],
        "is_active": True
    }

# ============================================================
# 2. SAFE CACHE FILE NAME
# ============================================================

def make_safe_cache_name(
    player_id,
    player_name
):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        player_name.strip().lower()
    )

    safe_name = safe_name.strip("_")

    return (
        f"{player_id}_"
        f"{safe_name}.pkl"
    )


# ============================================================
# 3. GET PLAYER CACHE PATH
# ============================================================

def get_player_cache_path(
    player_id,
    player_name
):

    file_name = make_safe_cache_name(
        player_id=player_id,
        player_name=player_name
    )

    return CACHE_DIR / file_name


# ============================================================
# 4. CREATE EMPTY PLAYER CACHE
# ============================================================

def create_empty_player_cache(
    player_id,
    player_name
):

    return {
        "player_id": player_id,
        "player_name": player_name,
        "current_team": None,
        "seasons": {}
    }


# ============================================================
# 5. LOAD PLAYER CACHE
# ============================================================

def load_player_cache(
    player_id,
    player_name
):

    cache_path = get_player_cache_path(
        player_id=player_id,
        player_name=player_name
    )

    empty_cache = create_empty_player_cache(
        player_id=player_id,
        player_name=player_name
    )

    if not cache_path.exists():

        print(
            f"Cache not found: "
            f"{player_name}"
        )

        return empty_cache

    try:

        with open(
            cache_path,
            "rb"
        ) as file:

            cache = pickle.load(file)

        if not isinstance(
            cache,
            dict
        ):

            print(
                f"WARNING: Invalid cache format "
                f"for {player_name}."
            )

            return empty_cache

        if "seasons" not in cache:

            cache["seasons"] = {}

        if not isinstance(
            cache["seasons"],
            dict
        ):

            cache["seasons"] = {}

        if "current_team" not in cache:

            cache["current_team"] = None

        cache["player_id"] = player_id
        cache["player_name"] = player_name

        print(
            f"Cache loaded: "
            f"{player_name}"
        )

        return cache

    except Exception as error:

        print(
            f"WARNING: Could not read cache "
            f"for {player_name}"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        return empty_cache


# ============================================================
# 6. SAVE PLAYER CACHE
# ============================================================

def save_player_cache(
    player_id,
    player_name,
    cache
):

    cache_path = get_player_cache_path(
        player_id=player_id,
        player_name=player_name
    )

    temp_path = cache_path.with_suffix(
        ".tmp"
    )

    cache["player_id"] = player_id
    cache["player_name"] = player_name

    if "current_team" not in cache:

        cache["current_team"] = None

    if "seasons" not in cache:

        cache["seasons"] = {}

    try:

        with open(
            temp_path,
            "wb"
        ) as file:

            pickle.dump(
                cache,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        temp_path.replace(
            cache_path
        )

    except Exception as error:

        print(
            f"WARNING: Could not save cache "
            f"for {player_name}"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        if temp_path.exists():

            try:

                temp_path.unlink()

            except Exception:

                pass


# ============================================================
# 7. GET CURRENT PLAYER TEAM
#
# Current team source:
# CommonPlayerInfo.TEAM_ABBREVIATION
#
# Every roster load:
# Try to refresh current team from the NBA API.
#
# Success:
# Update current_team in the player cache.
#
# Timeout / API error:
# Use the last successfully cached current_team.
#
# No API result and no cached team:
# Raise an error instead of using an old historical team.
# ============================================================

def get_current_player_team(
    player_id,
    player_name,
    cache,
    timeout=60
):

    cached_team = cache.get(
        "current_team"
    )

    if cached_team is not None:

        cached_team = str(
            cached_team
        ).strip().upper()

        if cached_team in {
            "",
            "NONE",
            "NAN",
            "NULL",
            "0"
        }:

            cached_team = None

    try:

        player_info_endpoint = (
            commonplayerinfo.CommonPlayerInfo(
                player_id=player_id,
                timeout=timeout
            )
        )

        player_info_df = (
            player_info_endpoint
            .common_player_info
            .get_data_frame()
        )

        if not isinstance(
            player_info_df,
            pd.DataFrame
        ):

            raise ValueError(
                "CommonPlayerInfo did not return "
                "a DataFrame"
            )

        if player_info_df.empty:

            raise ValueError(
                "CommonPlayerInfo returned no rows"
            )

        if (
            "TEAM_ABBREVIATION"
            not in player_info_df.columns
        ):

            raise ValueError(
                "CommonPlayerInfo result is missing "
                "TEAM_ABBREVIATION"
            )

        current_team = str(
            player_info_df.iloc[0][
                "TEAM_ABBREVIATION"
            ]
        ).strip().upper()

        invalid_team_values = {
            "",
            "NONE",
            "NAN",
            "NULL",
            "0"
        }

        if current_team in invalid_team_values:

            raise ValueError(
                "TEAM_ABBREVIATION is empty"
            )

        cache["current_team"] = current_team

        save_player_cache(
            player_id=player_id,
            player_name=player_name,
            cache=cache
        )

        print(
            f"Current team refreshed: "
            f"{player_name} -> {current_team}"
        )

        return current_team

    except Exception as error:

        print(
            f"WARNING: Could not refresh current team "
            f"for {player_name}"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        if cached_team is not None:

            print(
                f"Using cached current team: "
                f"{player_name} -> {cached_team}"
            )

            return cached_team

        raise RuntimeError(
            f"無法取得 {player_name} 的目前球隊，"
            "而且 Cache 中沒有可用的 current_team。"
            "請再次按下 Load，讓程式重新嘗試 NBA API。"
        ) from error


# ============================================================
# 8. CHECK WHETHER A SEASON EXISTS IN CACHE
# ============================================================

def has_valid_cached_season(
    cache,
    season
):

    if "seasons" not in cache:

        return False

    if season not in cache["seasons"]:

        return False

    cached_df = cache["seasons"][season]

    return isinstance(
        cached_df,
        pd.DataFrame
    )


# ============================================================
# 9. DOWNLOAD ONE PLAYER SEASON
# ============================================================

def download_player_season(
    player_id,
    season,
    timeout=60
):

    try:

        df_season = (
            playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=(
                    "Regular Season"
                ),
                timeout=timeout
            )
            .get_data_frames()[0]
        )

        if not isinstance(
            df_season,
            pd.DataFrame
        ):

            print(
                f"    Invalid API response: "
                f"{season}"
            )

            return False, None

        df_season = df_season.copy()

        df_season["Season"] = season

        return True, df_season

    except (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError,
        TimeoutError
    ) as error:

        print(
            f"    TIMEOUT: {season}"
        )

        print(
            f"    {type(error).__name__}: "
            f"{error}"
        )

        return False, None

    except Exception as error:

        print(
            f"    API ERROR: {season}"
        )

        print(
            f"    {type(error).__name__}: "
            f"{error}"
        )

        return False, None


# ============================================================
# 10. LOAD ONE HISTORICAL SEASON
#
# Completed season:
#
# Cache exists:
# Use cache.
#
# Cache missing:
# Download from API.
#
# Download succeeds:
# Save immediately.
#
# Download fails:
# Do not create that season in cache.
# Retry on the next Load.
# ============================================================

def load_historical_season(
    player_id,
    player_name,
    season,
    cache,
    timeout=60
):

    if has_valid_cached_season(
        cache=cache,
        season=season
    ):

        cached_df = (
            cache["seasons"][season]
            .copy()
        )

        print(
            f"    Cache: "
            f"{len(cached_df)} games"
        )

        return cached_df

    success, downloaded_df = (
        download_player_season(
            player_id=player_id,
            season=season,
            timeout=timeout
        )
    )

    if not success:

        print(
            f"    Not cached. "
            f"Will retry {season} "
            f"on next Load."
        )

        return None

    cache["seasons"][season] = (
        downloaded_df.copy()
    )

    save_player_cache(
        player_id=player_id,
        player_name=player_name,
        cache=cache
    )

    if downloaded_df.empty:

        print(
            f"    Downloaded and cached "
            f"empty season: {season}"
        )

    else:

        print(
            f"    Downloaded and cached: "
            f"{len(downloaded_df)} games"
        )

    return downloaded_df.copy()


# ============================================================
# 11. UPDATE CURRENT SEASON
#
# Current season:
#
# Every Load:
# Try API again.
#
# Download succeeds:
# Replace cached current season.
#
# Download fails:
# Use previous cached current season.
#
# No previous current-season cache:
# Return None and retry on the next Load.
# ============================================================

def update_current_season(
    player_id,
    player_name,
    season,
    cache,
    timeout=60
):

    success, downloaded_df = (
        download_player_season(
            player_id=player_id,
            season=season,
            timeout=timeout
        )
    )

    if success:

        cache["seasons"][season] = (
            downloaded_df.copy()
        )

        save_player_cache(
            player_id=player_id,
            player_name=player_name,
            cache=cache
        )

        if downloaded_df.empty:

            print(
                f"    Updated current season: "
                f"No games found"
            )

        else:

            print(
                f"    Updated current season: "
                f"{len(downloaded_df)} games"
            )

        return downloaded_df.copy()

    if has_valid_cached_season(
        cache=cache,
        season=season
    ):

        cached_df = (
            cache["seasons"][season]
            .copy()
        )

        print(
            "    Current-season update failed."
        )

        print(
            f"    Using cached {season}: "
            f"{len(cached_df)} games"
        )

        return cached_df

    print(
        f"    Current-season update failed "
        f"and no cache exists: {season}"
    )

    print(
        f"    Will retry {season} "
        f"on next Load."
    )

    return None


# ============================================================
# 12. NORMALIZE PLAYER GAME LOG
#
# IMPORTANT:
#
# Team parsed here is the historical team for each game.
# It is not used as the player's current team.
#
# The current team is stored separately in:
# df.attrs["current_team"]
# ============================================================

def normalize_player_game_log(
    df,
    player_name
):

    if df.empty:

        return df

    df = df.copy()

    rename_map = {
        "GAME_DATE": "Date",
        "MATCHUP": "Matchup",
        "PTS": "Pts",
        "REB": "Reb",
        "AST": "Ast",
        "STL": "Stl",
        "BLK": "Blk",
        "FG3M": "3PM",
        "TOV": "TO",
        "FGM": "FGM",
        "FGA": "FGA",
        "FTM": "FTM",
        "FTA": "FTA"
    }

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "Date",
        "Matchup",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO",
        "FGM",
        "FGA",
        "FTM",
        "FTA",
        "Season"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{player_name} 缺少必要欄位："
            f"{missing_columns}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    def parse_matchup(matchup):

        matchup = str(
            matchup
        ).strip()

        if " vs. " in matchup:

            historical_team, opponent = (
                matchup.split(
                    " vs. ",
                    maxsplit=1
                )
            )

            return pd.Series({
                "Team": historical_team.strip(),
                "Opp": opponent.strip(),
                "HomeAway": "HOME",
                "IsHome": True
            })

        if " @ " in matchup:

            historical_team, opponent = (
                matchup.split(
                    " @ ",
                    maxsplit=1
                )
            )

            return pd.Series({
                "Team": historical_team.strip(),
                "Opp": opponent.strip(),
                "HomeAway": "AWAY",
                "IsHome": False
            })

        return pd.Series({
            "Team": None,
            "Opp": None,
            "HomeAway": None,
            "IsHome": None
        })

    matchup_columns = (
        df["Matchup"]
        .apply(parse_matchup)
    )

    for column in [
        "Team",
        "Opp",
        "HomeAway",
        "IsHome"
    ]:

        df[column] = (
            matchup_columns[column]
        )

    numeric_columns = [
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO",
        "FGM",
        "FGA",
        "FTM",
        "FTA"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = (
        df
        .dropna(
            subset=[
                "Date"
            ]
        )
        .sort_values(
            "Date",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# 13. GET MULTI-SEASON PLAYER GAME LOG
# ============================================================

def get_player_df_from_api(
    name,
    seasons=SEASONS,
    timeout=60
):

    player = find_player(
        name
    )

    player_id = player["id"]

    official_player_name = (
        player["full_name"]
    )

    cache = load_player_cache(
        player_id=player_id,
        player_name=official_player_name
    )

    # ========================================================
    # CURRENT TEAM
    #
    # This is refreshed before loading the game-log pool.
    # It does not come from historical MATCHUP.
    # ========================================================

    current_team = get_current_player_team(
        player_id=player_id,
        player_name=official_player_name,
        cache=cache,
        timeout=timeout
    )

    season_dfs = []

    print(
        f"\nLoading historical pool "
        f"for {name}"
    )

    print(
        f"Current schedule team: "
        f"{current_team}"
    )

    for season in seasons:

        print(
            f"  Loading {season}..."
        )

        if season == CURRENT_SEASON:

            df_season = (
                update_current_season(
                    player_id=player_id,
                    player_name=official_player_name,
                    season=season,
                    cache=cache,
                    timeout=timeout
                )
            )

        else:

            df_season = (
                load_historical_season(
                    player_id=player_id,
                    player_name=official_player_name,
                    season=season,
                    cache=cache,
                    timeout=timeout
                )
            )

        if df_season is None:

            continue

        if df_season.empty:

            continue

        df_season = df_season.copy()

        if "Season" not in df_season.columns:

            df_season["Season"] = season

        season_dfs.append(
            df_season
        )

    if not season_dfs:

        print(
            f"\nWARNING: No game data "
            f"available for {name}"
        )

        return pd.DataFrame()

    df = pd.concat(
        season_dfs,
        ignore_index=True
    )

    df = normalize_player_game_log(
        df=df,
        player_name=name
    )

    # ========================================================
    # STORE CURRENT TEAM AS DATAFRAME METADATA
    #
    # Historical Team column remains unchanged.
    # Future schedule lookup must use this current_team attr.
    # ========================================================

    df.attrs["player_id"] = player_id

    df.attrs["official_player_name"] = (
        official_player_name
    )

    df.attrs["current_team"] = (
        current_team
    )

    df.attrs["current_team_source"] = (
        "CommonPlayerInfo.TEAM_ABBREVIATION"
    )

    print(
        f"\n{name}"
    )

    print(
        f"Current team: "
        f"{current_team}"
    )

    print(
        f"Total historical games: "
        f"{len(df)}"
    )

    season_counts = (
        df["Season"]
        .value_counts()
        .reindex(seasons)
        .fillna(0)
        .astype(int)
    )

    print(
        season_counts
    )

    missing_cache_seasons = [
        season
        for season in seasons
        if (
            season != CURRENT_SEASON
            and not has_valid_cached_season(
                cache=cache,
                season=season
            )
        )
    ]

    if missing_cache_seasons:

        print(
            "\nHistorical seasons still "
            "missing from cache:"
        )

        for season in missing_cache_seasons:

            print(
                f"  - {season}"
            )

        print(
            "Press Load again to retry "
            "the missing seasons."
        )

    else:

        print(
            "\nAll completed seasons are cached."
        )

    return df


# ============================================================
# 13. BUILD ROSTER
# ============================================================

def build_roster(
    player_names,
    seasons=SEASONS
):

    roster = {}

    for name in player_names:

        print(
            "\n" + "=" * 60
        )

        print(
            f"Loading {name}"
        )

        print(
            "=" * 60
        )

        player_df = (
            get_player_df_from_api(
                name=name,
                seasons=seasons
            )
        )

        if player_df.empty:

            print(
                f"WARNING: {name} "
                f"DataFrame is empty"
            )

        roster[name] = player_df

    return roster

# =====================================================
# Schedule Functions
# =====================================================

# ============================================================
# 4. LOAD SCHEDULE
# ============================================================

def load_schedule(
    path="schedule.csv"
):

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df



# ============================================================
# 5. GET GAMES IN RANGE
# ============================================================

def get_games_in_range(
    schedule_df,
    team,
    start_date,
    end_date
):

    start_date = pd.to_datetime(
        start_date
    )

    end_date = pd.to_datetime(
        end_date
    )

    mask = (

        (schedule_df["team"] == team)

        &

        (schedule_df["date"] >= start_date)

        &

        (schedule_df["date"] <= end_date)
    )

    return (
        schedule_df[mask]
        .sort_values("date")
        .reset_index(drop=True)
    )



# ============================================================
# 6. SINGLE FUTURE GAME WEIGHT
#
# Recency
# × Home/Away
# × Opponent
# ============================================================

def apply_single_game_weight(
    df,
    game,
    decay=DECAY,
    home_boost=HOME_BOOST,
    away_boost=AWAY_BOOST,
    opp_boost=OPP_BOOST
):

    df = df.copy()

    # --------------------------------------------------------
    # NEW:
    #
    # 不論資料來自哪一季
    # 全部按照真正 Date 排序
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "Date",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Recency Weight
    #
    # 最新一場 = decay^0
    # 次新一場 = decay^1
    # ...
    # --------------------------------------------------------

    df["Weight"] = (
        decay **
        np.arange(len(df))
    )

    # --------------------------------------------------------
    # Future Game Environment
    # --------------------------------------------------------

    next_is_home = (
        game["home_away"] == "home"
    )

    next_opp = (
        game["opponent"]
    )

    # --------------------------------------------------------
    # Historical Game Environment
    # --------------------------------------------------------

    is_home_sample = (
        df["HomeAway"] == "HOME"
    )

    is_away_sample = (
        df["HomeAway"] == "AWAY"
    )

    # --------------------------------------------------------
    # Home / Away Boost
    # --------------------------------------------------------

    if next_is_home:

        df.loc[
            is_home_sample,
            "Weight"
        ] *= home_boost

    else:

        df.loc[
            is_away_sample,
            "Weight"
        ] *= away_boost

    # --------------------------------------------------------
    # Opponent Boost
    # --------------------------------------------------------

    df.loc[
        df["Opp"] == next_opp,
        "Weight"
    ] *= opp_boost

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------
    print("\nWeight Summary Before Normalize")

    print(
        df.groupby("HomeAway")["Weight"]
          .mean()
    )

    df["Weight"] = (
        df["Weight"]
        /
        df["Weight"].sum()
    )

    print("\n========== HOME/AWAY DEBUG ==========")

    print(
        df["HomeAway"]
        .value_counts(dropna=False)
    )

    print(
        "Home matches:",
        (df["HomeAway"] == "HOME").sum()
    )

    print(
        "Away matches:",
        (df["HomeAway"] == "AWAY").sum()
    )


    return df


# ============================================================
# 7. PREPARE PLAYER SIMULATION
#
# Weight Cache
# ============================================================

def prepare_player_simulation(
    player_name,
    player_df,
    schedule_df,
    start_date,
    end_date
):

    print("\n" + "=" * 80)
    print(f"Preparing player: {player_name}")

    # ====================================================
    # VALIDATION
    # ====================================================

    if player_df is None:

        raise ValueError(
            f"{player_name}: player_df is None"
        )

    if getattr(player_df, "empty", True):

        raise ValueError(
            f"{player_name}: EMPTY DATAFRAME"
        )

    # ====================================================
    # CURRENT TEAM
    # ====================================================

    team = player_df.attrs.get(
        "current_team"
    )

    if not team:

        raise ValueError(
            f"{player_name} 缺少 current_team。"
        )

    team = str(
        team
    ).strip().upper()

    print(
        f"Current Team: {team}"
    )

    # ====================================================
    # FUTURE GAMES
    # ====================================================

    games_df = get_games_in_range(
        schedule_df=schedule_df,
        team=team,
        start_date=start_date,
        end_date=end_date
    )

    print(
        f"Future Games Found: "
        f"{len(games_df)}"
    )

    prepared_games = []

    for _, game in games_df.iterrows():

        weighted_df = (
            apply_single_game_weight(
                player_df,
                game
            )
        )

        prepared_games.append({
            "date": game["date"],
            "opponent": game["opponent"],
            "home_away": game["home_away"],
            "weighted_df": weighted_df
        })

    return {
        "player": player_name,
        "team": team,
        "games": prepared_games
    }

# ============================================================
# 8. PREPARE ROSTER
# ============================================================

def prepare_roster_simulation(
    roster,
    schedule_df,
    start_date,
    end_date
):

    prepared_roster = {}

    for (
        player_name,
        player_df
    ) in roster.items():

        print("\n" + "=" * 80)
        print(f"Preparing roster player: {player_name}")

        if hasattr(player_df, "shape"):
            print(
                f"shape = {player_df.shape}"
            )

        prepared_roster[
            player_name
        ] = (
            prepare_player_simulation(
                player_name,
                player_df,
                schedule_df,
                start_date,
                end_date
            )
        )

    return prepared_roster


# ============================================================
# 9. VECTORIZED PLAYER SIMULATION
# ============================================================

def simulate_player_vectorized(
    prepared_player,
    n_sim,
    rng
):

    totals = pd.DataFrame(
        0.0,
        index=np.arange(n_sim),
        columns=STAT_COLS
    )

    # --------------------------------------------------------
    # 每場 Future NBA Game
    # --------------------------------------------------------

    for game_info in prepared_player["games"]:

        weighted_df = (
            game_info["weighted_df"]
        )

        weights = (
            weighted_df["Weight"]
            .to_numpy()
        )

        # ----------------------------------------------------
        # 一次產生全部 Monte Carlo draws
        # ----------------------------------------------------

        sampled_indices = rng.choice(
            len(weighted_df),
            size=n_sim,
            replace=True,
            p=weights
        )

        # ----------------------------------------------------
        # Joint Game-Level Bootstrap
        #
        # 整列 Box Score 一起抽
        # ----------------------------------------------------

        sampled_games = (

            weighted_df

            .iloc[sampled_indices]

            [STAT_COLS]

            .reset_index(drop=True)
        )

        totals += sampled_games

    return totals


# ============================================================
# 10. VECTORIZED ROSTER SIMULATION
# ============================================================

def simulate_roster_vectorized(
    prepared_roster,
    n_sim,
    rng
):

    roster_totals = pd.DataFrame(
        0.0,
        index=np.arange(n_sim),
        columns=STAT_COLS
    )

    for (
        player_name,
        prepared_player
    ) in prepared_roster.items():

        player_totals = (
            simulate_player_vectorized(
                prepared_player,
                n_sim,
                rng
            )
        )

        roster_totals += (
            player_totals
        )

    return roster_totals

# ============================================================
# MONTE CARLO DEBUG SAMPLE
# ============================================================

def create_simulation_debug_sample(
    prepared_roster,
    current_totals,
    n_sim,
    sample_size=5,
    simulation_seed=12345,
    sample_seed=24680
):
    """
    執行球員層級 Monte Carlo，並抽取指定數量的模擬結果，
    用來檢查每次模擬的團隊九項累積值與球員 breakdown。

    Returns
    -------
    dict

    team_sample:
        抽樣模擬的團隊九項總數。
        包含 current_totals 加上未來模擬數字。

    player_sample:
        抽樣模擬中，每位球員的未來貢獻。
        不包含 current_totals。

    sampled_simulation_ids:
        被選出的模擬編號。
    """

    if n_sim < 1:
        raise ValueError(
            "n_sim 必須至少為 1"
        )

    actual_sample_size = min(
        int(sample_size),
        int(n_sim)
    )

    if actual_sample_size < 1:
        raise ValueError(
            "sample_size 必須至少為 1"
        )

    # 和 roster simulation 使用相同的 RNG 架構
    simulation_rng = np.random.default_rng(
        simulation_seed
    )

    player_results = {}

    team_future = pd.DataFrame(
        0.0,
        index=np.arange(n_sim),
        columns=STAT_COLS
    )

    # ========================================================
    # SIMULATE EACH PLAYER
    # ========================================================

    for player_name, prepared_player in prepared_roster.items():

        player_future = simulate_player_vectorized(
            prepared_player=prepared_player,
            n_sim=n_sim,
            rng=simulation_rng
        )

        player_future = (
            player_future
            .reset_index(drop=True)
            .copy()
        )

        player_results[player_name] = player_future

        team_future = (
            team_future
            .add(
                player_future[STAT_COLS],
                fill_value=0.0
            )
        )

    # ========================================================
    # SELECT 20 SIMULATION IDS
    # ========================================================

    sample_rng = np.random.default_rng(
        sample_seed
    )

    sampled_indices = np.sort(
        sample_rng.choice(
            n_sim,
            size=actual_sample_size,
            replace=False
        )
    )

    # 顯示從 1 開始的 Simulation ID
    sampled_simulation_ids = (
        sampled_indices + 1
    )

    # ========================================================
    # TEAM SAMPLE
    #
    # Includes current matchup totals
    # ========================================================

    team_sample_raw = (
        team_future
        .iloc[sampled_indices]
        .reset_index(drop=True)
        .copy()
    )

    counting_stats = [
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO",
        "FGM",
        "FGA",
        "FTM",
        "FTA"
    ]

    for stat in counting_stats:

        current_value = float(
            current_totals.get(
                stat,
                0
            )
        )

        team_sample_raw[stat] = (
            team_sample_raw[stat]
            + current_value
        )

    team_sample_raw["FG%"] = np.where(
        team_sample_raw["FGA"] > 0,
        (
            team_sample_raw["FGM"]
            / team_sample_raw["FGA"]
        ),
        0.0
    )

    team_sample_raw["FT%"] = np.where(
        team_sample_raw["FTA"] > 0,
        (
            team_sample_raw["FTM"]
            / team_sample_raw["FTA"]
        ),
        0.0
    )

    team_sample = pd.DataFrame({
        "Simulation": sampled_simulation_ids,
        "FG%": team_sample_raw["FG%"],
        "FT%": team_sample_raw["FT%"],
        "Pts": team_sample_raw["Pts"],
        "Reb": team_sample_raw["Reb"],
        "Ast": team_sample_raw["Ast"],
        "Stl": team_sample_raw["Stl"],
        "Blk": team_sample_raw["Blk"],
        "3PM": team_sample_raw["3PM"],
        "TO": team_sample_raw["TO"]
    })

    team_sample["FG%"] = (
        team_sample["FG%"]
        .round(3)
    )

    team_sample["FT%"] = (
        team_sample["FT%"]
        .round(3)
    )

    for stat in [
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO"
    ]:
        team_sample[stat] = (
            team_sample[stat]
            .round(1)
        )

    # ========================================================
    # PLAYER BREAKDOWN
    #
    # Future games only
    # ========================================================

    player_sample_frames = []

    for player_name, player_future in player_results.items():

        player_sample_raw = (
            player_future
            .iloc[sampled_indices]
            .reset_index(drop=True)
            .copy()
        )

        player_sample_raw["FG%"] = np.where(
            player_sample_raw["FGA"] > 0,
            (
                player_sample_raw["FGM"]
                / player_sample_raw["FGA"]
            ),
            0.0
        )

        player_sample_raw["FT%"] = np.where(
            player_sample_raw["FTA"] > 0,
            (
                player_sample_raw["FTM"]
                / player_sample_raw["FTA"]
            ),
            0.0
        )

        player_display = pd.DataFrame({
            "Simulation": sampled_simulation_ids,
            "Player": player_name,
            "FG%": player_sample_raw["FG%"],
            "FT%": player_sample_raw["FT%"],
            "Pts": player_sample_raw["Pts"],
            "Reb": player_sample_raw["Reb"],
            "Ast": player_sample_raw["Ast"],
            "Stl": player_sample_raw["Stl"],
            "Blk": player_sample_raw["Blk"],
            "3PM": player_sample_raw["3PM"],
            "TO": player_sample_raw["TO"]
        })

        player_display["FG%"] = (
            player_display["FG%"]
            .round(3)
        )

        player_display["FT%"] = (
            player_display["FT%"]
            .round(3)
        )

        for stat in [
            "Pts",
            "Reb",
            "Ast",
            "Stl",
            "Blk",
            "3PM",
            "TO"
        ]:
            player_display[stat] = (
                player_display[stat]
                .round(1)
            )

        player_sample_frames.append(
            player_display
        )

    if player_sample_frames:

        player_sample = pd.concat(
            player_sample_frames,
            ignore_index=True
        )

        player_sample = (
            player_sample
            .sort_values(
                [
                    "Simulation",
                    "Player"
                ]
            )
            .reset_index(drop=True)
        )

    else:

        player_sample = pd.DataFrame(
            columns=[
                "Simulation",
                "Player",
                "FG%",
                "FT%",
                "Pts",
                "Reb",
                "Ast",
                "Stl",
                "Blk",
                "3PM",
                "TO"
            ]
        )

    return {
        "team_sample": team_sample,
        "player_sample": player_sample,
        "team_future_all": team_future,
        "player_future_all": player_results,
        "sampled_simulation_ids": (
            sampled_simulation_ids.tolist()
        )
    }

# ============================================================
# 11. FAST MATCHUP SIMULATION
# ============================================================

def simulate_matchup_fast(
    prepared_my_roster,
    prepared_opp_roster,
    current_my,
    current_opp,
    n_sim=100000,
    random_seed=12345
):

    rng = np.random.default_rng(
        random_seed
    )

    # ========================================================
    # SIMULATE ENTIRE FUTURE
    # ========================================================

    my_future = simulate_roster_vectorized(
        prepared_roster=prepared_my_roster,
        n_sim=n_sim,
        rng=rng
    )

    opp_future = simulate_roster_vectorized(
        prepared_roster=prepared_opp_roster,
        n_sim=n_sim,
        rng=rng
    )

    # ========================================================
    # FG%
    # ========================================================

    my_FGM = (
        float(current_my["FGM"])
        + my_future["FGM"]
    )

    my_FGA = (
        float(current_my["FGA"])
        + my_future["FGA"]
    )

    opp_FGM = (
        float(current_opp["FGM"])
        + opp_future["FGM"]
    )

    opp_FGA = (
        float(current_opp["FGA"])
        + opp_future["FGA"]
    )

    my_FG_pct = pd.Series(
        np.divide(
            my_FGM.to_numpy(dtype=float),
            my_FGA.to_numpy(dtype=float),
            out=np.zeros(n_sim, dtype=float),
            where=(
                my_FGA.to_numpy(dtype=float) != 0
            )
        ),
        index=my_future.index
    )

    opp_FG_pct = pd.Series(
        np.divide(
            opp_FGM.to_numpy(dtype=float),
            opp_FGA.to_numpy(dtype=float),
            out=np.zeros(n_sim, dtype=float),
            where=(
                opp_FGA.to_numpy(dtype=float) != 0
            )
        ),
        index=opp_future.index
    )

    # ========================================================
    # FT%
    # ========================================================

    my_FTM = (
        float(current_my["FTM"])
        + my_future["FTM"]
    )

    my_FTA = (
        float(current_my["FTA"])
        + my_future["FTA"]
    )

    opp_FTM = (
        float(current_opp["FTM"])
        + opp_future["FTM"]
    )

    opp_FTA = (
        float(current_opp["FTA"])
        + opp_future["FTA"]
    )

    my_FT_pct = pd.Series(
        np.divide(
            my_FTM.to_numpy(dtype=float),
            my_FTA.to_numpy(dtype=float),
            out=np.zeros(n_sim, dtype=float),
            where=(
                my_FTA.to_numpy(dtype=float) != 0
            )
        ),
        index=my_future.index
    )

    opp_FT_pct = pd.Series(
        np.divide(
            opp_FTM.to_numpy(dtype=float),
            opp_FTA.to_numpy(dtype=float),
            out=np.zeros(n_sim, dtype=float),
            where=(
                opp_FTA.to_numpy(dtype=float) != 0
            )
        ),
        index=opp_future.index
    )

    # ========================================================
    # WIN AND MARGIN MATRICES
    # ========================================================

    categories = [
        "FG%",
        "FT%",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO"
    ]

    win_df = pd.DataFrame(
        0,
        index=np.arange(n_sim),
        columns=categories,
        dtype=int
    )

    margin_df = pd.DataFrame(
        0.0,
        index=np.arange(n_sim),
        columns=categories
    )

    # ========================================================
    # FG%
    # ========================================================

    win_df["FG%"] = (
        my_FG_pct
        > opp_FG_pct
    ).astype(int)

    margin_df["FG%"] = (
        my_FG_pct
        - opp_FG_pct
    )

    # ========================================================
    # FT%
    # ========================================================

    win_df["FT%"] = (
        my_FT_pct
        > opp_FT_pct
    ).astype(int)

    margin_df["FT%"] = (
        my_FT_pct
        - opp_FT_pct
    )

    # ========================================================
    # COUNTING CATEGORIES
    # ========================================================

    counting_categories = [
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM"
    ]

    for category in counting_categories:

        my_total = (
            float(current_my[category])
            + my_future[category]
        )

        opp_total = (
            float(current_opp[category])
            + opp_future[category]
        )

        win_df[category] = (
            my_total
            > opp_total
        ).astype(int)

        margin_df[category] = (
            my_total
            - opp_total
        )

    # ========================================================
    # TURNOVERS
    #
    # Lower is better.
    # The margin remains My TO minus Opponent TO.
    # Therefore, a negative margin is favorable for My Team.
    # ========================================================

    my_TO = (
        float(current_my["TO"])
        + my_future["TO"]
    )

    opp_TO = (
        float(current_opp["TO"])
        + opp_future["TO"]
    )

    win_df["TO"] = (
        my_TO
        < opp_TO
    ).astype(int)

    margin_df["TO"] = (
        my_TO
        - opp_TO
    )

    # ========================================================
    # CATEGORY RESULTS
    # ========================================================

    category_results = {}

    for category in categories:

        margins = (
            margin_df[category]
            .to_numpy(dtype=float)
        )

        category_results[category] = {
            "win_prob": float(
                win_df[category].mean()
            ),
            "avg_margin": float(
                margins.mean()
            ),
            "p5": float(
                np.quantile(
                    margins,
                    0.05
                )
            ),
            "p50": float(
                np.quantile(
                    margins,
                    0.50
                )
            ),
            "p95": float(
                np.quantile(
                    margins,
                    0.95
                )
            )
        }

    # ========================================================
    # FINAL SCORE FOR EVERY SIMULATION
    #
    # win_df is an n_sim × 9 matrix containing zeros and ones.
    #
    # Summing each row is equivalent to taking its dot product
    # with the vector:
    #
    # [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #
    # The result is an array of length n_sim containing values
    # from 0 through 9.
    # ========================================================

    category_win_matrix = (
        win_df[categories]
        .to_numpy(dtype=int)
    )

    ones_vector = np.ones(
        len(categories),
        dtype=int
    )

    total_wins = (
        category_win_matrix
        @ ones_vector
    ).astype(int)

    # ========================================================
    # SCORE DISTRIBUTION
    #
    # Example:
    # 0 wins -> 0:9
    # 5 wins -> 5:4
    # 9 wins -> 9:0
    # ========================================================

    score_counts = np.bincount(
        total_wins,
        minlength=10
    )

    score_probabilities = (
        score_counts
        / float(n_sim)
    )

    score_distribution = {
        wins: float(
            score_probabilities[wins]
        )
        for wins in range(10)
    }

    score_distribution_df = pd.DataFrame({
        "Wins": np.arange(
            10,
            dtype=int
        ),
        "Losses": np.arange(
            9,
            -1,
            -1,
            dtype=int
        ),
        "Count": score_counts.astype(int),
        "Probability": score_probabilities.astype(float)
    })

    score_distribution_df["Score"] = (
        score_distribution_df["Wins"]
        .astype(str)
        + ":"
        + score_distribution_df["Losses"]
        .astype(str)
    )

    score_distribution_df = (
        score_distribution_df[
            [
                "Score",
                "Wins",
                "Losses",
                "Count",
                "Probability"
            ]
        ]
        .reset_index(drop=True)
    )

    # ========================================================
    # CUMULATIVE MATCHUP PROBABILITIES
    # ========================================================

    p_ge_5 = float(
        np.mean(
            total_wins >= 5
        )
    )

    p_ge_6 = float(
        np.mean(
            total_wins >= 6
        )
    )

    p_ge_7 = float(
        np.mean(
            total_wins >= 7
        )
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    probability_sum = float(
        score_distribution_df[
            "Probability"
        ].sum()
    )

    count_sum = int(
        score_distribution_df[
            "Count"
        ].sum()
    )

    if count_sum != int(n_sim):

        raise ValueError(
            "Score-distribution count does not equal n_sim. "
            f"Count={count_sum}, n_sim={n_sim}"
        )

    if not np.isclose(
        probability_sum,
        1.0
    ):

        raise ValueError(
            "Score-distribution probabilities do not sum to 1. "
            f"Sum={probability_sum}"
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "categories": category_results,
        "score_distribution": score_distribution,
        "score_distribution_df": score_distribution_df,
        "p_ge_5": p_ge_5,
        "p_ge_6": p_ge_6,
        "p_ge_7": p_ge_7,
        "total_wins": total_wins,
        "win_df": win_df,
        "margin_df": margin_df,
        "my_future": my_future,
        "opp_future": opp_future
    }

# ============================================================
# 12. CREATE ADD / DROP ROSTER
#
# UPDATED:
# Add Player 也抓 Multi-Season Pool
# ============================================================

def create_add_drop_roster(
    baseline_roster,
    drop_player,
    add_player,
    seasons=SEASONS
):

    if drop_player not in baseline_roster:

        raise ValueError(
            f"{drop_player} 不在目前 roster 裡"
        )

    if add_player in baseline_roster:

        raise ValueError(
            f"{add_player} 已經在目前 roster 裡"
        )

    print("\n" + "=" * 60)
    print("CREATING ADD / DROP ROSTER")
    print("=" * 60)

    print(
        f"DROP : {drop_player}"
    )

    print(
        f"ADD  : {add_player}"
    )

    print(
        f"\nLoading multi-season data for {add_player}..."
    )

    add_df = (
        get_player_df_from_api(
            add_player,
            seasons=seasons
        )
    )

    scenario_roster = {}

    for (
        name,
        df
    ) in baseline_roster.items():

        if name == drop_player:

            scenario_roster[
                add_player
            ] = add_df

        else:

            scenario_roster[
                name
            ] = df

    return scenario_roster



# ============================================================
# 13. CREATE ADD / DROP HEATMAP FIGURE
# ============================================================


def create_add_drop_heatmap(
    category_comparison,
    prepared_baseline
):
    """
    建立三層 Heatmap：

    1. Baseline 與 Add/Drop 的九項勝率
    2. Add/Drop 對各類別勝率的改變
    3. Baseline roster 球員對各類別的預期貢獻比例

    Heatmap 3 使用 prepared_baseline 中每一場未來比賽的
    weighted_df 計算球員預期累積數字。

    Counting categories:
        player expected total / roster expected total

    FG%:
        player expected FGM / roster expected FGM

    FT%:
        player expected FTM / roster expected FTM

    TO:
        顯示球員占全隊失誤的比例，不做反向處理。
    """

    comparison_df = (
        category_comparison.copy()
    )

    required_columns = [
        "Category",
        "Baseline",
        "AddDrop",
        "Delta"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in comparison_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "category_comparison 缺少欄位："
            f"{missing_columns}"
        )

    # ========================================================
    # CONVERT PROBABILITIES TO 0-100 WHEN NEEDED
    # ========================================================

    probability_max = (
        comparison_df[
            [
                "Baseline",
                "AddDrop"
            ]
        ]
        .abs()
        .max()
        .max()
    )

    if probability_max <= 1.000001:

        comparison_df[
            [
                "Baseline",
                "AddDrop",
                "Delta"
            ]
        ] = (
            comparison_df[
                [
                    "Baseline",
                    "AddDrop",
                    "Delta"
                ]
            ]
            * 100
        )

    # ========================================================
    # HEATMAP 1 DATA
    # ========================================================

    category_heat = (
        comparison_df
        .set_index("Category")[
            [
                "Baseline",
                "AddDrop"
            ]
        ]
        .T
    )

    category_heat.index = [
        "Baseline",
        "Add / Drop"
    ]

    # ========================================================
    # HEATMAP 2 DATA
    # ========================================================

    delta_heat = pd.DataFrame(
        [
            comparison_df
            .set_index("Category")[
                "Delta"
            ]
        ],
        index=[
            "Add / Drop Delta"
        ]
    )

    # ========================================================
    # HEATMAP 3
    #
    # EXPECTED PLAYER TOTALS FROM PREPARED WEIGHTED DATA
    # ========================================================

    raw_stat_columns = [
        "FGM",
        "FGA",
        "FTM",
        "FTA",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO"
    ]

    contribution_categories = [
        "FG%",
        "FT%",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO"
    ]

    player_expected_rows = []

    for (
        player_name,
        prepared_player
    ) in prepared_baseline.items():

        expected_totals = {
            stat: 0.0
            for stat in raw_stat_columns
        }

        prepared_games = (
            prepared_player.get(
                "games",
                []
            )
        )

        for game_info in prepared_games:

            weighted_df = game_info.get(
                "weighted_df"
            )

            if (
                weighted_df is None
                or not isinstance(
                    weighted_df,
                    pd.DataFrame
                )
                or weighted_df.empty
            ):

                continue

            if "Weight" not in weighted_df.columns:

                continue

            weights = pd.to_numeric(
                weighted_df["Weight"],
                errors="coerce"
            ).fillna(0.0)

            weight_sum = float(
                weights.sum()
            )

            if weight_sum <= 0:

                continue

            normalized_weights = (
                weights
                / weight_sum
            )

            weight_array = (
                normalized_weights
                .to_numpy(dtype=float)
            )

            for stat in raw_stat_columns:

                if stat not in weighted_df.columns:

                    continue

                stat_values = pd.to_numeric(
                    weighted_df[stat],
                    errors="coerce"
                ).fillna(0.0)

                expected_value = float(
                    np.sum(
                        stat_values.to_numpy(
                            dtype=float
                        )
                        * weight_array
                    )
                )

                expected_totals[stat] += (
                    expected_value
                )

        player_expected_rows.append({
            "Player": player_name,
            **expected_totals
        })

    player_expected_df = pd.DataFrame(
        player_expected_rows
    )

    # ========================================================
    # CONVERT EXPECTED TOTALS INTO TEAM SHARES
    # ========================================================

    if player_expected_df.empty:

        contribution_heat = pd.DataFrame(
            0.0,
            index=[
                "No player data"
            ],
            columns=contribution_categories
        )

    else:

        player_expected_df = (
            player_expected_df
            .set_index("Player")
        )

        roster_totals = (
            player_expected_df[
                raw_stat_columns
            ]
            .sum(axis=0)
        )

        contribution_heat = pd.DataFrame(
            0.0,
            index=player_expected_df.index,
            columns=contribution_categories
        )

        # ====================================================
        # FG CONTRIBUTION
        #
        # Share of the roster's expected made field goals.
        # ====================================================

        roster_fgm = float(
            roster_totals["FGM"]
        )

        if roster_fgm > 0:

            contribution_heat["FG%"] = (
                player_expected_df["FGM"]
                / roster_fgm
                * 100
            )

        # ====================================================
        # FT CONTRIBUTION
        #
        # Share of the roster's expected made free throws.
        # ====================================================

        roster_ftm = float(
            roster_totals["FTM"]
        )

        if roster_ftm > 0:

            contribution_heat["FT%"] = (
                player_expected_df["FTM"]
                / roster_ftm
                * 100
            )

        # ====================================================
        # COUNTING CATEGORY CONTRIBUTIONS
        # ====================================================

        counting_categories = [
            "Pts",
            "Reb",
            "Ast",
            "Stl",
            "Blk",
            "3PM",
            "TO"
        ]

        for category in counting_categories:

            roster_category_total = float(
                roster_totals[category]
            )

            if roster_category_total > 0:

                contribution_heat[category] = (
                    player_expected_df[category]
                    / roster_category_total
                    * 100
                )

        contribution_heat = (
            contribution_heat
            .fillna(0.0)
            .round(1)
        )

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPECTED PLAYER FUTURE TOTALS"
    )

    print(
        "=" * 70
    )

    if not player_expected_df.empty:

        print(
            player_expected_df
            .round(2)
            .to_string()
        )

    print(
        "\nPLAYER CATEGORY CONTRIBUTION (%)"
    )

    print(
        "-" * 70
    )

    print(
        contribution_heat.to_string()
    )

    print(
        "\nCONTRIBUTION COLUMN SUM CHECK"
    )

    print(
        "-" * 70
    )

    print(
        contribution_heat
        .sum(axis=0)
        .round(1)
    )

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    player_count = max(
        len(contribution_heat),
        1
    )

    contribution_height = max(
        3.5,
        player_count * 0.48
    )

    figure_height = (
        7.0
        + contribution_height
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            15,
            figure_height
        ),
        gridspec_kw={
            "height_ratios": [
                1.35,
                0.85,
                contribution_height
            ]
        }
    )

    # ========================================================
    # HEATMAP 1
    #
    # NINE-CATEGORY WIN PROBABILITY
    # ========================================================

    sns.heatmap(
        category_heat,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        vmin=0,
        vmax=100,
        center=50,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": (
                "Win probability (%)"
            )
        },
        ax=axes[0]
    )

    axes[0].set_title(
        "9-Cat Win Probability: Baseline vs Add/Drop (%)",
        fontsize=13,
        fontweight="bold"
    )

    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    axes[0].tick_params(
        axis="x",
        rotation=0
    )

    axes[0].tick_params(
        axis="y",
        rotation=0
    )

    # ========================================================
    # HEATMAP 2
    #
    # CATEGORY DELTA
    # ========================================================

    delta_values = (
        delta_heat
        .to_numpy(dtype=float)
    )

    finite_delta_values = (
        delta_values[
            np.isfinite(
                delta_values
            )
        ]
    )

    if finite_delta_values.size == 0:

        max_abs_delta = 5.0

    else:

        max_abs_delta = max(
            5.0,
            float(
                np.max(
                    np.abs(
                        finite_delta_values
                    )
                )
            )
        )

    sns.heatmap(
        delta_heat,
        annot=True,
        fmt="+.1f",
        cmap="coolwarm",
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": (
                "Percentage-point change"
            )
        },
        ax=axes[1]
    )

    axes[1].set_title(
        "Change in Category Win Probability "
        "(percentage points)",
        fontsize=13,
        fontweight="bold"
    )

    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    axes[1].tick_params(
        axis="x",
        rotation=0
    )

    axes[1].tick_params(
        axis="y",
        rotation=0
    )

    # ========================================================
    # HEATMAP 3
    #
    # TEAM CATEGORY CONTRIBUTION
    # ========================================================

    contribution_values = (
        contribution_heat
        .to_numpy(dtype=float)
    )

    finite_contribution_values = (
        contribution_values[
            np.isfinite(
                contribution_values
            )
        ]
    )

    if finite_contribution_values.size == 0:

        heatmap_max = 10.0

    else:

        heatmap_max = max(
            10.0,
            float(
                np.ceil(
                    np.max(
                        finite_contribution_values
                    )
                    / 5.0
                )
                * 5.0
            )
        )

    sns.heatmap(
        contribution_heat,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        vmin=0,
        vmax=heatmap_max,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": (
                "Expected team contribution (%)"
            )
        },
        ax=axes[2]
    )

    axes[2].set_title(
        "Baseline Team Category Contribution (%)",
        fontsize=13,
        fontweight="bold"
    )

    axes[2].set_xlabel(
        "Fantasy Category"
    )

    axes[2].set_ylabel(
        "Player"
    )

    axes[2].tick_params(
        axis="x",
        rotation=0
    )

    axes[2].tick_params(
        axis="y",
        rotation=0
    )

    fig.text(
        0.5,
        0.012,
        (
            "Heatmap 3 shows each player's contribution to the baseline roster's projected production. "
            "Higher values indicate greater impact on the corresponding category."
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1
        ]
    )

    return fig

# ============================================================
# FAST ADD / DROP SCENARIO
# ============================================================

def compare_add_drop_fast(
    baseline_roster,
    opp_roster,
    schedule_df,
    start_date,
    end_date,
    current_my,
    current_opp,
    add_player,
    drop_player,
    seasons=SEASONS,
    n_sim=3000,
    random_seed=12345
):

    # ========================================================
    # CREATE ADD / DROP SCENARIO ROSTER
    # ========================================================

    if str(add_player).strip():

        scenario_roster = (
            create_add_drop_roster(
                baseline_roster=baseline_roster,
                drop_player=drop_player,
                add_player=add_player,
                seasons=seasons
            )
        )

    else:

        scenario_roster = baseline_roster.copy()

    scenario_roster.pop(
        drop_player,
        None
    )

    # ========================================================
    # PREPARE BASELINE WEIGHTS
    # ========================================================

    print(
        "\nPreparing BASELINE weights..."
    )

    prepared_baseline = prepare_roster_simulation(
        roster=baseline_roster,
        schedule_df=schedule_df,
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================
    # PREPARE SCENARIO WEIGHTS
    # ========================================================

    print(
        "Preparing SCENARIO weights..."
    )

    prepared_scenario = prepare_roster_simulation(
        roster=scenario_roster,
        schedule_df=schedule_df,
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================
    # PREPARE OPPONENT WEIGHTS
    # ========================================================

    print(
        "Preparing OPPONENT weights..."
    )

    prepared_opp = prepare_roster_simulation(
        roster=opp_roster,
        schedule_df=schedule_df,
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================
    # RUN BASELINE MONTE CARLO
    # ========================================================

    print(
        "\nRunning BASELINE..."
    )

    baseline = simulate_matchup_fast(
        prepared_my_roster=prepared_baseline,
        prepared_opp_roster=prepared_opp,
        current_my=current_my,
        current_opp=current_opp,
        n_sim=n_sim,
        random_seed=random_seed
    )

    # ========================================================
    # RUN ADD / DROP MONTE CARLO
    # ========================================================

    print(
        "Running ADD / DROP..."
    )

    scenario = simulate_matchup_fast(
        prepared_my_roster=prepared_scenario,
        prepared_opp_roster=prepared_opp,
        current_my=current_my,
        current_opp=current_opp,
        n_sim=n_sim,
        random_seed=random_seed
    )

    # ========================================================
    # CATEGORY COMPARISON
    # ========================================================

    categories = [
        "FG%",
        "FT%",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "3PM",
        "TO"
    ]

    category_rows = []

    for category in categories:

        baseline_probability = float(
            baseline["categories"][
                category
            ]["win_prob"]
        )

        scenario_probability = float(
            scenario["categories"][
                category
            ]["win_prob"]
        )

        category_rows.append({
            "Category": category,
            "Baseline": baseline_probability,
            "AddDrop": scenario_probability,
            "Delta": (
                scenario_probability
                - baseline_probability
            )
        })

    comparison_df = pd.DataFrame(
        category_rows
    )

    # ========================================================
    # OVERALL MATCHUP COMPARISON
    # ========================================================

    overall = pd.DataFrame({
        "Target": [
            "P(>=5)",
            "P(>=6)",
            "P(>=7)"
        ],
        "Baseline": [
            baseline["p_ge_5"],
            baseline["p_ge_6"],
            baseline["p_ge_7"]
        ],
        "AddDrop": [
            scenario["p_ge_5"],
            scenario["p_ge_6"],
            scenario["p_ge_7"]
        ]
    })

    overall["Delta"] = (
        overall["AddDrop"]
        - overall["Baseline"]
    )

    # ========================================================
    # SCORE DISTRIBUTION COMPARISON
    #
    # Expected scores:
    # 0:9, 1:8, 2:7, ..., 9:0
    # ========================================================

    baseline_score_distribution = (
        baseline["score_distribution_df"]
        .copy()
    )

    baseline_score_distribution[
        "Scenario"
    ] = "Baseline"

    scenario_score_distribution = (
        scenario["score_distribution_df"]
        .copy()
    )

    scenario_score_distribution[
        "Scenario"
    ] = "Add / Drop"

    score_distribution_comparison = pd.concat(
        [
            baseline_score_distribution,
            scenario_score_distribution
        ],
        ignore_index=True
    )

    score_order = [
        f"{wins}:{9 - wins}"
        for wins in range(10)
    ]

    score_distribution_comparison[
        "Score"
    ] = pd.Categorical(
        score_distribution_comparison["Score"],
        categories=score_order,
        ordered=True
    )

    score_distribution_comparison = (
        score_distribution_comparison
        .sort_values(
            [
                "Score",
                "Scenario"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SCORE DISTRIBUTION CONSISTENCY CHECK
    # ========================================================

    baseline_probability_sum = float(
        baseline_score_distribution[
            "Probability"
        ].sum()
    )

    scenario_probability_sum = float(
        scenario_score_distribution[
            "Probability"
        ].sum()
    )

    if not np.isclose(
        baseline_probability_sum,
        1.0
    ):

        raise ValueError(
            "Baseline score probabilities do not sum to 1. "
            f"Sum={baseline_probability_sum}"
        )

    if not np.isclose(
        scenario_probability_sum,
        1.0
    ):

        raise ValueError(
            "Add/Drop score probabilities do not sum to 1. "
            f"Sum={scenario_probability_sum}"
        )

    # ========================================================
    # VERIFY P(>=5), P(>=6), P(>=7)
    #
    # This checks that the cumulative probabilities agree
    # with the score-distribution table.
    # ========================================================

    baseline_p_ge_5_check = float(
        baseline_score_distribution.loc[
            baseline_score_distribution[
                "Wins"
            ] >= 5,
            "Probability"
        ].sum()
    )

    baseline_p_ge_6_check = float(
        baseline_score_distribution.loc[
            baseline_score_distribution[
                "Wins"
            ] >= 6,
            "Probability"
        ].sum()
    )

    baseline_p_ge_7_check = float(
        baseline_score_distribution.loc[
            baseline_score_distribution[
                "Wins"
            ] >= 7,
            "Probability"
        ].sum()
    )

    scenario_p_ge_5_check = float(
        scenario_score_distribution.loc[
            scenario_score_distribution[
                "Wins"
            ] >= 5,
            "Probability"
        ].sum()
    )

    scenario_p_ge_6_check = float(
        scenario_score_distribution.loc[
            scenario_score_distribution[
                "Wins"
            ] >= 6,
            "Probability"
        ].sum()
    )

    scenario_p_ge_7_check = float(
        scenario_score_distribution.loc[
            scenario_score_distribution[
                "Wins"
            ] >= 7,
            "Probability"
        ].sum()
    )

    if not np.isclose(
        baseline_p_ge_5_check,
        baseline["p_ge_5"]
    ):
        raise ValueError(
            "Baseline P(>=5) does not match "
            "the score distribution."
        )

    if not np.isclose(
        baseline_p_ge_6_check,
        baseline["p_ge_6"]
    ):
        raise ValueError(
            "Baseline P(>=6) does not match "
            "the score distribution."
        )

    if not np.isclose(
        baseline_p_ge_7_check,
        baseline["p_ge_7"]
    ):
        raise ValueError(
            "Baseline P(>=7) does not match "
            "the score distribution."
        )

    if not np.isclose(
        scenario_p_ge_5_check,
        scenario["p_ge_5"]
    ):
        raise ValueError(
            "Add/Drop P(>=5) does not match "
            "the score distribution."
        )

    if not np.isclose(
        scenario_p_ge_6_check,
        scenario["p_ge_6"]
    ):
        raise ValueError(
            "Add/Drop P(>=6) does not match "
            "the score distribution."
        )

    if not np.isclose(
        scenario_p_ge_7_check,
        scenario["p_ge_7"]
    ):
        raise ValueError(
            "Add/Drop P(>=7) does not match "
            "the score distribution."
        )

    # ========================================================
    # CREATE HEATMAP FIGURE
    # ========================================================

    heatmap_fig = create_add_drop_heatmap(
        category_comparison=comparison_df,
        prepared_baseline=prepared_baseline
    )

    # ========================================================
    # MONTE CARLO INSPECTION SAMPLES
    #
    # Uses the same n_sim selected in app.py.
    # Only five Simulation IDs are displayed.
    # ========================================================

    baseline_debug = create_simulation_debug_sample(
        prepared_roster=prepared_baseline,
        current_totals=current_my,
        n_sim=n_sim,
        sample_size=7,
        simulation_seed=random_seed,
        sample_seed=24680
    )
    
    opp_debug = create_simulation_debug_sample(
        prepared_roster=prepared_opp,
        current_totals=current_opp,
        n_sim=n_sim,
        sample_size=7,
        simulation_seed=random_seed,
        sample_seed=24680
    )

    scenario_debug = create_simulation_debug_sample(
        prepared_roster=prepared_scenario,
        current_totals=current_my,
        n_sim=n_sim,
        sample_size=7,
        simulation_seed=random_seed,
        sample_seed=24680
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ADD / DROP SCENARIO ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"ADD  : {add_player}"
    )

    print(
        f"DROP : {drop_player}"
    )

    display_category = (
        comparison_df.copy()
    )

    for column in [
        "Baseline",
        "AddDrop",
        "Delta"
    ]:

        display_category[column] = (
            display_category[column]
            * 100
        ).round(1)

    print(
        "\nCATEGORY WIN PROBABILITY (%)"
    )

    print(
        "-" * 70
    )

    print(
        display_category.to_string(
            index=False
        )
    )

    display_overall = (
        overall.copy()
    )

    for column in [
        "Baseline",
        "AddDrop",
        "Delta"
    ]:

        display_overall[column] = (
            display_overall[column]
            * 100
        ).round(1)

    print(
        "\nMATCHUP TARGET PROBABILITY (%)"
    )

    print(
        "-" * 70
    )

    print(
        display_overall.to_string(
            index=False
        )
    )

    display_scores = (
        score_distribution_comparison[
            [
                "Score",
                "Scenario",
                "Count",
                "Probability"
            ]
        ]
        .copy()
    )

    display_scores["Probability"] = (
        display_scores["Probability"]
        * 100
    ).round(2)

    print(
        "\nFINAL SCORE DISTRIBUTION (%)"
    )

    print(
        "-" * 70
    )

    print(
        display_scores.to_string(
            index=False
        )
    )

    # ========================================================
    # RETURN COMPLETE RESULTS
    # ========================================================

    return {
        "baseline": baseline,
        "scenario": scenario,
        "category_comparison": comparison_df,
        "overall_comparison": overall,
        "score_distribution_comparison": (
            score_distribution_comparison
        ),
        "scenario_roster": scenario_roster,
        "prepared_baseline": prepared_baseline,
        "prepared_scenario": prepared_scenario,
        "prepared_opp": prepared_opp,
        "heatmap_fig": heatmap_fig,
        "baseline_debug": baseline_debug,
        "opp_debug": opp_debug,
        "scenario_debug": scenario_debug
    }