# Imports

import pickle
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
import unicodedata

import matplotlib.pyplot as plt
import seaborn as sns



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
# YAHOO DATA SOURCE
# ============================================================
import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import quote

YAHOO_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
YAHOO_NS = {"f": "http://fantasysports.yahooapis.com/fantasy/v2/base.rng"}
YAHOO_STAT_IDS = {
    "3": "FGA", "4": "FGM", "6": "FTA", "7": "FTM",
    "10": "3PM", "12": "Pts", "15": "Reb", "16": "Ast",
    "17": "Stl", "18": "Blk", "19": "TO",
}
SCHEDULE_CACHE_PATH = Path(__file__).resolve().parent / "nba_schedule_2018_2027.csv"
YAHOO_DAILY_CACHE_DIR = Path(__file__).resolve().parent / "yahoo_daily_cache"
YAHOO_DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
YAHOO_GAME_KEY_CACHE_PATH = (
    Path(__file__).resolve().parent / "yahoo_nba_game_keys.json"
)
YAHOO_PLAYER_KEY_CACHE_PATH = (
    Path(__file__).resolve().parent / "yahoo_player_keys.json"
)



def _local_name(tag):
    return str(tag).split("}")[-1]


def _child_text(element, name, default=None):
    if element is None:
        return default
    for child in list(element):
        if _local_name(child.tag) == name:
            value = child.text
            return value.strip() if value else default
    return default


def _descendants(element, name):
    if element is None:
        return []
    return [node for node in element.iter() if _local_name(node.tag) == name]


def _safe_number(value):
    if value in (None, "", "-", "--"):
        return 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def _get_yahoo_app_credentials():
    key = os.getenv("YAHOO_CONSUMER_KEY")
    secret = os.getenv("YAHOO_CONSUMER_SECRET")
    try:
        key = key or st.secrets.get("YAHOO_CONSUMER_KEY")
        secret = secret or st.secrets.get("YAHOO_CONSUMER_SECRET")
    except Exception:
        pass
    if not key or not secret:
        raise RuntimeError("Missing YAHOO_CONSUMER_KEY / YAHOO_CONSUMER_SECRET.")
    return key, secret


def get_yahoo_authorization_url():
    key, _ = _get_yahoo_app_credentials()
    return requests.Request("GET", "https://api.login.yahoo.com/oauth2/request_auth", params={
        "client_id": key, "redirect_uri": "oob", "response_type": "code", "language": "en-us"
    }).prepare().url


def exchange_yahoo_verifier(verifier_code):
    code = str(verifier_code or "").strip()
    if not code:
        raise ValueError("Yahoo Verify Code 不可為空白。")
    key, secret = _get_yahoo_app_credentials()
    r = requests.post("https://api.login.yahoo.com/oauth2/get_token", auth=(key, secret),
        data={"grant_type": "authorization_code", "redirect_uri": "oob", "code": code}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo token exchange HTTP {r.status_code}: {r.text[:500]}")
    token = r.json(); token["token_time"] = pd.Timestamp.utcnow().timestamp()
    return token


def refresh_yahoo_token(token):
    key, secret = _get_yahoo_app_credentials()
    old_refresh = token.get("refresh_token")
    if not old_refresh:
        raise RuntimeError("Yahoo token 缺少 refresh_token，請重新授權。")
    r = requests.post("https://api.login.yahoo.com/oauth2/get_token", auth=(key, secret),
        data={"grant_type": "refresh_token", "redirect_uri": "oob", "refresh_token": old_refresh}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo token refresh HTTP {r.status_code}: {r.text[:500]}")
    refreshed = r.json(); refreshed["refresh_token"] = refreshed.get("refresh_token") or old_refresh
    refreshed["token_time"] = pd.Timestamp.utcnow().timestamp()
    token.clear(); token.update(refreshed)


def _yahoo_get_xml(path, params=None, timeout=60):
    token = st.session_state.get("yahoo_token")
    if not isinstance(token, dict) or not token.get("access_token"):
        raise RuntimeError("Yahoo 尚未完成網頁授權。")
    now = pd.Timestamp.utcnow().timestamp()
    if now >= float(token.get("token_time", 0)) + float(token.get("expires_in", 3600)) - 120:
        refresh_yahoo_token(token)
    def send():
        return requests.get(f"{YAHOO_BASE}/{path.lstrip('/')}", params=params,
            headers={"Authorization": f"Bearer {token['access_token']}"}, timeout=timeout)
    response = send()
    if response.status_code == 401:
        refresh_yahoo_token(token); response = send()
    if response.status_code != 200:
        raise RuntimeError(f"Yahoo API HTTP {response.status_code}: {response.text[:500]}")
    return ET.fromstring(response.text)

def _extract_player(player_element):
    eligible_positions = []
    eligible_node = next(iter(_descendants(player_element, "eligible_positions")), None)
    if eligible_node is not None:
        for position_node in _descendants(eligible_node, "position"):
            value = (position_node.text or "").strip()
            if value and value not in eligible_positions:
                eligible_positions.append(value)

    return {
        "player_key": _child_text(player_element, "player_key"),
        "player_id": _child_text(player_element, "player_id"),
        "name": _child_text(
            next(iter(_descendants(player_element, "name")), None),
            "full",
        ),
        "team": _child_text(player_element, "editorial_team_abbr"),
        "status": _child_text(player_element, "status"),
        "selected_position": _child_text(
            next(iter(_descendants(player_element, "selected_position")), None),
            "position",
        ),
        "eligible_positions": eligible_positions,
    }


def get_my_yahoo_team_key(game_key, league_key=None):
    """Return the signed-in user's team key for the requested league."""
    root = _yahoo_get_xml(
        f"users;use_login=1/games;game_keys={game_key}/teams"
    )
    team_keys = [
        _child_text(team, "team_key")
        for team in _descendants(root, "team")
    ]
    team_keys = [value for value in team_keys if value]

    if league_key:
        league_prefix = f"{league_key}.t."
        league_team_keys = [
            key for key in team_keys
            if str(key).startswith(league_prefix)
        ]
        if len(league_team_keys) == 1:
            print(
                f"[YAHOO_MY_TEAM] league={league_key} | "
                f"team={league_team_keys[0]}"
            )
            return league_team_keys[0]
        if len(league_team_keys) > 1:
            raise RuntimeError(
                f"Multiple signed-in teams found for league {league_key}: "
                f"{league_team_keys}"
            )
        raise RuntimeError(
            f"Signed-in Yahoo account has no team in league {league_key}. "
            f"Returned team keys: {team_keys}"
        )

    if not team_keys:
        raise RuntimeError(f"No Yahoo team found for game key {game_key}")
    return team_keys[0]

def get_yahoo_league_context(league_key):
    """Return current Yahoo matchup, rosters and accumulated weekly stats."""
    game_key = str(league_key).split(".")[0]
    my_team_key = get_my_yahoo_team_key(game_key, league_key=league_key)

    league_root = _yahoo_get_xml(f"league/{league_key}")
    current_week = _child_text(
        next(iter(_descendants(league_root, "league")), None),
        "current_week",
    )
    if not current_week:
        raise RuntimeError("Yahoo league response did not contain current_week")

    scoreboard = _yahoo_get_xml(
        f"league/{league_key}/scoreboard;week={current_week}"
    )

    def extract_team_stats(team_element):
        """Extract Yahoo weekly matchup totals into engine raw-stat fields."""
        values = {}
        if team_element is None:
            return values

        team_stats_nodes = _descendants(team_element, "team_stats")
        search_root = team_stats_nodes[0] if team_stats_nodes else team_element

        for stat in _descendants(search_root, "stat"):
            stat_id = _child_text(stat, "stat_id")
            value = _child_text(stat, "value")

            # Yahoo weekly composite stats confirmed from the live response:
            # 9004003 = FGM/FGA, for example 150/307
            # 9007006 = FTM/FTA, for example 58/67
            if stat_id == "9004003" and value and "/" in str(value):
                made, attempted = str(value).split("/", 1)
                values["FGM"] = _safe_number(made.strip())
                values["FGA"] = _safe_number(attempted.strip())
                continue

            if stat_id == "9007006" and value and "/" in str(value):
                made, attempted = str(value).split("/", 1)
                values["FTM"] = _safe_number(made.strip())
                values["FTA"] = _safe_number(attempted.strip())
                continue

            # Normal scoring categories. stat_id 5 and 8 are percentages and
            # are intentionally not used for the raw FGM/FGA/FTM/FTA totals.
            if stat_id in YAHOO_STAT_IDS and value is not None:
                values[YAHOO_STAT_IDS[stat_id]] = _safe_number(value)

        print(
            "[YAHOO_WEEK_PARSED] "
            + " | ".join(f"{key}={value}" for key, value in values.items())
        )
        return values

    opponent_team_key = None
    my_scoreboard_team = None
    opp_scoreboard_team = None

    for matchup in _descendants(scoreboard, "matchup"):
        matchup_teams = _descendants(matchup, "team")
        keyed_teams = [
            (_child_text(team, "team_key"), team)
            for team in matchup_teams
            if _child_text(team, "team_key")
        ]
        matchup_team_keys = [key for key, _ in keyed_teams]

        if my_team_key in matchup_team_keys:
            my_scoreboard_team = next(
                team for key, team in keyed_teams if key == my_team_key
            )
            others = [
                (key, team) for key, team in keyed_teams
                if key != my_team_key
            ]
            if others:
                opponent_team_key, opp_scoreboard_team = others[0]
            break

    if not opponent_team_key:
        scoreboard_matchups = []
        for matchup_node in _descendants(scoreboard, "matchup"):
            keys = [
                _child_text(team, "team_key")
                for team in _descendants(matchup_node, "team")
                if _child_text(team, "team_key")
            ]
            if keys:
                scoreboard_matchups.append(keys)
        raise RuntimeError(
            "Could not identify the current Yahoo matchup opponent. "
            f"my_team_key={my_team_key}; "
            f"scoreboard_matchups={scoreboard_matchups}"
        )

    def load_roster(team_key):
        root = _yahoo_get_xml(f"team/{team_key}/roster")
        players_out = []
        for player in _descendants(root, "player"):
            item = _extract_player(player)
            if item["player_key"] and item["name"]:
                players_out.append(item)
        return players_out

    def load_week_stats(team_key, scoreboard_team):
        # Always inspect both Yahoo sources. Scoreboard contains the visible
        # matchup categories; team weekly stats can contain richer display data.
        scoreboard_values = extract_team_stats(scoreboard_team)

        team_root = _yahoo_get_xml(
            f"team/{team_key}/stats;type=week;week={current_week}"
        )
        team_nodes = _descendants(team_root, "team")
        team_values = extract_team_stats(
            team_nodes[0] if team_nodes else team_root
        )

        values = dict(scoreboard_values)
        values.update(team_values)

        required_fields = list(YAHOO_STAT_IDS.values())
        missing_fields = [
            field for field in required_fields
            if field not in values
        ]
        if missing_fields:
            print(
                f"[YAHOO_WEEK_WARNING] team={team_key} | "
                f"missing={missing_fields}"
            )

        complete = {
            field: int(round(values.get(field, 0)))
            for field in required_fields
        }
        print(
            f"[YAHOO_WEEK_STATS] team={team_key} | week={current_week} | "
            + " | ".join(f"{key}={value}" for key, value in complete.items())
        )
        return complete

    current_my = load_week_stats(my_team_key, my_scoreboard_team)
    current_opp = load_week_stats(opponent_team_key, opp_scoreboard_team)

    return {
        "league_key": league_key,
        "game_key": game_key,
        "current_week": int(current_week),
        "my_team_key": my_team_key,
        "opponent_team_key": opponent_team_key,
        "my_roster": load_roster(my_team_key),
        "opponent_roster": load_roster(opponent_team_key),
        "current_my": current_my,
        "current_opp": current_opp,
    }

def _season_start_year(season):
    return str(season).split("-")[0]


def get_yahoo_nba_game_keys(seasons=SEASONS, current_league_key=None):
    """Resolve Yahoo NBA game_key for every requested fantasy season."""
    requested = [str(season) for season in seasons]
    cached = {}
    if YAHOO_GAME_KEY_CACHE_PATH.exists():
        try:
            cached = json.loads(
                YAHOO_GAME_KEY_CACHE_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            cached = {}

    # Current league key is authoritative for the active season.
    if current_league_key:
        cached[CURRENT_SEASON] = str(current_league_key).split(".")[0]

    missing = [season for season in requested if not cached.get(season)]
    if missing:
        years = ",".join(_season_start_year(season) for season in missing)
        print(f"[YAHOO_GAME_KEYS] resolving NBA seasons: {', '.join(missing)}")
        root = _yahoo_get_xml(
            f"games;game_codes=nba;seasons={years}"
        )
        for game in _descendants(root, "game"):
            code = _child_text(game, "code")
            game_key = _child_text(game, "game_key")
            yahoo_season = _child_text(game, "season")
            if code != "nba" or not game_key or not yahoo_season:
                continue
            matching = [
                season for season in requested
                if _season_start_year(season) == str(yahoo_season)
            ]
            for season in matching:
                cached[season] = game_key
                print(f"[YAHOO_GAME_KEYS] {season} -> {game_key}")

    unresolved = [season for season in requested if not cached.get(season)]
    if unresolved:
        raise RuntimeError(
            "Yahoo game_key 未能解析的球季：" + ", ".join(unresolved)
        )

    YAHOO_GAME_KEY_CACHE_PATH.write_text(
        json.dumps(cached, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {season: cached[season] for season in requested}


def _exact_player_from_root(root, player_name):
    target = normalize_player_name(player_name)
    candidates = []
    for player in _descendants(root, "player"):
        item = _extract_player(player)
        if item.get("player_key") and item.get("name"):
            candidates.append(item)

    exact = [
        item for item in candidates
        if normalize_player_name(item["name"]) == target
    ]
    if len(exact) == 1:
        return exact[0]
    return None


def _load_yahoo_player_key_cache():
    if not YAHOO_PLAYER_KEY_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(
            YAHOO_PLAYER_KEY_CACHE_PATH.read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except Exception as error:
        print(
            f"[YAHOO_PLAYER_KEY_CACHE] read failed | "
            f"error={type(error).__name__}: {error}"
        )
        return {}


def _save_yahoo_player_key_cache(cache):
    temp_path = YAHOO_PLAYER_KEY_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(YAHOO_PLAYER_KEY_CACHE_PATH)


def _player_cache_identity(player_name):
    return normalize_player_name(player_name)


def find_yahoo_player(
    player_name,
    game_key,
    season,
    current_roster_player=None,
):
    """
    Resolve season-specific Yahoo player metadata.

    Rules:
    - Current season may use the roster player_key as a shortcut.
    - Historical seasons search directly by name inside that season.
    - There is no full-collection/page-scanning fallback.
    - A successful result or NOT_FOUND is cached by player + season.
    """
    cache = _load_yahoo_player_key_cache()
    identity = _player_cache_identity(player_name)
    player_cache = cache.setdefault(identity, {})
    cached_value = player_cache.get(season, "__MISSING__")

    if cached_value is None:
        # A previous Yahoo search failure must not permanently suppress a
        # season that really exists. Remove stale negative cache and retry.
        print(
            f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
            "STALE NOT_FOUND CACHE -> RETRY"
        )
        player_cache.pop(season, None)
        _save_yahoo_player_key_cache(cache)

    if isinstance(cached_value, dict) and cached_value.get("player_key"):
        print(
            f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
            f"CACHE HIT -> {cached_value['player_key']}"
        )
        return cached_value

    # Only the current season may use the roster key directly.
    if season == CURRENT_SEASON and current_roster_player:
        roster_item = dict(current_roster_player)
        roster_key = roster_item.get("player_key")
        roster_name = roster_item.get("name")
        if (
            roster_key
            and roster_name
            and normalize_player_name(roster_name)
            == normalize_player_name(player_name)
        ):
            player_cache[season] = roster_item
            _save_yahoo_player_key_cache(cache)
            print(
                f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
                f"CURRENT ROSTER -> {roster_key}"
            )
            return roster_item

    # Historical seasons always resolve name -> player_key within that game.
    search_terms = [str(player_name).strip()]
    last_name = str(player_name).strip().split()[-1]
    if last_name and last_name not in search_terms:
        search_terms.append(last_name)

    for term in search_terms:
        encoded = quote(term, safe="")
        try:
            root = _yahoo_get_xml(
                f"game/{game_key}/players;search={encoded}"
            )
        except Exception as error:
            print(
                f"[YAHOO_PLAYER_KEY] search failed | "
                f"player={player_name} season={season} term={term} | "
                f"error={type(error).__name__}: {error}"
            )
            continue

        returned_names = []
        for returned_node in _descendants(root, "player"):
            returned_item = _extract_player(returned_node)
            if returned_item.get("name"):
                returned_names.append(returned_item["name"])
        print(
            f"[YAHOO_PLAYER_SEARCH] {player_name} | {season} | "
            f"term={term!r} | returned={len(returned_names)} | "
            f"sample={returned_names[:5]}"
        )

        exact = _exact_player_from_root(root, player_name)
        if exact:
            player_cache[season] = exact
            _save_yahoo_player_key_cache(cache)
            print(
                f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
                f"NAME SEARCH -> {exact['player_key']}"
            )
            return exact

    # Do not cache NOT_FOUND. A transient Yahoo filtering/response problem
    # must not permanently turn an actual historical season into zero games.
    print(
        f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
        "NOT FOUND THIS RUN -> SKIP (NOT CACHED)"
    )
    return None


def _season_date_bounds(season):
    start_year = int(str(season).split("-")[0])
    return date(start_year, 10, 1), date(start_year + 1, 6, 30)


def load_schedule_context():
    if not SCHEDULE_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing schedule cache: {SCHEDULE_CACHE_PATH}")
    schedule = pd.read_csv(SCHEDULE_CACHE_PATH, dtype={"GAME_ID": str})
    schedule["DATE"] = pd.to_datetime(schedule["DATE"], errors="coerce")
    schedule["TEAM"] = schedule["TEAM"].astype(str).str.upper().str.strip()
    if schedule.duplicated(["DATE", "TEAM"]).any():
        raise ValueError("Schedule cache DATE + TEAM is not unique")
    return schedule


def _fetch_yahoo_daily(player_key, game_date, timeout=60):
    date_text = pd.Timestamp(game_date).strftime("%Y-%m-%d")
    root = _yahoo_get_xml(
        f"player/{player_key}/stats;type=date;date={date_text}",
        timeout=timeout,
    )
    player_nodes = _descendants(root, "player")
    if not player_nodes:
        return None
    player = player_nodes[0]
    team = _child_text(player, "editorial_team_abbr")
    values = {}
    for stat in _descendants(player, "stat"):
        stat_id = _child_text(stat, "stat_id")
        value = _child_text(stat, "value")
        if stat_id in YAHOO_STAT_IDS:
            values[YAHOO_STAT_IDS[stat_id]] = _safe_number(value)
    if not team or not any(values.values()):
        return None
    return {"Date": pd.Timestamp(game_date), "Team": team.upper(), **values}


def download_yahoo_player_season(
    player_name,
    game_key,
    season,
    timeout=60,
    roster_player=None,
):
    """
    Build one Yahoo season with trade-safe date discovery.

    Historical seasons deliberately use unique NBA calendar dates rather than
    filtering by player metadata team. Yahoo's editorial team can represent a
    later/current team and is not reliable for deciding historical game dates.

    Successful game rows are checkpointed immediately. A separate checked-date
    file records both game and no-game dates, so interrupted runs resume without
    repeating completed requests.
    """
    player = find_yahoo_player(
        player_name=player_name,
        game_key=game_key,
        season=season,
        current_roster_player=roster_player,
    )
    if player is None:
        return None, None

    print(
        f"[YAHOO_PLAYER_KEY] {player_name} | {season} | "
        f"game_key={game_key} | player_key={player['player_key']}"
    )

    safe_key = str(player["player_key"]).replace(".", "_")
    cache_path = YAHOO_DAILY_CACHE_DIR / f"{safe_key}_{season}.csv"
    checked_path = (
        YAHOO_DAILY_CACHE_DIR
        / f"{safe_key}_{season}_checked_dates.txt"
    )

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
        cached = cached.dropna(subset=["Date"])
    else:
        cached = pd.DataFrame()

    checked_dates = set()
    if checked_path.exists():
        for line in checked_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                parsed = pd.to_datetime(line, errors="coerce")
                if not pd.isna(parsed):
                    checked_dates.add(pd.Timestamp(parsed).normalize())

    # Backward compatibility: every previously cached game date is already
    # checked even when it predates the checked-date sidecar file.
    if not cached.empty:
        checked_dates.update(
            cached["Date"].dropna().dt.normalize().tolist()
        )

    start_date, season_end = _season_date_bounds(season)
    end_date = min(season_end, date.today())
    schedule = load_schedule_context()

    date_mask = (
        (schedule["DATE"].dt.date >= start_date)
        & (schedule["DATE"].dt.date <= end_date)
    )

    # TRADE-SAFE: one request per unique NBA calendar date. Do not constrain
    # the historical season to editorial_team_abbr/current team.
    candidate_dates = sorted(
        pd.to_datetime(
            schedule.loc[date_mask, "DATE"], errors="coerce"
        )
        .dropna()
        .dt.normalize()
        .unique()
    )

    pending_dates = [
        pd.Timestamp(value).normalize()
        for value in candidate_dates
        if pd.Timestamp(value).normalize() not in checked_dates
    ]

    print(
        f"[YAHOO_DAILY] {player_name} | {season} | "
        f"trade_safe_dates=yes | games_cached={len(cached)} | "
        f"dates_checked={len(checked_dates)} | "
        f"pending_dates={len(pending_dates)}"
    )

    def save_game_cache(frame):
        if frame is None or frame.empty:
            return
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = (
            frame.dropna(subset=["Date"])
            .drop_duplicates(["Date"], keep="last")
            .sort_values("Date", ascending=False)
            .reset_index(drop=True)
        )
        temp_path = cache_path.with_suffix(".tmp")
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        temp_path.replace(cache_path)

    def save_checked_dates(values):
        temp_path = checked_path.with_suffix(".tmp")
        text = "\n".join(
            sorted(pd.Timestamp(value).strftime("%Y-%m-%d") for value in values)
        )
        if text:
            text += "\n"
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(checked_path)

    combined = cached.copy()
    total = len(pending_dates)

    for index, timestamp in enumerate(pending_dates, start=1):
        print(
            f"[YAHOO_DAILY] {player_name} | {season} | "
            f"{index}/{total} | {timestamp:%Y-%m-%d}"
        )

        row = _fetch_yahoo_daily(
            player["player_key"], timestamp, timeout=timeout
        )

        if row is not None:
            combined = pd.concat(
                [combined, pd.DataFrame([row])],
                ignore_index=True,
            )
            save_game_cache(combined)
            print(
                f"[YAHOO_CHECKPOINT] {player_name} | {season} | "
                f"{timestamp:%Y-%m-%d} | GAME SAVED | "
                f"total_games={len(combined.drop_duplicates(['Date']))}"
            )

        # Only mark the date checked after the request completed normally.
        # A timeout/exception exits before this point and is retried next run.
        checked_dates.add(timestamp)
        save_checked_dates(checked_dates)

    if combined.empty:
        print(
            f"[YAHOO_SEASON_CACHE] {player_name} | {season} | games=0"
        )
        return combined, player

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = (
        combined.dropna(subset=["Date"])
        .drop_duplicates(["Date"], keep="last")
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )
    save_game_cache(combined)

    print(
        f"[YAHOO_SEASON_CACHE] {player_name} | {season} | "
        f"games={len(combined)} | dates_checked={len(checked_dates)}"
    )
    return combined, player

def normalize_yahoo_game_log(yahoo_df, season, player_name):
    if yahoo_df is None or yahoo_df.empty:
        return pd.DataFrame()
    df = yahoo_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Team"] = df["Team"].astype(str).str.upper().str.strip()
    schedule = load_schedule_context()[
        ["DATE", "TEAM", "OPP", "H_A", "IS_HOME"]
    ].copy()
    df = df.merge(
        schedule,
        left_on=["Date", "Team"],
        right_on=["DATE", "TEAM"],
        how="left",
        validate="many_to_one",
    )
    missing = df["OPP"].isna()
    if missing.any():
        print(
            f"[YAHOO_SCHEDULE_WARNING] {player_name} {season}: "
            f"{int(missing.sum())} unmatched DATE + TEAM rows"
        )
    df["Opp"] = df["OPP"]
    df["HomeAway"] = df["H_A"].map({"H": "HOME", "A": "AWAY"})
    df["IsHome"] = df["IS_HOME"]
    df["Season"] = season
    for column in STAT_COLS:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    keep = [
        "Date", "Team", "Opp", "HomeAway", "IsHome",
        "Pts", "Reb", "Ast", "Stl", "Blk", "3PM", "TO",
        "FGM", "FGA", "FTM", "FTA", "Season",
    ]
    return (
        df[keep]
        .dropna(subset=["Date"])
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )


def get_player_df_from_yahoo(
    name,
    league_key,
    seasons=SEASONS,
    timeout=60,
    roster_player=None,
):
    game_keys = get_yahoo_nba_game_keys(
        seasons=seasons,
        current_league_key=league_key,
    )
    season_frames = []
    identified = dict(roster_player) if roster_player is not None else None

    for season in seasons:
        season_game_key = game_keys[season]
        raw, season_player = download_yahoo_player_season(
            player_name=name,
            game_key=season_game_key,
            season=season,
            timeout=timeout,
            roster_player=roster_player,
        )

        if season_player is None:
            continue

        if season == CURRENT_SEASON:
            identified = season_player
        elif identified is None:
            identified = season_player

        normalized = normalize_yahoo_game_log(raw, season, name)
        if not normalized.empty:
            season_frames.append(normalized)

    if not season_frames:
        print(
            f"[YAHOO_PLAYER] {name} | no seasons with game data"
        )
        return pd.DataFrame()

    result = pd.concat(season_frames, ignore_index=True)
    current_team = None
    if roster_player is not None:
        current_team = roster_player.get("team")
    if not current_team and identified:
        current_team = identified.get("team")
    if not current_team and not result.empty:
        current_team = result.iloc[0]["Team"]

    result.attrs["player_id"] = identified.get("player_id") if identified else None
    result.attrs["yahoo_player_key"] = identified.get("player_key") if identified else None
    result.attrs["official_player_name"] = identified.get("name", name) if identified else name
    result.attrs["current_team"] = current_team
    result.attrs["current_team_source"] = "Yahoo current roster"
    return result

def parse_yahoo_league_input(value):
    """Accept public Yahoo URL, league ID, or API league key."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("請輸入 Yahoo League URL、League ID 或 API League Key。")
    api_match = re.fullmatch(r"(\d+)\.l\.(\d+)", text)
    if api_match:
        return {"league_key": text, "league_id": api_match.group(2), "team_number": None}
    url_match = re.search(r"basketball\.fantasysports\.yahoo\.com/nba/(\d+)(?:/(\d+))?", text)
    if url_match:
        return {"league_key": None, "league_id": url_match.group(1), "team_number": url_match.group(2)}
    if re.fullmatch(r"\d+", text):
        return {"league_key": None, "league_id": text, "team_number": None}
    raise ValueError("無法辨識輸入。範例：1486、Yahoo league URL，或 478.l.621。")


def resolve_yahoo_league_key(value):
    parsed = parse_yahoo_league_input(value)
    if parsed["league_key"]:
        return parsed
    root = _yahoo_get_xml("users;use_login=1/games/leagues")
    matches = []
    for league in _descendants(root, "league"):
        league_key = _child_text(league, "league_key")
        league_id = _child_text(league, "league_id")
        if league_key and league_id == parsed["league_id"]:
            matches.append(league_key)
    if not matches:
        raise ValueError(
            f"登入的 Yahoo 帳號找不到 League ID {parsed['league_id']}。"
        )
    if len(matches) > 1:
        raise ValueError(
            f"League ID {parsed['league_id']} 對應到多個 Yahoo game。請輸入完整 API league key。"
        )
    parsed["league_key"] = matches[0]
    return parsed


# ============================================================
# PLAYER NAME NORMALIZATION
# ============================================================
def normalize_player_name(name):
    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def build_roster(
    player_names,
    seasons=SEASONS,
    league_key=None,
    yahoo_players=None,
):
    """Yahoo-only roster builder. Runtime NBA API access is forbidden."""
    if not league_key:
        raise RuntimeError(
            "NBA API player loading is disabled. Load a Yahoo matchup first."
        )
    yahoo_by_name = {
        normalize_player_name(p["name"]): p
        for p in (yahoo_players or [])
        if p.get("name")
    }
    roster = {}
    for name in player_names:
        print("\n" + "=" * 60)
        print(f"Loading Yahoo history: {name}")
        print("=" * 60)
        roster_player = yahoo_by_name.get(normalize_player_name(name))
        if roster_player is None:
            raise ValueError(f"{name}: Yahoo roster metadata / player_key not found")
        df = get_player_df_from_yahoo(
            name=name,
            league_key=league_key,
            seasons=seasons,
            roster_player=roster_player,
        )
        if df.empty:
            print(f"WARNING: {name} Yahoo DataFrame is empty")
        roster[name] = df
    return roster

def load_yahoo_matchup_context(league_input):
    """Load only current matchup, rosters, statuses, and weekly totals."""
    resolved = resolve_yahoo_league_key(league_input)
    league_key = resolved["league_key"]
    context = get_yahoo_league_context(league_key)
    context["public_league_id"] = resolved["league_id"]
    context["public_team_number"] = resolved["team_number"]
    context["historical_loaded"] = False
    print(
        f"[YAHOO_MATCHUP_READY] league={league_key} | "
        f"week={context['current_week']} | "
        f"my_players={len(context['my_roster'])} | "
        f"opp_players={len(context['opponent_roster'])}"
    )
    return context


def build_yahoo_historical_rosters(context, seasons=SEASONS):
    """Build simulation-ready historical DataFrames from a loaded context."""
    if not isinstance(context, dict):
        raise ValueError("Yahoo matchup context is missing or invalid.")

    required = {"league_key", "my_roster", "opponent_roster"}
    missing = required - set(context)
    if missing:
        raise ValueError(
            "Yahoo matchup context is missing: "
            + ", ".join(sorted(missing))
        )

    league_key = context["league_key"]
    my_players = list(context["my_roster"])
    opponent_players = list(context["opponent_roster"])

    injured_slots = {"IL", "IL+", "IR", "IR+"}
    my_simulation_players = [
        player for player in my_players
        if str(player.get("selected_position") or "").strip().upper()
        not in injured_slots
    ]
    opponent_simulation_players = [
        player for player in opponent_players
        if str(player.get("selected_position") or "").strip().upper()
        not in injured_slots
    ]

    my_names = [player["name"] for player in my_simulation_players]
    opponent_names = [player["name"] for player in opponent_simulation_players]

    print(
        f"[YAHOO_HISTORY_START] league={league_key} | "
        f"seasons={len(seasons)} | my_players={len(my_names)} | "
        f"opp_players={len(opponent_names)}"
    )

    my_roster_engine = build_roster(
        my_names,
        seasons=seasons,
        league_key=league_key,
        yahoo_players=my_simulation_players,
    )
    opponent_roster_engine = build_roster(
        opponent_names,
        seasons=seasons,
        league_key=league_key,
        yahoo_players=opponent_players,
    )

    print(
        f"[YAHOO_HISTORY_READY] league={league_key} | "
        f"my_players={len(my_roster_engine)} | "
        f"opp_players={len(opponent_roster_engine)}"
    )
    return {
        "my_roster_engine": my_roster_engine,
        "opponent_roster_engine": opponent_roster_engine,
    }


def build_yahoo_matchup_rosters(league_input, seasons=SEASONS):
    """Backward-compatible combined loader."""
    context = load_yahoo_matchup_context(league_input)
    historical = build_yahoo_historical_rosters(context, seasons=seasons)
    combined = dict(context)
    combined.update(historical)
    combined["historical_loaded"] = True
    return combined

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
    opp_boost=OPP_BOOST,
):
    """Apply recency x future home/away x future opponent sampling weights."""
    df = df.copy()

    required = {"Date", "HomeAway", "Opp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Weighting data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if df.empty:
        raise ValueError("Cannot calculate sampling weights from an empty DataFrame.")

    df = (
        df.sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )

    future_home_away = str(game["home_away"]).strip().lower()
    future_opp = str(game["opponent"]).strip().upper()
    if future_home_away not in {"home", "away"}:
        raise ValueError(
            f"Unexpected future home_away={game['home_away']!r}; expected home/away."
        )

    hist_home_away = df["HomeAway"].astype(str).str.strip().str.upper()
    hist_opp = df["Opp"].astype(str).str.strip().str.upper()

    # Keep each component separate so the final sampling probability is auditable.
    df["RecencyWeight"] = decay ** np.arange(len(df), dtype=float)
    df["HomeAwayMultiplier"] = 1.0
    if future_home_away == "home":
        df.loc[hist_home_away.eq("HOME"), "HomeAwayMultiplier"] = home_boost
        expected_ha_mask = hist_home_away.eq("HOME")
    else:
        df.loc[hist_home_away.eq("AWAY"), "HomeAwayMultiplier"] = away_boost
        expected_ha_mask = hist_home_away.eq("AWAY")

    df["OpponentMultiplier"] = 1.0
    opp_mask = hist_opp.eq(future_opp)
    df.loc[opp_mask, "OpponentMultiplier"] = opp_boost

    df["RawWeight"] = (
        df["RecencyWeight"]
        * df["HomeAwayMultiplier"]
        * df["OpponentMultiplier"]
    )

    raw_sum = float(df["RawWeight"].sum())
    if not np.isfinite(raw_sum) or raw_sum <= 0:
        raise ValueError(f"Invalid RawWeight sum: {raw_sum}")

    df["Weight"] = df["RawWeight"] / raw_sum

    # --------------------------------------------------------
    # AUTOMATIC WEIGHT CHECKS
    # --------------------------------------------------------
    checks = {
        "normalized_sum": bool(np.isclose(df["Weight"].sum(), 1.0, atol=1e-12)),
        "all_weights_finite": bool(np.isfinite(df["Weight"]).all()),
        "all_weights_positive": bool((df["Weight"] > 0).all()),
        "recency_formula": bool(np.allclose(
            df["RecencyWeight"].to_numpy(dtype=float),
            decay ** np.arange(len(df), dtype=float),
        )),
        "home_away_multiplier": bool(np.allclose(
            df["HomeAwayMultiplier"].to_numpy(dtype=float),
            np.where(
                expected_ha_mask.to_numpy(),
                home_boost if future_home_away == "home" else away_boost,
                1.0,
            ),
        )),
        "opponent_multiplier": bool(np.allclose(
            df["OpponentMultiplier"].to_numpy(dtype=float),
            np.where(opp_mask.to_numpy(), opp_boost, 1.0),
        )),
        "raw_weight_product": bool(np.allclose(
            df["RawWeight"].to_numpy(dtype=float),
            (
                df["RecencyWeight"]
                * df["HomeAwayMultiplier"]
                * df["OpponentMultiplier"]
            ).to_numpy(dtype=float),
        )),
    }

    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise AssertionError(
            "Weight audit failed: " + ", ".join(failed_checks)
        )

    # --------------------------------------------------------
    # WEIGHT AUDIT TABLE
    # --------------------------------------------------------
    audit_columns = [
        "Date", "Team", "Opp", "HomeAway", "Season",
        "RecencyWeight", "HomeAwayMultiplier", "OpponentMultiplier",
        "RawWeight", "Weight",
    ]
    available_audit_columns = [
        column for column in audit_columns if column in df.columns
    ]
    audit_table = df[available_audit_columns].copy()
    audit_table = audit_table.rename(columns={"Weight": "NormalizedWeight"})

    print("\n" + "=" * 92)
    print("WEIGHT AUDIT")
    print("=" * 92)
    print(
        f"Future Game: date={game['date']} | opp={future_opp} | "
        f"home_away={future_home_away.upper()}"
    )
    print(
        f"Settings: DECAY={decay} | HOME_BOOST={home_boost} | "
        f"AWAY_BOOST={away_boost} | OPP_BOOST={opp_boost}"
    )
    print(
        f"Historical rows={len(df)} | matching H/A={int(expected_ha_mask.sum())} | "
        f"matching opponent={int(opp_mask.sum())}"
    )
    print("Automatic checks:")
    for check_name, passed in checks.items():
        print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")
    print(f"  normalized_weight_sum: {df['Weight'].sum():.12f}")

    # Print the rows most useful for manual verification: highest final sampling
    # probability, while preserving all component multipliers.
    display_audit = (
        audit_table.sort_values("NormalizedWeight", ascending=False)
        .head(20)
        .copy()
    )
    numeric_cols = [
        "RecencyWeight", "HomeAwayMultiplier", "OpponentMultiplier",
        "RawWeight", "NormalizedWeight",
    ]
    for column in numeric_cols:
        if column in display_audit.columns:
            display_audit[column] = display_audit[column].astype(float).round(8)
    print("\nTop 20 historical rows by sampling probability:")
    print(display_audit.to_string(index=False))
    print("=" * 92)

    # Keep the audit table attached without changing the existing caller API.
    df.attrs["weight_audit"] = audit_table
    df.attrs["weight_checks"] = checks
    df.attrs["future_game"] = {
        "date": game["date"],
        "opponent": future_opp,
        "home_away": future_home_away,
    }
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

def identify_yahoo_add_player(player_name, league_key):
    """
    Resolve Add Player metadata from the selected Yahoo LEAGUE player pool.

    Add/Drop candidates may be free agents or owned by another team, so they
    are not present in either loaded matchup roster. Searching the current
    game's generic players collection proved unreliable because Yahoo could
    return an unfiltered first page. The league-scoped Players Collection is
    the correct eligible-player scope for this Add Player workflow.
    """
    cleaned_name = " ".join(str(player_name or "").strip().split())
    if not cleaned_name:
        raise ValueError("Add Player 名稱不可為空白。")
    if not league_key:
        raise RuntimeError("缺少 Yahoo league_key，無法識別 Add Player。")

    current_game_key = str(league_key).split(".")[0]
    target = normalize_player_name(cleaned_name)
    search_terms = [cleaned_name]
    last_name = cleaned_name.split()[-1]
    if last_name not in search_terms:
        search_terms.append(last_name)

    print(
        f"[YAHOO_ADD_IDENTIFY] player={cleaned_name} | "
        f"league_key={league_key} | game_key={current_game_key}"
    )

    # Search league scope first. It represents players eligible for this
    # fantasy league and includes players outside the current matchup rosters.
    search_scopes = [
        ("LEAGUE", f"league/{league_key}/players"),
        ("GAME_FALLBACK", f"game/{current_game_key}/players"),
    ]

    for scope_name, base_path in search_scopes:
        for term in search_terms:
            encoded = quote(term, safe="")
            root = _yahoo_get_xml(f"{base_path};search={encoded}")

            candidates = []
            for player_node in _descendants(root, "player"):
                item = _extract_player(player_node)
                if item.get("player_key") and item.get("name"):
                    candidates.append(item)

            print(
                f"[YAHOO_ADD_SEARCH] scope={scope_name} | "
                f"term={term!r} | returned={len(candidates)} | "
                f"sample={[item['name'] for item in candidates[:8]]}"
            )

            exact = [
                item for item in candidates
                if normalize_player_name(item["name"]) == target
            ]
            if len(exact) == 1:
                metadata = exact[0]

                # Persist current-season metadata so the normal historical
                # loader takes exactly the same path as a roster player.
                cache = _load_yahoo_player_key_cache()
                identity = _player_cache_identity(metadata["name"])
                cache.setdefault(identity, {})[CURRENT_SEASON] = metadata
                _save_yahoo_player_key_cache(cache)

                print(
                    f"[YAHOO_ADD_IDENTIFIED] scope={scope_name} | "
                    f"name={metadata.get('name')} | "
                    f"player_key={metadata.get('player_key')} | "
                    f"player_id={metadata.get('player_id')} | "
                    f"team={metadata.get('team')} | "
                    f"status={metadata.get('status')}"
                )
                return metadata

            if len(exact) > 1:
                raise ValueError(
                    f"Yahoo 對 {cleaned_name} 回傳多個精確結果："
                    f"{[item['player_key'] for item in exact]}"
                )

    raise ValueError(
        f"Yahoo League {league_key} 找不到 {cleaned_name}。"
        "已檢查 league player pool 與 current game player pool；"
        "請查看 PowerShell 的 [YAHOO_ADD_SEARCH] 回傳名單。"
    )

def create_add_drop_roster(
    baseline_roster,
    drop_player,
    add_player,
    league_key,
    seasons=SEASONS,
):
    """Yahoo-only Add/Drop with current-game identification before history."""
    if drop_player not in baseline_roster:
        raise ValueError(f"{drop_player} 不在目前 roster 裡")
    if add_player in baseline_roster:
        raise ValueError(f"{add_player} 已經在目前 roster 裡")
    if not league_key:
        raise RuntimeError("Add Player 需要已連線的 Yahoo league context。")

    add_metadata = identify_yahoo_add_player(
        player_name=add_player,
        league_key=league_key,
    )
    official_name = add_metadata.get("name") or add_player
    add_df = get_player_df_from_yahoo(
        name=official_name,
        league_key=league_key,
        seasons=seasons,
        roster_player=add_metadata,
    )
    if add_df.empty:
        raise ValueError(
            f"Yahoo 已識別 {official_name} ({add_metadata.get('player_key')})，"
            "但沒有取得任何歷史比賽資料。"
        )

    result = {}
    for name, df in baseline_roster.items():
        result[official_name if name == drop_player else name] = (
            add_df if name == drop_player else df
        )
    print(
        f"[YAHOO_ADD_READY] player={official_name} | rows={len(add_df)} | "
        f"current_team={add_df.attrs.get('current_team')}"
    )
    return result

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
    
    comparison_df["Category"] = pd.Categorical(
        comparison_df["Category"],
        categories=category_order,
        ordered=True,
    )
    
    comparison_df = (
        comparison_df
        .sort_values("Category")
        .reset_index(drop=True)
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
        "3PM",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
        "TO"
    ]

    contribution_categories = [
        "FG%",
        "FT%",
        "3PM",
        "Pts",
        "Reb",
        "Ast",
        "Stl",
        "Blk",
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
    random_seed=12345,
    yahoo_league_key=None
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
                seasons=seasons,
                league_key=yahoo_league_key,
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