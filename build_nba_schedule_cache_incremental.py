"""
NBA 2018-19 to 2026-27 team game-context cache

Purpose
-------
Download team-level NBA game logs once, keep only the fields needed to enrich
Yahoo player daily stats, and save a reusable cache with:

    DATE, SEASON, GAME_ID, TEAM, OPP, H_A, IS_HOME, MATCHUP

The final Yahoo player log can be merged on DATE + TEAM.

Install once:
    pip install nba_api pandas

Run:
    python build_nba_schedule_cache.py

Notes
-----
* LeagueGameFinder returns played games, not future unplayed schedule fixtures.
* By default this script downloads Regular Season games only.
* One row represents one team's side of one game. Therefore every completed
  NBA game normally appears twice: one HOME row and one AWAY row.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


# ============================================================================
# SETTINGS
# ============================================================================

SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

# Keep Regular Season only for the Yahoo fantasy historical pool.
# Add "Playoffs" here later if you explicitly want postseason games too.
SEASON_TYPES = ["Regular Season"]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "nba_schedule_2018_2027.csv"
CHECKPOINT_FILE = BASE_DIR / "nba_schedule_checkpoint.csv"
FAILED_FILE = BASE_DIR / "nba_schedule_failed_requests.csv"
REFRESH_STATE_FILE = BASE_DIR / "nba_schedule_refresh_state.json"
REFRESH_TTL_HOURS = 24

LEAGUE_ID = "00"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 2.5

# True: call the API again even if a season/type is already in the checkpoint.
# False: reuse completed checkpoint data and download only missing season/types.
FORCE_REFRESH = False

# Incremental mode: completed historical seasons stay cached; this season is
# refreshed every run so newly completed games are added automatically.
CURRENT_SEASON = "2026-27"


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(series: pd.Series) -> pd.Series:
    """Return dates in YYYY-MM-DD format for reliable Yahoo merging."""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def parse_matchup(matchup: object) -> tuple[str | None, str | None]:
    """
    Examples:
        DEN vs. LAL -> (LAL, H)
        DEN @ LAL   -> (LAL, A)
    """
    text = str(matchup).strip()

    if " vs. " in text:
        return text.split(" vs. ", 1)[1].strip(), "H"

    if " vs " in text:
        return text.split(" vs ", 1)[1].strip(), "H"

    if " @ " in text:
        return text.split(" @ ", 1)[1].strip(), "A"

    return None, None


def transform_games(raw: pd.DataFrame, season: str, season_type: str) -> pd.DataFrame:
    """Convert LeagueGameFinder output to the small reusable context table."""
    required = {
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MATCHUP",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"NBA response is missing columns: {sorted(missing)}")

    df = raw.copy()
    parsed = df["MATCHUP"].apply(parse_matchup)
    df["OPP"] = parsed.str[0]
    df["H_A"] = parsed.str[1]

    out = pd.DataFrame(
        {
            "DATE": normalize_date(df["GAME_DATE"]),
            "SEASON": season,
            "SEASON_TYPE": season_type,
            "GAME_ID": df["GAME_ID"].astype(str),
            "TEAM_ID": pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64"),
            "TEAM": df["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip(),
            "OPP": df["OPP"].astype("string").str.upper().str.strip(),
            "H_A": df["H_A"],
            "IS_HOME": df["H_A"].eq("H"),
            "MATCHUP": df["MATCHUP"].astype(str),
        }
    )

    out = out.dropna(subset=["DATE", "TEAM", "OPP", "H_A"])
    out = out.drop_duplicates(subset=["DATE", "GAME_ID", "TEAM"])
    return out


def fetch_one(season: str, season_type: str) -> pd.DataFrame:
    """Fetch one season/type with bounded retries and increasing delays."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\nFetching {season} | {season_type} | attempt {attempt}/{MAX_RETRIES}")

            response = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                league_id_nullable=LEAGUE_ID,
                season_nullable=season,
                season_type_nullable=season_type,
                timeout=REQUEST_TIMEOUT,
            )

            frames = response.get_data_frames()
            if not frames:
                raise RuntimeError("NBA API returned no data frames")

            raw = frames[0]
            print(f"Received {len(raw):,} team-game rows")
            return transform_games(raw, season, season_type)

        except Exception as exc:
            last_error = exc
            print(f"Request failed: {type(exc).__name__}: {exc}")

            if attempt < MAX_RETRIES:
                delay = (5 * attempt) + random.uniform(0.5, 2.0)
                print(f"Retrying after {delay:.1f} seconds...")
                time.sleep(delay)

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts: {season} | {season_type}"
    ) from last_error


def load_checkpoint() -> pd.DataFrame:
    source = CHECKPOINT_FILE if CHECKPOINT_FILE.exists() else OUTPUT_FILE
    if not source.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            source,
            dtype={"GAME_ID": str, "TEAM": str, "OPP": str},
        )
        print(f"Loaded schedule seed: {source.name} | {len(df):,} rows")
        return df
    except Exception as exc:
        print(f"Schedule seed could not be read and will be ignored: {exc}")
        return pd.DataFrame()


def save_checkpoint(df: pd.DataFrame) -> None:
    df.to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")


def checkpoint_has(checkpoint: pd.DataFrame, season: str, season_type: str) -> bool:
    if checkpoint.empty:
        return False
    if not {"SEASON", "SEASON_TYPE"}.issubset(checkpoint.columns):
        return False

    mask = (
        checkpoint["SEASON"].astype(str).eq(season)
        & checkpoint["SEASON_TYPE"].astype(str).eq(season_type)
    )
    return bool(mask.any())


def validate_cache(df: pd.DataFrame) -> None:
    """Fail loudly if a merge key or parsed matchup is invalid."""
    if df.empty:
        raise RuntimeError("Final cache is empty")

    required = ["DATE", "SEASON", "GAME_ID", "TEAM", "OPP", "H_A"]
    null_counts = df[required].isna().sum()
    if int(null_counts.sum()) > 0:
        raise RuntimeError(f"Null values found in required columns:\n{null_counts}")

    invalid_ha = ~df["H_A"].isin(["H", "A"])
    if invalid_ha.any():
        raise RuntimeError("Invalid H_A values found")

    duplicate_keys = df.duplicated(subset=["DATE", "TEAM"], keep=False)
    if duplicate_keys.any():
        sample = df.loc[duplicate_keys, ["DATE", "TEAM", "GAME_ID", "MATCHUP"]].head(20)
        raise RuntimeError(
            "DATE + TEAM is not unique. Sample duplicate merge keys:\n"
            f"{sample.to_string(index=False)}"
        )

    # Completed team-level game logs normally contain two rows per GAME_ID.
    counts = df.groupby(["SEASON", "SEASON_TYPE", "GAME_ID"]).size()
    unusual = counts[counts != 2]
    if not unusual.empty:
        print(
            f"WARNING: {len(unusual):,} games do not have exactly two team rows. "
            "The file is still being saved; inspect if Yahoo merges are missing."
        )


def _load_refresh_state() -> dict:
    if not REFRESH_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(REFRESH_STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Schedule refresh state could not be read: {exc}")
        return {}

def _cache_is_fresh(max_age_hours: int = REFRESH_TTL_HOURS) -> tuple[bool, str | None, float | None]:
    state = _load_refresh_state()
    last_text = state.get("last_successful_refresh")
    if not last_text or not OUTPUT_FILE.exists():
        return False, last_text, None
    try:
        last_dt = datetime.fromisoformat(str(last_text).replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        return age_hours < float(max_age_hours), last_text, age_hours
    except Exception as exc:
        print(f"Schedule refresh timestamp is invalid: {exc}")
        return False, last_text, None

def _write_refresh_state(latest_completed_game_date: str | None) -> str:
    refreshed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "last_successful_refresh": refreshed_at,
        "current_season": CURRENT_SEASON,
        "latest_completed_game_date": latest_completed_game_date,
    }
    temp_path = REFRESH_STATE_FILE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(REFRESH_STATE_FILE)
    return refreshed_at

def update_schedule_cache() -> dict:
    is_fresh, last_refresh, age_hours = _cache_is_fresh()
    if is_fresh:
        cached = pd.read_csv(OUTPUT_FILE, dtype={"GAME_ID": str, "TEAM": str, "OPP": str})
        current_rows = cached.loc[cached["SEASON"].astype(str).eq(CURRENT_SEASON)]
        latest_date = None if current_rows.empty else str(current_rows["DATE"].max())
        print(f"[SCHEDULE_CACHE] FRESH | age_hours={age_hours:.2f} | ttl_hours={REFRESH_TTL_HOURS} | SKIP NBA API")
        return {
            "output_file": str(OUTPUT_FILE), "rows": int(len(cached)),
            "current_season": CURRENT_SEASON, "current_season_rows": int(len(current_rows)),
            "latest_completed_game_date": latest_date, "failures": [],
            "refreshed": False, "last_successful_refresh": last_refresh,
            "cache_age_hours": age_hours, "refresh_ttl_hours": REFRESH_TTL_HOURS,
        }

    print(f"[SCHEDULE_CACHE] STALE OR UNKNOWN | ttl_hours={REFRESH_TTL_HOURS} | REFRESH NBA API")
    checkpoint = load_checkpoint()
    combined = checkpoint.copy()
    failures: list[dict[str, str]] = []

    for season in SEASONS:
        for season_type in SEASON_TYPES:
            already_cached = checkpoint_has(combined, season, season_type)

            # Historical seasons are immutable for our purpose once cached.
            # The current season is deliberately refreshed every run because
            # LeagueGameFinder gains rows as new games are completed.
            should_refresh = FORCE_REFRESH or (season == CURRENT_SEASON)

            if already_cached and not should_refresh:
                print(f"Skipping historical cached {season} | {season_type}")
                continue

            if season == CURRENT_SEASON:
                print(f"Refreshing CURRENT season: {season} | {season_type}")
            elif not already_cached:
                print(f"Historical season missing from cache; downloading: {season} | {season_type}")

            try:
                fresh = fetch_one(season, season_type)

                # Replace this season/type atomically in memory. This is safer
                # than appending only today's rows because corrections to an
                # already played NBA game will also be reflected.
                if not combined.empty and {"SEASON", "SEASON_TYPE"}.issubset(combined.columns):
                    keep = ~(
                        combined["SEASON"].astype(str).eq(season)
                        & combined["SEASON_TYPE"].astype(str).eq(season_type)
                    )
                    combined = combined.loc[keep].copy()

                combined = pd.concat([combined, fresh], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["DATE", "GAME_ID", "TEAM"],
                    keep="last",
                )
                save_checkpoint(combined)
                print(f"Checkpoint saved: {len(combined):,} total rows")
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            except Exception as exc:
                # Important: if refreshing the current season fails, the old
                # cached current-season rows remain in `combined` because the
                # old rows are removed only AFTER a successful fetch.
                print(f"FAILED: {season} | {season_type}: {exc}")
                failures.append(
                    {
                        "SEASON": season,
                        "SEASON_TYPE": season_type,
                        "ERROR": f"{type(exc).__name__}: {exc}",
                    }
                )

    if combined.empty:
        raise RuntimeError("No season was downloaded successfully")

    combined["DATE"] = normalize_date(combined["DATE"])
    combined = combined.sort_values(
        ["DATE", "GAME_ID", "TEAM"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    validate_cache(combined)
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    if failures:
        pd.DataFrame(failures).to_csv(FAILED_FILE, index=False, encoding="utf-8-sig")
    elif FAILED_FILE.exists():
        FAILED_FILE.unlink()

    current_rows = combined.loc[combined["SEASON"].astype(str).eq(CURRENT_SEASON)]

    print("\n" + "=" * 78)
    print("NBA SCHEDULE/CONTEXT CACHE INCREMENTAL UPDATE COMPLETE")
    print("=" * 78)
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Rows: {len(combined):,}")
    print(f"Seasons represented: {combined['SEASON'].nunique()}")
    print(f"Full date range: {combined['DATE'].min()} to {combined['DATE'].max()}")
    print(f"Current season: {CURRENT_SEASON}")
    print(f"Current-season rows: {len(current_rows):,}")
    if not current_rows.empty:
        print(f"Current-season latest completed game date: {current_rows['DATE'].max()}")
    print(f"Failed requests: {len(failures)}")
    print("Merge Yahoo player stats using: DATE + TEAM")

    print("\nPreview:")
    print(
        combined[
            ["DATE", "SEASON", "GAME_ID", "TEAM", "OPP", "H_A", "IS_HOME"]
        ].tail(12).to_string(index=False)
    )

    latest_completed_game_date = None if current_rows.empty else str(current_rows["DATE"].max())
    successful_current_refresh = not any(item.get("SEASON") == CURRENT_SEASON for item in failures)
    refreshed_at = _write_refresh_state(latest_completed_game_date) if successful_current_refresh else last_refresh
    return {
        "output_file": str(OUTPUT_FILE), "rows": int(len(combined)),
        "current_season": CURRENT_SEASON, "current_season_rows": int(len(current_rows)),
        "latest_completed_game_date": latest_completed_game_date, "failures": failures,
        "refreshed": successful_current_refresh, "last_successful_refresh": refreshed_at,
        "cache_age_hours": 0.0 if successful_current_refresh else age_hours,
        "refresh_ttl_hours": REFRESH_TTL_HOURS,
    }


def main() -> None:
    update_schedule_cache()


if __name__ == "__main__":
    main()
