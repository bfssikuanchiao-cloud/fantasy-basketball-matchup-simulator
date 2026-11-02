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

import random
import time
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

OUTPUT_FILE = Path("nba_schedule_2018_2027.csv")
CHECKPOINT_FILE = Path("nba_schedule_checkpoint.csv")
FAILED_FILE = Path("nba_schedule_failed_requests.csv")

LEAGUE_ID = "00"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 2.5

# True: call the API again even if a season/type is already in the checkpoint.
# False: reuse completed checkpoint data and download only missing season/types.
FORCE_REFRESH = False


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
    if not CHECKPOINT_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            CHECKPOINT_FILE,
            dtype={"GAME_ID": str, "TEAM": str, "OPP": str},
        )
        print(f"Loaded checkpoint: {len(df):,} rows")
        return df
    except Exception as exc:
        print(f"Checkpoint could not be read and will be ignored: {exc}")
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


def main() -> None:
    checkpoint = load_checkpoint()
    combined = checkpoint.copy()
    failures: list[dict[str, str]] = []

    for season in SEASONS:
        for season_type in SEASON_TYPES:
            already_cached = checkpoint_has(combined, season, season_type)

            if already_cached and not FORCE_REFRESH:
                print(f"Skipping cached {season} | {season_type}")
                continue

            try:
                fresh = fetch_one(season, season_type)

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

    print("\n" + "=" * 78)
    print("NBA SCHEDULE/CONTEXT CACHE COMPLETE")
    print("=" * 78)
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Rows: {len(combined):,}")
    print(f"Seasons represented: {combined['SEASON'].nunique()}")
    print(f"Date range: {combined['DATE'].min()} to {combined['DATE'].max()}")
    print(f"Failed requests: {len(failures)}")
    print("Merge Yahoo player stats using: DATE + TEAM")

    print("\nPreview:")
    print(
        combined[
            ["DATE", "SEASON", "GAME_ID", "TEAM", "OPP", "H_A", "IS_HOME"]
        ].tail(12).to_string(index=False)
    )


if __name__ == "__main__":
    main()
