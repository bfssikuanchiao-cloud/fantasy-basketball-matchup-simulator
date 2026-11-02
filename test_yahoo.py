

from yahoo_oauth import OAuth2
import xml.etree.ElementTree as ET
from datetime import date, timedelta
import pandas as pd


# =========================================================
# YAHOO OAUTH
# =========================================================


oauth = OAuth2(
    "dj0yJmk9UkVCM0xualBaUmphJmQ9WVdrOWJqRm9iVkZKU3pnbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PWQ1",
    "e09763fe64c41653d233f0ac915399a863ff4ac4"
)

BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

NS = {
    "f": "http://fantasysports.yahooapis.com/fantasy/v2/base.rng"
}


# =========================================================
# PLAYER
# =========================================================

PLAYER_KEY = "478.p.5352"
PLAYER_NAME = "Nikola Jokic"


# =========================================================
# CONFIRMED YAHOO DAILY STAT IDS
# =========================================================

STAT_FGA = "3"
STAT_FGM = "4"

STAT_FTA = "6"
STAT_FTM = "7"

STAT_3PM = "10"
STAT_PTS = "12"
STAT_REB = "15"
STAT_AST = "16"
STAT_STL = "17"
STAT_BLK = "18"
STAT_TO = "19"


# =========================================================
# ENGINE OUTPUT COLUMNS
# =========================================================

ENGINE_COLUMNS = [
    "DATE",
    "TEAM",
    "FGM",
    "FGA",
    "FTM",
    "FTA",
    "3PM",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TO",
]


# =========================================================
# XML HELPER
# =========================================================

def get_text(element, path):

    if element is None:
        return None

    value = element.findtext(
        path,
        namespaces=NS
    )

    if value is None:
        return None

    return value.strip()


# =========================================================
# NUMBER HELPERS
# =========================================================

def safe_int(value):

    if value is None:
        return 0

    value = str(value).strip()

    if value in {"", "-", "--"}:
        return 0

    try:
        return int(float(value))

    except (ValueError, TypeError):
        return 0


def safe_float(value):

    if value is None:
        return None

    value = str(value).strip()

    if value in {"", "-", "--"}:
        return None

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


# =========================================================
# DAILY STAT FETCH
# =========================================================

def get_daily_raw_stats(player_key, game_date):

    date_string = game_date.strftime("%Y-%m-%d")

    url = (
        f"{BASE}/player/{player_key}/"
        f"stats;type=date;date={date_string}"
    )

    r = oauth.session.get(url)

    if r.status_code != 200:

        print(
            f"{date_string} | "
            f"HTTP {r.status_code} | API ERROR"
        )

        return None

    if date_string == "2026-10-22":
        with open(
            "jokic_2026_10_22_full.xml",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(r.text)
    
        print(
            "\nSaved full Yahoo response:"
            " jokic_2026_10_22_full.xml"
        )
    
    root = ET.fromstring(r.text)

    player = root.find(
        ".//f:player",
        NS
    )

    if player is None:

        print(
            f"{date_string} | "
            "HTTP 200 | NO PLAYER"
        )

        return None


    stats = player.findall(
        ".//f:player_stats/f:stats/f:stat",
        NS
    )

    if not stats:

        print(
            f"{date_string} | "
            "HTTP 200 | NO STATS"
        )

        return None


    stat_values = {}

    for stat in stats:

        stat_id = get_text(
            stat,
            "f:stat_id"
        )

        value = get_text(
            stat,
            "f:value"
        )

        if stat_id is not None:
            stat_values[stat_id] = value


    # -----------------------------------------------------
    # IMPORTANT:
    #
    # We already confirmed that PTS/REB/etc. appear on
    # actual game dates.
    #
    # Therefore use confirmed counting stats to determine
    # whether this is a real played game.
    # -----------------------------------------------------

    pts = safe_int(
        stat_values.get(STAT_PTS)
    )

    reb = safe_int(
        stat_values.get(STAT_REB)
    )

    ast = safe_int(
        stat_values.get(STAT_AST)
    )


    # A zero-point game is theoretically possible, so do
    # not require PTS > 0 alone.
    has_game_data = any(
        stat_values.get(stat_id) not in {
            None,
            "",
            "-",
            "--"
        }
        for stat_id in [
            STAT_PTS,
            STAT_REB,
            STAT_AST,
            STAT_3PM,
            STAT_STL,
            STAT_BLK,
            STAT_TO,
        ]
    )


    if not has_game_data:

        print(
            f"{date_string} | "
            "HTTP 200 | NO GAME"
        )

        return None


    team = get_text(
        player,
        "f:editorial_team_abbr"
    )

    if not team:
        print(
            f"{date_string} | "
            "HTTP 200 | GAME FOUND | TEAM MISSING"
        )
        return None

    print(
        f"{date_string} | "
        f"HTTP 200 | GAME FOUND | TEAM {team}"
    )

    return {
        "TEAM": team.upper(),
        "STATS": stat_values,
    }


# =========================================================
# MAIN
# =========================================================

print("\n")
print("=" * 80)
print(" YAHOO DAILY STATS -> DATAFRAME")
print("=" * 80)

print("\nToken valid:", oauth.token_is_valid())
print("Player     :", PLAYER_NAME)
print("Player Key :", PLAYER_KEY)


END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=14)


print("\nDate range:")
print("Start:", START_DATE)
print("End  :", END_DATE)


# =========================================================
# FETCH DAILY DATA
# =========================================================

daily_games = []

current_date = START_DATE


print("\n")
print("=" * 80)
print(" FETCHING DAILY STATS")
print("=" * 80)
print()


while current_date <= END_DATE:

    stats = get_daily_raw_stats(
        PLAYER_KEY,
        current_date
    )

    if stats is not None:

        daily_games.append({
            "DATE": current_date.strftime(
                "%Y-%m-%d"
            ),
            "TEAM": stats["TEAM"],
            "RAW": stats["STATS"],
        })

    current_date += timedelta(days=1)



# =========================================================
# BUILD NORMALIZED ENGINE DATAFRAME
# =========================================================

rows = []


for game in daily_games:

    raw = game["RAW"]

    row = {
        "DATE": game["DATE"],
        "TEAM": game["TEAM"],

        "FGM": safe_int(
            raw.get(STAT_FGM)
        ),
    
        "FGA": safe_int(
            raw.get(STAT_FGA)
        ),
    
        "FTM": safe_int(
            raw.get(STAT_FTM)
        ),
    
        "FTA": safe_int(
            raw.get(STAT_FTA)
        ),
    
        "3PM": safe_int(
            raw.get(STAT_3PM)
        ),
    
        "PTS": safe_int(
            raw.get(STAT_PTS)
        ),
    
        "REB": safe_int(
            raw.get(STAT_REB)
        ),
    
        "AST": safe_int(
            raw.get(STAT_AST)
        ),
    
        "STL": safe_int(
            raw.get(STAT_STL)
        ),
    
        "BLK": safe_int(
            raw.get(STAT_BLK)
        ),
    
        "TO": safe_int(
            raw.get(STAT_TO)
        ),
    }

    rows.append(row)


game_log = pd.DataFrame(
    rows,
    columns=ENGINE_COLUMNS
)


if not game_log.empty:

    game_log["DATE"] = pd.to_datetime(
        game_log["DATE"]
    )

    game_log = (
        game_log
        .sort_values("DATE")
        .reset_index(drop=True)
    )

game_log["FG% CHECK"] = (
    game_log["FGM"]
    / game_log["FGA"].replace(0, pd.NA)
)

game_log["FT% CHECK"] = (
    game_log["FTM"]
    / game_log["FTA"].replace(0, pd.NA)
)

# =========================================================
# LOAD NBA SCHEDULE CACHE + MERGE ON DATE + TEAM
# =========================================================

SCHEDULE_FILE = "nba_schedule_2018_2027.csv"

print("\n")
print("=" * 80)
print(" LOADING NBA SCHEDULE CACHE")
print("=" * 80)

schedule = pd.read_csv(
    SCHEDULE_FILE,
    dtype={"GAME_ID": str}
)

required_schedule_columns = {
    "DATE", "TEAM", "OPP", "H_A", "IS_HOME"
}
missing = required_schedule_columns - set(schedule.columns)
if missing:
    raise ValueError(
        "Schedule cache is missing columns: "
        + ", ".join(sorted(missing))
    )

schedule["DATE"] = pd.to_datetime(
    schedule["DATE"], errors="coerce"
)
schedule["TEAM"] = (
    schedule["TEAM"].astype(str).str.strip().str.upper()
)

# A team should play at most once on a given date.
dupes = schedule.duplicated(
    subset=["DATE", "TEAM"], keep=False
)
if dupes.any():
    print(schedule.loc[
        dupes, ["DATE", "TEAM", "OPP", "H_A"]
    ].sort_values(["DATE", "TEAM"]).to_string(index=False))
    raise ValueError("Schedule DATE + TEAM is not unique.")

schedule_lookup = schedule[
    ["DATE", "TEAM", "OPP", "H_A", "IS_HOME"]
].copy()

merged_game_log = game_log.merge(
    schedule_lookup,
    on=["DATE", "TEAM"],
    how="left",
    validate="many_to_one",
    indicator=True,
)

merged_game_log["SCHEDULE_MATCH"] = merged_game_log["_merge"].eq("both")
merged_game_log = merged_game_log.drop(columns=["_merge"])

final_columns = [
    "DATE", "TEAM", "OPP", "H_A", "IS_HOME",
    "FGA", "FGM", "FTA", "FTM", "3PM",
    "PTS", "REB", "AST", "STL", "BLK", "TO",
    "FG% CHECK", "FT% CHECK", "SCHEDULE_MATCH",
]
merged_game_log = merged_game_log[final_columns]

print("\n")
print("=" * 80)
print(" JOKIC YAHOO GAME LOG + NBA SCHEDULE MERGE")
print("=" * 80)
print("Yahoo games found :", len(game_log))
print("Schedule matches  :", int(merged_game_log["SCHEDULE_MATCH"].sum()))
print("Schedule missing  :", int((~merged_game_log["SCHEDULE_MATCH"]).sum()))

if not merged_game_log.empty:
    print("\n" + merged_game_log.to_string(index=False))

missing_rows = merged_game_log.loc[
    ~merged_game_log["SCHEDULE_MATCH"], ["DATE", "TEAM"]
]
if not missing_rows.empty:
    print("\nUNMATCHED DATE + TEAM:")
    print(missing_rows.to_string(index=False))

OUTPUT_FILE = "jokic_yahoo_schedule_merge.csv"
merged_game_log.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("\nSaved:", OUTPUT_FILE)
print("\nTEST COMPLETE")
print("=" * 80)
