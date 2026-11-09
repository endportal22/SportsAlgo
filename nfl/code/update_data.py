#!/usr/bin/env python3
"""
Download / refresh the data the model needs, straight from nflverse.

  data/games.csv          full schedule + results + betting lines + rest days (all seasons)
  data/pbp_<season>.csv.gz  slimmed play-by-play, one file per season

Finished seasons are downloaded once and cached. The current season is
re-downloaded every run (nflverse refreshes it nightly during the season).

Usage:
  python update_data.py                 # default: 2016 -> current season
  python update_data.py --start 2012    # more history
  python update_data.py --force         # re-download everything
"""
import argparse
import io
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

# Only the columns the feature builder uses (keeps files ~10x smaller and loads fast)
PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "pass", "rush", "epa", "success", "wp", "yards_gained",
    "qb_kneel", "qb_spike", "qb_dropback", "qb_scramble",
    "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
]
REL = "https://github.com/nflverse/nflverse-data/releases/download/"
SNAP_URL = REL + "snap_counts/snap_counts_{season}.csv"
INJ_URL = REL + "injuries/injuries_{season}.csv"
ROSTER_URL = REL + "weekly_rosters/roster_weekly_{season}.csv"
PLAYERS_URL = REL + "players/players.csv"
SNAP_COLS = ["game_id", "season", "game_type", "week", "player", "pfr_player_id", "position", "team", "opponent",
             "offense_snaps", "offense_pct", "defense_snaps", "defense_pct"]
INJ_COLS = ["season", "game_type", "team", "week", "gsis_id", "position", "full_name", "report_primary_injury",
            "report_status", "practice_status"]
ROSTER_COLS = ["season", "week", "team", "position", "depth_chart_position", "status", "status_description_abbr",
               "full_name", "gsis_id", "pfr_id", "headshot_url", "jersey_number"]
PLAYER_COLS = ["gsis_id", "display_name", "pfr_id", "position", "position_group", "latest_team", "headshot",
               "birth_date", "rookie_season", "draft_round", "draft_pick"]


def _certifi_context():
    try:  # some macOS Pythons ship without system certificates; certifi fixes that
        import certifi, ssl
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-algo"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.URLError as e:
        ctx = _certifi_context()
        if ctx is None or "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
            return r.read()


def _has_cols(path, cols):
    have = set(pd.read_csv(path, nrows=0).columns)
    return all(c in have for c in cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--start", type=int, default=2016, help="first season to download (default 2016)")
    ap.add_argument("--force", action="store_true", help="re-download finished seasons too")
    args = ap.parse_args()

    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)

    print("[games] downloading schedule/results/lines ...")
    games = pd.read_csv(io.BytesIO(fetch(GAMES_URL)), low_memory=False)
    games.to_csv(data / "games.csv", index=False)
    current = int(games.loc[games["home_score"].notna(), "season"].max())
    print(f"[games] {len(games):,} games; current season = {current}")

    for season in range(args.start, current + 1):
        fresh = season == current or args.force
        out = data / f"pbp_{season}.csv.gz"
        if fresh or not out.exists() or not _has_cols(out, PBP_COLS):
            print(f"[pbp] {season} ...", end=" ", flush=True)
            raw = fetch(PBP_URL.format(season=season))
            pbp = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in PBP_COLS, low_memory=False)
            pbp.to_csv(out, index=False, compression="gzip")
            print(f"{len(pbp):,} plays, {pbp['game_id'].nunique()} games")
        for name, url, cols in (("snaps", SNAP_URL, SNAP_COLS), ("injuries", INJ_URL, INJ_COLS)):
            out = data / f"{name}_{season}.csv.gz"
            if fresh or not out.exists():
                df = pd.read_csv(io.BytesIO(fetch(url.format(season=season))), low_memory=False,
                                 usecols=lambda c: c in cols)
                df.to_csv(out, index=False, compression="gzip")
                print(f"[{name}] {season}: {len(df):,} rows")

    print("[rosters] current-season weekly rosters ...")
    r = pd.read_csv(io.BytesIO(fetch(ROSTER_URL.format(season=current))), low_memory=False, usecols=lambda c: c in ROSTER_COLS)
    r.to_csv(data / f"rosters_{current}.csv.gz", index=False, compression="gzip")
    print("[players] player directory ...")
    p = pd.read_csv(io.BytesIO(fetch(PLAYERS_URL)), low_memory=False, usecols=lambda c: c in PLAYER_COLS)
    p.to_csv(data / "players.csv.gz", index=False, compression="gzip")

    print("done.")


if __name__ == "__main__":
    main()
