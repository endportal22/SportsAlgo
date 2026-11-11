#!/usr/bin/env python3
"""
Download NBA data into nba/data:

  schedule_<season>.csv    every game: date, tip-off, teams, final score
  box_<season>.csv.gz      every player's box score line (minutes, points, ... plus-minus)
  odds.csv                 betting lines (only shown next to the model, never used by it)
  injuries.csv             today's injury report
  rosters.csv              current rosters (catches trades and signings)
  stints_<season>.csv.gz   who was on the floor together, from the play-by-play (see stints.py)

Finished seasons come from sportsdataverse (ESPN's data, rebuilt daily as CSV files on GitHub);
the last few days and upcoming games come straight from ESPN, so the newest games are there
even if sportsdataverse hasn't caught up yet.

Usage: python update_data.py [--start 2015] [--force]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA, current_season, now_utc, today

SDV = os.environ.get("SDV_RELEASES", "https://github.com/sportsdataverse/sportsdataverse-data/releases/download")
ESPN_SITE = os.environ.get("ESPN_SITE", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba")
WORKERS = int(os.environ.get("NBA_ALGO_WORKERS", "8"))
# ESPN's short abbreviations -> the usual three letters
ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "NJ": "BKN"}
BOX_COLS = ["game_id", "team", "player_id", "name", "pos", "starter", "min", "pts", "fgm", "fga", "tpm", "tpa",
            "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm", "dnp"]
SCHED_COLS = ["game_id", "season", "season_type", "date", "start_utc", "neutral", "completed", "home", "away",
              "home_score", "away_score", "note"]


def ab(x):
    return ABBR.get(x, x) if isinstance(x, str) else x


# ------------------------------------------------------------------ http
_ctx = {"value": None, "tried": False}
_lock = threading.Lock()


def _certifi():
    with _lock:
        if not _ctx["tried"]:
            _ctx["tried"] = True
            try:  # some macOS Pythons ship without system certificates
                import certifi
                _ctx["value"] = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                pass
        return _ctx["value"]


def fetch(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sports-algo)", "Accept": "*/*"})
    ctx = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1) * (3 if e.code == 429 else 1))
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e) and ctx is None and _certifi() is not None:
                ctx = _certifi()
                continue
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except (TimeoutError, ConnectionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(url)


def get_json(url, **kw):
    return json.loads(fetch(url, **kw))


def _num(x):
    try:
        f = float(str(x).replace("+", ""))
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ sportsdataverse season files
def sdv_season(season: int):
    """(schedule, box) for a season from sportsdataverse's CSVs. ESPN labels 2026-27 as 2027."""
    y = season + 1
    s = pd.read_csv(io.BytesIO(fetch(f"{SDV}/espn_nba_schedules/nba_schedule_{y}.csv", timeout=180)), low_memory=False)
    s = s[s.season_type.isin([2, 3, 5])]
    sched = pd.DataFrame({
        "game_id": s.game_id.astype(int), "season": season, "season_type": s.season_type.astype(int),
        "date": pd.to_datetime(s.game_date).dt.strftime("%Y-%m-%d"),
        "start_utc": pd.to_datetime(s.date, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%MZ"),
        "neutral": s.neutral_site.fillna(False).astype(bool),
        "completed": s.status_type_completed.fillna(False).astype(bool),
        "home": s.home_abbreviation.map(ab), "away": s.away_abbreviation.map(ab),
        "home_score": pd.to_numeric(s.home_score, errors="coerce"), "away_score": pd.to_numeric(s.away_score, errors="coerce"),
        "note": s.get("notes_headline")})
    sched.loc[~sched.completed, ["home_score", "away_score"]] = np.nan
    try:
        p = pd.read_csv(io.BytesIO(fetch(f"{SDV}/espn_nba_player_boxscores/player_box_{y}.csv", timeout=300)), low_memory=False)
    except urllib.error.HTTPError:
        p = pd.DataFrame()
    if p.empty:
        return sched, pd.DataFrame(columns=BOX_COLS)
    box = pd.DataFrame({
        "game_id": p.game_id.astype(int), "team": p.team_abbreviation.map(ab), "player_id": p.athlete_id.astype("Int64"),
        "name": p.athlete_display_name, "pos": p.athlete_position_abbreviation, "starter": p.starter.fillna(False).astype(bool),
        "min": pd.to_numeric(p.minutes, errors="coerce"), "pts": p.points, "fgm": p.field_goals_made,
        "fga": p.field_goals_attempted, "tpm": p.three_point_field_goals_made, "tpa": p.three_point_field_goals_attempted,
        "ftm": p.free_throws_made, "fta": p.free_throws_attempted, "oreb": p.offensive_rebounds,
        "dreb": p.defensive_rebounds, "ast": p.assists, "stl": p.steals, "blk": p.blocks, "tov": p.turnovers,
        "pf": p.fouls, "pm": p.plus_minus, "dnp": p.did_not_play.fillna(False).astype(bool)})
    box = box[box.game_id.isin(sched.game_id) & box.player_id.notna()]
    return sched, box


# ------------------------------------------------------------------ ESPN (recent + upcoming)
def _comp_team(c):
    return ab((c.get("team") or {}).get("abbreviation"))


def parse_scoreboard_event(ev, season):
    comp = (ev.get("competitions") or [{}])[0]
    st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
    row = {"game_id": int(ev["id"]), "season": season,
           "season_type": int(((ev.get("season") or {}).get("type")) or 2),
           "start_utc": pd.Timestamp(ev.get("date")).strftime("%Y-%m-%dT%H:%MZ") if ev.get("date") else None,
           "neutral": bool(comp.get("neutralSite", False)), "completed": bool(st.get("completed", False)),
           "note": ((comp.get("notes") or [{}])[0] or {}).get("headline") if comp.get("notes") else None}
    for c in comp.get("competitors", []) or []:
        side = c.get("homeAway")
        row[side] = _comp_team(c)
        row[f"{side}_score"] = _num(c.get("score")) if row["completed"] else None
    if row["start_utc"]:
        row["date"] = pd.Timestamp(ev["date"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
    return row, comp.get("odds") or []


def parse_summary_box(js):
    """Player box score lines from an ESPN game summary."""
    gid = int((js.get("header") or {}).get("id") or js.get("gameId"))
    rows = []
    for tp in (js.get("boxscore") or {}).get("players", []) or []:
        team = ab((tp.get("team") or {}).get("abbreviation"))
        for st in tp.get("statistics", []) or []:
            keys = st.get("keys") or []
            for a in st.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                v = dict(zip(keys, a.get("stats") or []))

                def pair(k):
                    x = str(v.get(k, "") or "")
                    return [_num(y) for y in x.split("-")] if "-" in x else [None, None]
                fg, tp3, ft = pair("fieldGoalsMade-fieldGoalsAttempted"), pair("threePointFieldGoalsMade-threePointFieldGoalsAttempted"), pair("freeThrowsMade-freeThrowsAttempted")
                rows.append({"game_id": gid, "team": team, "player_id": int(ath["id"]), "name": ath.get("displayName"),
                             "pos": (ath.get("position") or {}).get("abbreviation"), "starter": bool(a.get("starter")),
                             "min": _num(v.get("minutes")), "pts": _num(v.get("points")), "fgm": fg[0], "fga": fg[1],
                             "tpm": tp3[0], "tpa": tp3[1], "ftm": ft[0], "fta": ft[1], "oreb": _num(v.get("offensiveRebounds")),
                             "dreb": _num(v.get("defensiveRebounds")), "ast": _num(v.get("assists")), "stl": _num(v.get("steals")),
                             "blk": _num(v.get("blocks")), "tov": _num(v.get("turnovers")), "pf": _num(v.get("fouls")),
                             "pm": _num(v.get("plusMinus")), "dnp": bool(a.get("didNotPlay"))})
    return pd.DataFrame(rows, columns=BOX_COLS)


# ------------------------------------------------------------------ betting lines
def american_to_prob(ml):
    ml = _num(ml)
    if not ml:
        return None
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _ml(side, which=None):
    """Moneyline from either of ESPN's odds formats."""
    v = side.get("moneyLine")
    if v not in (None, ""):
        return _num(v)
    return None


BOOK_PRIORITY = ["ESPN BET", "DraftKings", "Caesars", "FanDuel", "consensus", "teamrankings", "numberfire"]


def market_line(items):
    """Vig-free home win probability (median over books) + the line to display, from ESPN odds entries."""
    good = []
    for it in items or []:
        prov = (it.get("provider") or {}).get("name") or ""
        if "live" in prov.lower():
            continue
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        hml, aml = _ml(h), _ml(a)
        ml = it.get("moneyline") or {}
        if hml is None and isinstance(ml, dict):   # newer scoreboard format
            for k in ("close", "current", "open"):
                hml = hml if hml is not None else _num(((ml.get("home") or {}).get(k) or {}).get("odds"))
                aml = aml if aml is not None else _num(((ml.get("away") or {}).get(k) or {}).get("odds"))
        ph, pa = american_to_prob(hml), american_to_prob(aml)
        spread = _num(it.get("spread"))
        p = None
        if ph is not None and pa is not None and 0.98 <= ph + pa <= 1.12:
            p = ph / (ph + pa)
        good.append({"book": prov, "home_ml": hml, "away_ml": aml, "spread": spread, "total": _num(it.get("overUnder")),
                     "details": it.get("details"), "p": p})
    if not good:
        return None
    probs = sorted(g["p"] for g in good if g["p"] is not None)
    best = sorted(good, key=lambda g: (g["p"] is None, BOOK_PRIORITY.index(g["book"]) if g["book"] in BOOK_PRIORITY else 99))[0]
    if probs:
        n = len(probs)
        med = probs[n // 2] if n % 2 else 0.5 * (probs[n // 2 - 1] + probs[n // 2])
    elif best["spread"] is not None:   # no moneylines: convert the spread (home line) with a normal curve
        from math import erf, sqrt
        med = 0.5 * (1 + erf((-best["spread"] / 12.0) / sqrt(2)))
    else:
        return None
    return {"book": best["book"], "home_ml": best["home_ml"], "away_ml": best["away_ml"], "spread": best["spread"],
            "total": best["total"], "details": best["details"], "market_home_prob": round(float(med), 4)}


# ------------------------------------------------------------------ main steps
def season_files(data: Path, season: int):
    return data / f"schedule_{season}.csv", data / f"box_{season}.csv.gz"


def update_seasons(data, seasons, cur, force=False):
    for s in seasons:
        fs, fb = season_files(data, s)
        if fs.exists() and fb.exists() and s < cur and not force:
            continue
        try:
            sched, box = sdv_season(s)
        except Exception as e:  # noqa: BLE001
            print(f"[seasons] {s}: sportsdataverse unavailable ({type(e).__name__}: {e})")
            continue
        if s == cur and fb.exists():   # keep anything ESPN gave us that sportsdataverse doesn't have yet
            old = pd.read_csv(fb)
            box = pd.concat([box, old[~old.game_id.isin(box.game_id)]], ignore_index=True)
            olds = pd.read_csv(fs)
            done = olds[olds.completed & ~olds.game_id.isin(sched[sched.completed].game_id)]
            sched = pd.concat([sched[~sched.game_id.isin(done.game_id)], done], ignore_index=True)
        sched.sort_values(["date", "start_utc", "game_id"])[SCHED_COLS].to_csv(fs, index=False)
        box.to_csv(fb, index=False, compression="gzip")
        print(f"[seasons] {s}-{str(s + 1)[-2:]}: {len(sched)} games ({int(sched.completed.sum())} played), {box.game_id.nunique()} box scores")


def update_recent(data, cur, back=5, ahead=14):
    """ESPN scoreboard for the last few days and next two weeks: results, tip-off times, lines;
    then ESPN box scores for any finished game sportsdataverse doesn't have yet."""
    fs, fb = season_files(data, cur)
    sched = pd.read_csv(fs) if fs.exists() else pd.DataFrame(columns=SCHED_COLS)
    box = pd.read_csv(fb) if fb.exists() else pd.DataFrame(columns=BOX_COLS)
    days = [today() + timedelta(days=d) for d in range(-back, ahead + 1)]
    rows, odds = {}, {}
    now = pd.Timestamp(now_utc())

    def one(d):
        return get_json(f"{ESPN_SITE}/scoreboard?dates={d.strftime('%Y%m%d')}&limit=100", timeout=30, retries=3)

    fails = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(one, d) for d in days]):
            try:
                js = fut.result()
            except Exception:  # noqa: BLE001
                fails += 1
                continue
            for ev in js.get("events", []) or []:
                try:
                    r, o = parse_scoreboard_event(ev, cur)
                except Exception:  # noqa: BLE001
                    continue
                if r["season_type"] not in (2, 3, 5) or not r.get("home"):
                    continue
                rows[r["game_id"]] = r
                m = market_line(o)
                if m and not r["completed"] and pd.Timestamp(r["start_utc"]) > now:
                    odds[r["game_id"]] = {**m, "pregame": True}
    if rows:
        new = pd.DataFrame(list(rows.values()))
        sched = pd.concat([sched[~sched.game_id.isin(new.game_id)], new], ignore_index=True)
        sched = sched.sort_values(["date", "start_utc", "game_id"])[SCHED_COLS]
        sched.to_csv(fs, index=False)
    print(f"[espn] scoreboard: {len(rows)} games over {len(days)} days" + (f" ({fails} days failed)" if fails else ""))

    # box scores (and closing lines) for finished games we don't have yet
    sched["completed"] = sched.completed.astype(str).str.lower().isin(["true", "1", "1.0"])
    have = set(box.game_id)
    need = sched[sched.completed & ~sched.game_id.isin(have)].game_id.tolist()
    oddsf = data / "odds.csv"
    oddsdf = pd.read_csv(oddsf) if oddsf.exists() else pd.DataFrame(columns=["game_id"])
    recent_done = sched[sched.completed & (pd.to_datetime(sched.date) >= pd.Timestamp(today() - timedelta(days=back)))].game_id
    with_line = set(oddsdf[oddsdf.get("market_home_prob", pd.Series(dtype=float)).notna()].game_id) if not oddsdf.empty else set()
    need_line = [g for g in recent_done if g not in with_line or g in need]
    todo = sorted(set(need) | set(need_line))
    got, added = 0, []
    if todo:
        def summ(g):
            return g, get_json(f"{ESPN_SITE}/summary?event={g}", timeout=30, retries=3)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(summ, g) for g in todo]):
                try:
                    g, js = fut.result()
                except Exception:  # noqa: BLE001
                    continue
                if g in need:
                    b = parse_summary_box(js)
                    if not b.empty:
                        added.append(b)
                        got += 1
                m = market_line(js.get("pickcenter"))
                if m:
                    odds[g] = {**m, "pregame": False}
    if added:
        box = pd.concat([box] + added, ignore_index=True)
        box.to_csv(fb, index=False, compression="gzip")
    print(f"[espn] box scores: {got}/{len(need)} missing games added")
    if odds:
        stamp = now.isoformat(timespec="minutes")
        new = pd.DataFrame([{"game_id": g, **v, "fetched": stamp} for g, v in odds.items()])
        keep = oddsdf[~oddsdf.game_id.isin(new.game_id)] if not oddsdf.empty else oddsdf
        out = pd.concat([keep, new], ignore_index=True).sort_values("game_id")
        out.to_csv(oddsf, index=False)
        print(f"[odds] {len(new)} games updated, {int(out.market_home_prob.notna().sum()):,} with a line")


def update_stints(data, seasons, cur):
    """Play-by-play -> on-floor stints, for seasons that don't have them yet and for new games this
    season (sportsdataverse's play-by-play files are large, so the current one is re-read at most
    every 6 hours)."""
    import stints as ST
    for s in seasons:
        fst, fb = data / f"stints_{s}.csv.gz", data / f"box_{s}.csv.gz"
        fs = data / f"schedule_{s}.csv"
        if not fb.exists() or not fs.exists():
            continue
        stamp = data / f".stints_{s}.fetched"
        if fst.exists():
            if s < cur:
                continue
            have = set(pd.read_csv(fst, usecols=["game_id"]).game_id)
            need = set(pd.read_csv(fb, usecols=["game_id"]).game_id) - have
            if not need or (stamp.exists() and time.time() - stamp.stat().st_mtime < 6 * 3600):
                continue
        try:
            raw = fetch(f"{SDV}/espn_nba_pbp/play_by_play_{s + 1}.csv", timeout=600)
            pbp = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in ST.PBP_COLS, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"[stints] {s}: play-by-play unavailable ({type(e).__name__}: {e})")
            continue
        sched = pd.read_csv(fs)
        box = pd.read_csv(fb, dtype={"pm": str})
        st = ST.build_season(pbp, box, sched)
        st.to_csv(fst, index=False, compression="gzip")
        stamp.touch()
        print(f"[stints] {s}-{str(s + 1)[-2:]}: {len(st):,} stints from {st.game_id.nunique()} games")


def update_injuries(data):
    try:
        js = get_json(f"{ESPN_SITE}/injuries", timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"[injuries] unavailable ({type(e).__name__}: {e})")
        return
    rows = []
    for t in js.get("injuries", []) or []:
        tname = t.get("displayName")
        for inj in t.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            team = ab((ath.get("team") or {}).get("abbreviation"))
            pid = ath.get("id")
            if pid is None:
                for l in ath.get("links", []) or []:
                    h = l.get("href", "")
                    if "/id/" in h:
                        pid = h.split("/id/")[1].split("/")[0]
                        break
            det = inj.get("details") or {}
            rows.append({"team": team, "team_name": tname, "player_id": _num(pid), "name": ath.get("displayName"),
                         "status": inj.get("status"), "detail": det.get("type") or det.get("detail"),
                         "return_date": det.get("returnDate"), "comment": inj.get("shortComment"), "date": inj.get("date")})
    df = pd.DataFrame(rows)
    df["fetched"] = today().isoformat()
    df.to_csv(data / "injuries.csv", index=False)
    print(f"[injuries] {len(df)} players listed")


def update_rosters(data, teams):
    rows = []

    espn = {v: k for k, v in ABBR.items() if k != "NJ"}

    def one(team):
        js = get_json(f"{ESPN_SITE}/teams/{espn.get(team, team).lower()}/roster", timeout=30)
        out = []
        for a in js.get("athletes", []) or []:
            items = a.get("items") if isinstance(a, dict) and "items" in a else [a]
            for p in items:
                out.append({"team": team, "player_id": int(p["id"]), "name": p.get("displayName"),
                            "pos": (p.get("position") or {}).get("abbreviation")})
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(one, t) for t in teams]):
            try:
                rows += fut.result()
            except Exception:  # noqa: BLE001
                pass
    if rows:
        df = pd.DataFrame(rows)
        df["fetched"] = today().isoformat()
        df.to_csv(data / "rosters.csv", index=False)
        print(f"[rosters] {df.team.nunique()} teams, {len(df)} players")
    else:
        print("[rosters] unavailable")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--start", type=int, default=2015, help="first season (start year), default 2015 (=2015-16)")
    ap.add_argument("--force", action="store_true", help="re-download finished seasons too")
    ap.add_argument("--no-espn", action="store_true", help="skip ESPN (sportsdataverse files only)")
    args = ap.parse_args()
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    cur = current_season()
    update_seasons(data, range(args.start, cur + 1), cur, args.force)
    update_stints(data, range(args.start, cur + 1), cur)
    if not args.no_espn:
        try:
            update_recent(data, cur)
        except Exception as e:  # noqa: BLE001
            print(f"[espn] unavailable ({type(e).__name__}: {e})")
        update_injuries(data)
        sched = pd.read_csv(season_files(data, cur)[0])
        update_rosters(data, sorted(set(sched.home.dropna()) | set(sched.away.dropna())))
    print("done.")


if __name__ == "__main__":
    main()
