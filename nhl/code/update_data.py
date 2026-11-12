#!/usr/bin/env python3
"""
Download / refresh the data the model needs. No API keys: everything comes from the
NHL's public API (api-web.nhle.com) and ESPN's public API (betting lines, injuries).

  data/games.csv                 schedule + results, regular season and playoffs, all seasons
  data/teams.csv                 team abbreviations and full names
  data/shots_<season>.csv.gz     every shot attempt: location, shooter, goalie in net, game state
  data/skaters_<season>.csv.gz   per skater per game: time on ice and box score
  data/goalies_<season>.csv.gz   per goalie per game: time on ice, starter, shots and goals against
  data/states_<season>.csv.gz    per team per game: minutes at 5-on-5, on the power play, short-handed
  data/players.csv               player directory (id, full name, position)
  data/odds.csv                  pre-game moneylines (ESPN; NHL/DraftKings for upcoming games)
  data/injuries.csv              current injury list (ESPN)
  data/rosters.csv               current NHL rosters

Finished games are downloaded once and kept. Each run only fetches games that are new
since the last run, plus the schedule, lines and injuries for the current season.
The first run downloads every game since --start (two requests per game), so it
takes a while (roughly 15-30 minutes for 8 seasons). If it is interrupted, just
run it again: it carries on where it stopped.

Usage:
  python update_data.py                 # default: 2018-19 season -> current season
  python update_data.py --start 2015    # more history
  python update_data.py --no-odds       # skip the betting lines
"""
from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from paths import DATA, now_utc, today

sys.path.append(str(Path(__file__).resolve().parents[2] / "shared"))   # repo layout; a copy next to this file wins
try:
    import statuslog
except ImportError:
    statuslog = None

NHL = os.environ.get("NHL_API", "https://api-web.nhle.com/v1")
ESPN_SITE = os.environ.get("ESPN_SITE", "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl")
ESPN_CORE = os.environ.get("ESPN_CORE", "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl")
WORKERS = int(os.environ.get("NHL_ALGO_WORKERS", "8"))

SHOT_EVENTS = {"goal": "G", "shot-on-goal": "S", "missed-shot": "M", "blocked-shot": "B"}
# ESPN abbreviations that differ from the NHL's
ESPN_ABBR = {"NJ": "NJD", "LA": "LAK", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA", "UTM": "UTA", "MON": "MTL",
             "WAS": "WSH", "CLB": "CBJ", "NAS": "NSH", "LV": "VGK", "VEG": "VGK", "PHX": "ARI"}


# ------------------------------------------------------------------ http
_ctx_lock = threading.Lock()
_ctx = {"value": None, "tried": False}


def _certifi_context():
    with _ctx_lock:
        if not _ctx["tried"]:
            _ctx["tried"] = True
            try:  # some macOS Pythons ship without system certificates; certifi fixes that
                import certifi
                _ctx["value"] = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                _ctx["value"] = None
        return _ctx["value"]


def fetch(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nhl-algo)", "Accept": "application/json"})
    ctx = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1) * (3 if e.code == 429 else 1))
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e) and ctx is None:
                ctx = _certifi_context()
                if ctx is None:
                    raise
                continue
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except (TimeoutError, ConnectionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def get_json(url: str, **kw):
    return json.loads(fetch(url, **kw))


def _txt(x):
    """NHL API text fields are {"default": "...", "fr": "..."}."""
    if isinstance(x, dict):
        return x.get("default") or next(iter(x.values()), None)
    return x


def _sec(s):
    """'17:23' -> 1043 seconds."""
    if not isinstance(s, str) or ":" not in s:
        return 0
    m, sec = s.split(":")[:2]
    try:
        return int(m) * 60 + int(sec)
    except ValueError:
        return 0


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())


# ------------------------------------------------------------------ schedule
def _sched_row(g, day, y):
    h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
    st = g.get("gameState", "")
    out = (g.get("gameOutcome") or {}).get("lastPeriodType")
    return {
        "game_id": int(g["id"]), "season": y, "game_type": "R" if g.get("gameType") == 2 else "P",
        "gameday": day, "start_utc": g.get("startTimeUTC"),
        "home_team": h.get("abbrev"), "away_team": a.get("abbrev"),
        "home_score": h.get("score") if st in ("OFF", "FINAL") else None,
        "away_score": a.get("score") if st in ("OFF", "FINAL") else None,
        "state": st, "schedule_state": g.get("gameScheduleState"), "last_period": out,
        "neutral": bool(g.get("neutralSite", False)), "venue": _txt(g.get("venue")),
        "venue_tz": g.get("venueTimezone"),
        "home_name": f"{_txt(h.get('placeName')) or ''} {_txt(h.get('commonName')) or ''}".strip(),
        "away_name": f"{_txt(a.get('placeName')) or ''} {_txt(a.get('commonName')) or ''}".strip(),
    }


def season_schedule(y: int) -> pd.DataFrame:
    """All regular-season and playoff games of the season starting in year y."""
    sid = f"{y}{y + 1}"
    d, end = date(y, 8, 15), date(y + 1, 10, 31)
    games = {}
    while d <= end:
        try:
            js = get_json(f"{NHL}/schedule/{d.isoformat()}")
        except urllib.error.HTTPError:
            d += timedelta(days=7)
            continue
        for day in js.get("gameWeek", []) or []:
            for g in day.get("games", []) or []:
                if str(g.get("season")) == sid and g.get("gameType") in (2, 3):
                    games[int(g["id"])] = _sched_row(g, day.get("date"), y)
        rs, pe = js.get("regularSeasonStartDate"), js.get("playoffEndDate")
        nxt = js.get("nextStartDate")
        nd = date.fromisoformat(nxt) if nxt else d + timedelta(days=7)
        d = nd if nd > d else d + timedelta(days=7)
        if rs and pe and rs >= f"{y}-08-01" and d > date.fromisoformat(pe) + timedelta(days=3):
            break
    df = pd.DataFrame(list(games.values()))
    if not df.empty:
        df = df[df.schedule_state.fillna("OK") != "CNCL"]
    return df


# ------------------------------------------------------------------ one game
def parse_game(gid: int, box: dict, pbp: dict, season: int):
    """Box score + play-by-play -> (shots, skaters, goalies, states, players) rows."""
    home, away = pbp.get("homeTeam") or box.get("homeTeam"), pbp.get("awayTeam") or box.get("awayTeam")
    H, A = home["abbrev"], away["abbrev"]
    tid = {home.get("id"): H, away.get("id"): A}
    pteam, skaters, goalies, players = {}, [], [], {}

    pbg = box.get("playerByGameStats") or {}
    for side, team in (("homeTeam", H), ("awayTeam", A)):
        st = pbg.get(side) or {}
        for grp in ("forwards", "defense"):
            for p in st.get(grp, []) or []:
                pid = p.get("playerId")
                if pid is None:
                    continue
                pteam[pid] = team
                skaters.append({"game_id": gid, "team": team, "player_id": pid, "name": _txt(p.get("name")),
                                "pos": p.get("position") or ("D" if grp == "defense" else "F"),
                                "toi": _sec(p.get("toi")), "goals": p.get("goals", 0) or 0,
                                "assists": p.get("assists", 0) or 0, "sog": p.get("sog", p.get("shots", 0)) or 0,
                                "hits": p.get("hits", 0) or 0, "blocks": p.get("blockedShots", 0) or 0,
                                "pim": p.get("pim", 0) or 0, "shifts": p.get("shifts", 0) or 0})
        gl = []
        for p in st.get("goalies", []) or []:
            pid = p.get("playerId")
            if pid is None:
                continue
            pteam[pid] = team
            gl.append({"game_id": gid, "team": team, "player_id": pid, "name": _txt(p.get("name")),
                       "toi": _sec(p.get("toi")), "starter": p.get("starter"),
                       "shots_against": p.get("shotsAgainst"), "saves": p.get("saves"),
                       "goals_against": p.get("goalsAgainst"), "decision": p.get("decision")})
        gl = [g for g in gl if g["toi"] > 0 or g["starter"]]
        if gl and not any(g["starter"] for g in gl):
            max(gl, key=lambda g: g["toi"])["starter"] = True
        for g in gl:
            g["starter"] = bool(g["starter"])
        goalies += gl

    for r in pbp.get("rosterSpots", []) or []:
        pid = r.get("playerId")
        team = tid.get(r.get("teamId"))
        if pid is None:
            continue
        if team:
            pteam.setdefault(pid, team)
        players[pid] = {"player_id": pid, "first": _txt(r.get("firstName")), "last": _txt(r.get("lastName")),
                        "pos": r.get("positionCode"), "team": team, "season": season}

    plays = [p for p in pbp.get("plays", []) or [] if (p.get("periodDescriptor") or {}).get("periodType") != "SO"]
    plays.sort(key=lambda p: p.get("sortOrder", 0))

    # which way each team attacks in each period, from its offensive-zone shots
    # (older seasons don't say which end each team defends)
    att = defaultdict(float)
    for p in plays:
        d = p.get("details") or {}
        if p.get("typeDescKey") in ("shot-on-goal", "missed-shot", "goal") and d.get("zoneCode") == "O" and d.get("xCoord"):
            team = pteam.get(d.get("shootingPlayerId") or d.get("scoringPlayerId")) or tid.get(d.get("eventOwnerTeamId"))
            if team:
                att[((p.get("periodDescriptor") or {}).get("number"), team)] += 1 if d["xCoord"] > 0 else -1

    # time in each game state, per team (from the gaps between consecutive events)
    toi = {H: defaultdict(float), A: defaultdict(float)}
    sc_last = "1551"
    shots = []
    last_attempt = {H: (-1, -99), A: (-1, -99)}  # (period, t)
    prev = None
    goals = {H: 0, A: 0}
    for i, p in enumerate(plays):
        per = (p.get("periodDescriptor") or {}).get("number") or 1
        t = (per - 1) * 1200 + _sec(p.get("timeInPeriod"))
        sc = p.get("situationCode") or sc_last
        if not (isinstance(sc, str) and len(sc) == 4 and sc.isdigit()):
            sc = sc_last
        sc_last = sc
        ag, ask, hsk, hg = (int(c) for c in sc)
        if i + 1 < len(plays):
            nper = (plays[i + 1].get("periodDescriptor") or {}).get("number") or 1
            if nper == per:
                dt = (nper - 1) * 1200 + _sec(plays[i + 1].get("timeInPeriod")) - t
                if 0 < dt < 1200:
                    for team, own, opp, og, pg in ((H, hsk, ask, hg, ag), (A, ask, hsk, ag, hg)):
                        if og and pg:
                            k = "toi_5v5" if own == opp == 5 else "toi_pp" if own > opp else "toi_pk" if own < opp else "toi_ev_other"
                        else:
                            k = "toi_en"
                        toi[team][k] += dt

        typ = p.get("typeDescKey")
        d = p.get("details") or {}
        if typ in SHOT_EVENTS:
            shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId")
            team = pteam.get(shooter)
            if team is None:
                owner = tid.get(d.get("eventOwnerTeamId"))
                if typ == "blocked-shot" or owner is None:
                    prev = p
                    continue
                team = owner
            is_home = team == H
            own_sk, opp_sk = (hsk, ask) if is_home else (ask, hsk)
            own_g, opp_g = (hg, ag) if is_home else (ag, hg)
            x, y = d.get("xCoord"), d.get("yCoord")
            zone = d.get("zoneCode")
            hds = p.get("homeTeamDefendingSide")
            xo = dist = angle = None
            if x is not None and y is not None:
                if zone == "O" and x != 0:
                    sign = 1 if x > 0 else -1
                elif zone == "D" and x != 0 and typ != "blocked-shot":
                    sign = -1 if x > 0 else 1
                elif hds in ("left", "right"):
                    sign = 1 if (is_home == (hds == "left")) else -1
                elif att.get((per, team)):
                    sign = 1 if att[(per, team)] > 0 else -1
                else:
                    sign = 1 if x >= 0 else -1
                xo = x * sign
                dist = math.hypot(89 - xo, y)
                angle = math.degrees(math.atan2(abs(y), 89 - xo))
            lp, lt = last_attempt[team]
            rebound = int(lp == per and 0 <= t - lt <= 3)
            last_attempt[team] = (per, t)
            rush = 0
            if prev is not None:
                pper = (prev.get("periodDescriptor") or {}).get("number") or 1
                pt = (pper - 1) * 1200 + _sec(prev.get("timeInPeriod"))
                pd_ = prev.get("details") or {}
                pz = pd_.get("zoneCode")
                powner = tid.get(pd_.get("eventOwnerTeamId"))
                if pz and pper == per and 0 <= t - pt <= 4:
                    zp = pz if powner == team else {"O": "D", "D": "O"}.get(pz, pz)
                    rush = int(zp in ("N", "D"))
            if opp_g == 0:
                strength = "EN"
            elif own_sk == opp_sk == 5:
                strength = "5v5"
            elif own_sk > opp_sk:
                strength = "PP"
            elif own_sk < opp_sk:
                strength = "SH"
            else:
                strength = "EV"
            shots.append({"game_id": gid, "period": per, "t": t, "team": team, "opp": A if is_home else H,
                          "is_home": int(is_home), "shooter": shooter, "goalie": d.get("goalieInNetId"),
                          "ev": SHOT_EVENTS[typ], "x": x, "y": y, "xo": xo, "dist": dist, "angle": angle,
                          "shot_type": d.get("shotType"), "zone": zone, "rebound": rebound, "rush": rush,
                          "own_sk": own_sk, "opp_sk": opp_sk, "own_g": own_g, "opp_g": opp_g, "strength": strength,
                          "score_diff": goals[team] - goals[A if is_home else H]})
            if typ == "goal":
                goals[team] += 1
        prev = p

    states = [{"game_id": gid, "team": team, **{k: round(v, 1) for k, v in toi[team].items()}} for team in (H, A)]
    return shots, skaters, goalies, states, list(players.values())


def fetch_game(gid: int, season: int):
    box = get_json(f"{NHL}/gamecenter/{gid}/boxscore")
    pbp = get_json(f"{NHL}/gamecenter/{gid}/play-by-play")
    return parse_game(gid, box, pbp, season)


# ------------------------------------------------------------------ per-season game store
KINDS = ("shots", "skaters", "goalies", "states")


def _path(data, kind, season):
    return data / f"{kind}_{season}.csv.gz"


def _load(data, kind, season):
    f = _path(data, kind, season)
    return pd.read_csv(f, low_memory=False) if f.exists() else pd.DataFrame()


def update_games(data: Path, games: pd.DataFrame, seasons, refetch_days: int = 2):
    """Download box score + play-by-play for every finished game not stored yet."""
    tday = today()
    players_new = []
    for season in seasons:
        g = games[(games.season == season) & games.state.isin(["OFF", "FINAL"])]
        have = {k: _load(data, k, season) for k in KINDS}
        done = set(have["states"].game_id.unique()) if not have["states"].empty else set()
        # games that just finished may be re-published with corrections; refresh the last couple of days
        recent = set(g[pd.to_datetime(g.gameday).dt.date >= tday - timedelta(days=refetch_days)].game_id)
        todo = sorted((set(g.game_id) - done) | (recent & done))
        if not todo:
            print(f"[games] {season}: {len(done)} games, up to date")
            continue
        print(f"[games] {season}: downloading {len(todo)} games ({len(done)} already stored) ...", flush=True)
        for k in KINDS:
            if not have[k].empty:
                have[k] = have[k][~have[k].game_id.isin(todo)]
        new = {k: [] for k in KINDS}
        failed, t0 = [], time.time()
        n_done = 0

        def flush():
            for i, k in enumerate(KINDS):
                if new[k]:
                    have[k] = pd.concat([have[k], pd.DataFrame(new[k])], ignore_index=True)
                    new[k] = []
                if not have[k].empty:
                    have[k].to_csv(_path(data, k, season), index=False, compression="gzip")

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch_game, gid, season): gid for gid in todo}
            for fut in as_completed(futs):
                gid = futs[fut]
                try:
                    sh, sk, gl, st, pl = fut.result()
                except Exception as e:  # noqa: BLE001
                    failed.append((gid, f"{type(e).__name__}: {e}"))
                    continue
                if not st or not sk:
                    failed.append((gid, "empty box score"))
                    continue
                new["shots"] += sh; new["skaters"] += sk; new["goalies"] += gl; new["states"] += st
                players_new += pl
                n_done += 1
                if n_done % 250 == 0:
                    rate = n_done / max(time.time() - t0, 1e-6)
                    print(f"  {n_done}/{len(todo)} games  ({rate:.1f}/s, ~{(len(todo) - n_done) / max(rate, 1e-6) / 60:.0f} min left)",
                          flush=True)
                    flush()
        flush()
        print(f"[games] {season}: stored {n_done} new games" + (f", {len(failed)} failed" if failed else ""), flush=True)
        for gid, why in failed[:5]:
            print(f"   {gid}: {why}")
    return players_new


def update_players(data: Path, new_rows):
    f = data / "players.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["player_id", "first", "last", "pos", "team", "season"])
    if new_rows:
        df = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df = old
    if df.empty:
        return
    df = df.sort_values("season").drop_duplicates("player_id", keep="last")
    df["name"] = (df["first"].fillna("") + " " + df["last"].fillna("")).str.strip()
    df.to_csv(f, index=False)


# ------------------------------------------------------------------ betting lines
def american_to_prob(ml):
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml == 0 or math.isnan(ml):
        return None
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _ml(side):
    """Closing moneyline if ESPN has it, else the current one."""
    for k in ("close", "current"):
        v = ((side.get(k) or {}).get("moneyLine") or {})
        if isinstance(v, dict) and v.get("american") not in (None, ""):
            try:
                return float(str(v["american"]).replace("+", ""))
            except ValueError:
                pass
    return side.get("moneyLine")


PREFERRED_BOOKS = ("58", "40", "100", "31", "38", "45", "52")  # ESPN BET, DraftKings, Caesars ...


def market_from_espn(items):
    """Vig-free home win probability from ESPN's list of sportsbooks (live-odds feeds skipped)."""
    good = []
    for it in items or []:
        prov = it.get("provider") or {}
        name = str(prov.get("name", ""))
        if "live" in name.lower():
            continue
        ho, ao = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        hml, aml = _ml(ho), _ml(ao)
        ph, pa = american_to_prob(hml), american_to_prob(aml)
        if ph is None or pa is None:
            continue
        over = ph + pa
        if not (0.99 <= over <= 1.12):   # inconsistent feed (some books list puck-line or 3-way prices)
            continue
        good.append((str(prov.get("id")), name, hml, aml, ph / over))
    if not good:
        return None
    probs = sorted(g[4] for g in good)
    med = probs[len(probs) // 2] if len(probs) % 2 else 0.5 * (probs[len(probs) // 2 - 1] + probs[len(probs) // 2])
    best = next((g for pid in PREFERRED_BOOKS for g in good if g[0] == pid), good[0])
    return {"home_ml": best[2], "away_ml": best[3], "book": best[1], "market_home_prob": round(med, 4), "n_books": len(good)}


def _has_line(r):
    v = r.get("market_home_prob") if r else None
    try:
        return v is not None and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _espn_events(day: str):
    """ESPN scoreboard for one (Eastern) date -> list of (espn_id, home, away)."""
    js = get_json(f"{ESPN_SITE}/scoreboard?dates={day.replace('-', '')}&limit=100", retries=3, timeout=30)
    out = []
    for ev in js.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        h = a = None
        for c in comp.get("competitors", []) or []:
            ab = (c.get("team") or {}).get("abbreviation")
            ab = ESPN_ABBR.get(ab, ab)
            if c.get("homeAway") == "home":
                h = ab
            else:
                a = ab
        out.append((str(ev.get("id")), h, a))
    return out


def update_odds(data: Path, games: pd.DataFrame, first_season: int):
    f = data / "odds.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["game_id"])
    have = {int(r.game_id): r for r in old.itertuples(index=False)} if not old.empty else {}
    now = now_utc()
    tday = today()
    g = games[games.season >= first_season].copy()
    g["start"] = pd.to_datetime(g.start_utc, utc=True, errors="coerce")
    g["gd"] = pd.to_datetime(g.gameday).dt.date
    upcoming = (g.start > pd.Timestamp(now)) & (g.gd <= tday + timedelta(days=7))
    # games we still need a line for: anything never looked up, anything looked up before
    # its line was posted (retry once it has been played: ESPN keeps the closing line),
    # plus upcoming games (lines move)
    retry = {gid for gid, r in have.items()
             if not _has_line(r._asdict()) and str(getattr(r, "pregame", "")).strip().lower() in ("true", "1", "1.0")}
    need = g[(~g.game_id.isin(list(have)) | g.game_id.isin(list(retry))) & (g.gd <= tday + timedelta(days=7))]
    need = pd.concat([need, g[upcoming & g.game_id.isin(list(have))]]).drop_duplicates("game_id")
    if need.empty:
        print("[odds] up to date")
    else:
        print(f"[odds] looking up lines for {len(need)} games on {need.gameday.nunique()} dates (ESPN) ...", flush=True)
    rows = {}

    def one_day(day, sub):
        out = {}
        try:
            evs = _espn_events(day)
        except Exception as e:  # noqa: BLE001
            return day, out, f"{type(e).__name__}: {e}"
        for r in sub.itertuples(index=False):
            hh = ESPN_ABBR.get(r.home_team, r.home_team)
            match = [e for e in evs if e[1] in (r.home_team, hh) and e[2] in (r.away_team, ESPN_ABBR.get(r.away_team, r.away_team))]
            if not match:
                match = [e for e in evs if e[1] in (r.home_team, hh) or e[2] in (r.away_team, ESPN_ABBR.get(r.away_team, r.away_team))]
                match = match if len(match) == 1 else []
            res = {"game_id": r.game_id, "espn_id": match[0][0] if match else None, "source": "espn",
                   "fetched": now.isoformat(timespec="minutes"), "pregame": bool(r.start > pd.Timestamp(now)) if pd.notna(r.start) else False}
            if match:
                try:
                    js = get_json(f"{ESPN_CORE}/events/{match[0][0]}/competitions/{match[0][0]}/odds", retries=2, timeout=30)
                    m = market_from_espn(js.get("items"))
                    if m:
                        res.update(m)
                except urllib.error.HTTPError:
                    pass
                except Exception:  # noqa: BLE001
                    pass
            out[r.game_id] = res
        return day, out, None

    errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one_day, day, sub) for day, sub in need.groupby("gameday")]
        for i, fut in enumerate(as_completed(futs), 1):
            day, out, err = fut.result()
            if err:
                errors += 1
            rows.update(out)
            if i % 50 == 0:
                print(f"  {i}/{len(futs)} dates", flush=True)

    # NHL's own partner odds (DraftKings) for games that haven't started: fills ESPN gaps
    try:
        js = get_json(f"{NHL}/partner-game/US/now")
        book = (js.get("bettingPartner") or {}).get("name", "NHL partner")
        for pg in js.get("games", []) or []:
            gid = int(pg.get("gameId", 0))
            st = pd.to_datetime(pg.get("startTimeUTC"), utc=True, errors="coerce")
            if gid not in set(g.game_id) or pd.isna(st) or st <= pd.Timestamp(now):
                continue
            def ml(team):
                for o in (team or {}).get("odds", []) or []:
                    if o.get("description") == "MONEY_LINE_2_WAY":
                        return o.get("value")
                return None
            hml, aml = ml(pg.get("homeTeam")), ml(pg.get("awayTeam"))
            ph, pa = american_to_prob(hml), american_to_prob(aml)
            if ph is None or pa is None or not (0.99 <= ph + pa <= 1.12):
                continue
            cur = rows.get(gid) or (have[gid]._asdict() if gid in have else {})
            if not _has_line(cur):
                rows[gid] = {**cur, "game_id": gid, "source": "nhl", "book": book, "home_ml": hml, "away_ml": aml,
                             "market_home_prob": round(ph / (ph + pa), 4), "n_books": 1, "pregame": True,
                             "fetched": now.isoformat(timespec="minutes")}
    except Exception as e:  # noqa: BLE001
        print(f"[odds] NHL partner odds unavailable ({type(e).__name__})")

    if rows:
        merged = {gid: r._asdict() for gid, r in have.items()}
        for gid, r in rows.items():
            if _has_line(r) or not _has_line(merged.get(gid)):   # never replace a real line with an empty lookup
                merged[gid] = r
        out = pd.DataFrame(list(merged.values())).sort_values("game_id")
        out.to_csv(f, index=False)
        n = int(out.market_home_prob.notna().sum()) if "market_home_prob" in out else 0
        print(f"[odds] {n:,} games with a line" + (f" ({errors} dates failed)" if errors else ""))


# ------------------------------------------------------------------ injuries + rosters
def update_injuries(data: Path, teams: pd.DataFrame):
    try:
        js = get_json(f"{ESPN_SITE}/injuries")
    except Exception as e:  # noqa: BLE001
        print(f"[injuries] unavailable ({type(e).__name__}: {e})")
        return
    by_name = {_norm(n): ab for ab, n in zip(teams.abbr, teams.name)}
    by_nick = {_norm(str(n).split(" ")[-1]): ab for ab, n in zip(teams.abbr, teams.name)}
    rows = []
    for t in js.get("injuries", []) or []:
        tname = t.get("displayName", "")
        for inj in t.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            ab = ((ath.get("team") or {}).get("abbreviation"))
            ab = ESPN_ABBR.get(ab, ab) if ab else None
            if ab not in set(teams.abbr):
                ab = by_name.get(_norm(tname)) or by_nick.get(_norm(str(tname).split(" ")[-1]))
            det = inj.get("details") or {}
            typ = inj.get("type") or {}
            rows.append({"team": ab, "name": ath.get("displayName"), "pos": (ath.get("position") or {}).get("abbreviation")
                         if isinstance(ath.get("position"), dict) else ath.get("position"),
                         "status": inj.get("status"), "type": typ.get("description") if isinstance(typ, dict) else typ,
                         "injury": det.get("type"), "detail": det.get("detail"), "return_date": det.get("returnDate"),
                         "comment": inj.get("shortComment"), "date": inj.get("date")})
    df = pd.DataFrame(rows)
    df["fetched"] = today().isoformat()
    df.to_csv(data / "injuries.csv", index=False)
    print(f"[injuries] {len(df)} players listed")
    if statuslog is not None and len(df):
        statuslog.update(data / "injury_log.csv", df.assign(player_id=None)[["team", "player_id", "name", "status", "detail"]],
                         league="nhl")


def update_rosters(data: Path, team_abbrs):
    rows = []

    def one(t):
        js = get_json(f"{NHL}/roster/{t}/current")
        out = []
        for grp in ("forwards", "defensemen", "goalies"):
            for p in js.get(grp, []) or []:
                out.append({"team": t, "player_id": p.get("id"), "first": _txt(p.get("firstName")),
                            "last": _txt(p.get("lastName")), "pos": p.get("positionCode"), "number": p.get("sweaterNumber")})
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(one, t) for t in team_abbrs]):
            try:
                rows += fut.result()
            except Exception:  # noqa: BLE001
                pass
    if rows:
        df = pd.DataFrame(rows)
        df["name"] = (df["first"].fillna("") + " " + df["last"].fillna("")).str.strip()
        df["fetched"] = today().isoformat()
        df.to_csv(data / "rosters.csv", index=False)
        print(f"[rosters] {df.team.nunique()} teams, {len(df)} players")


# ------------------------------------------------------------------ main
def current_season() -> int:
    t = today()
    return t.year if t.month >= 8 else t.year - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--start", type=int, default=2018, help="first season (start year) to download, default 2018 (=2018-19)")
    ap.add_argument("--odds-start", type=int, default=2021, help="first season to look up betting lines for (default 2021)")
    ap.add_argument("--no-odds", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download finished seasons' schedules too")
    args = ap.parse_args()

    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    cur = current_season()
    seasons = list(range(args.start, cur + 1))

    # schedules: finished seasons once, the current one every run
    frames = []
    for s in seasons:
        f = data / f"schedule_{s}.csv"
        if f.exists() and s < cur and not args.force:
            frames.append(pd.read_csv(f))
            continue
        print(f"[schedule] {s}-{str(s + 1)[-2:]} ...", end=" ", flush=True)
        df = season_schedule(s)
        if df.empty:
            print("no games")
            continue
        done = df.state.isin(["OFF", "FINAL"]).sum()
        print(f"{len(df)} games, {done} played")
        df.to_csv(f, index=False)
        frames.append(df)
    games = pd.concat(frames, ignore_index=True).sort_values(["gameday", "start_utc", "game_id"])
    games.to_csv(data / "games.csv", index=False)

    tn = pd.concat([games[["season", "home_team", "home_name"]].rename(columns={"home_team": "abbr", "home_name": "name"}),
                    games[["season", "away_team", "away_name"]].rename(columns={"away_team": "abbr", "away_name": "name"})])
    teams = tn.sort_values("season").drop_duplicates("abbr", keep="last")[["abbr", "name", "season"]]
    teams.to_csv(data / "teams.csv", index=False)

    new_players = update_games(data, games, seasons)
    update_players(data, new_players)

    active = sorted(teams[teams.season == games.season.max()].abbr)
    update_rosters(data, active)
    update_injuries(data, teams[teams.season == games.season.max()])
    if not args.no_odds:
        update_odds(data, games, max(args.odds_start, args.start))
    print("done.")


if __name__ == "__main__":
    main()
