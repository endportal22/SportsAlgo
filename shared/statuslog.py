"""
Injury-status log: every change in a player's listed injury status, with the time it was seen.

Each data update compares ESPN's current injury report with the log and appends a row for every
player who is newly listed or whose status changed, and a "Cleared" row for every player who
dropped off the report. So for any game we can later look up exactly what the report said before
tip-off, and learn how often players with each status ("Day-To-Day", "Questionable", ...) actually
play (play_rates). The log lives in <league>/data/injury_log.csv; the app build also publishes an
encrypted copy so it survives losing the build cache.
"""
from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["seen_utc", "date_et", "team", "player_id", "name", "status", "detail"]


def _norm(n):
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in n if ch.isalnum())


def _key(team, pid, name):
    if pid is not None and not (isinstance(pid, float) and np.isnan(pid)):
        return f"id:{int(float(pid))}"
    return f"nm:{team}:{_norm(name)}"


def load(path, league=None):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    if league:
        try:
            import appdata
            js = appdata.fetch_published(league, "injury_log.json")
        except ImportError:
            js = None
        if js and js.get("rows"):
            print(f"[injuries] recovered {len(js['rows'])} logged statuses from the published app")
            return pd.DataFrame(js["rows"], columns=COLS)
    return pd.DataFrame(columns=COLS)


def update(path, current: pd.DataFrame, league=None, now=None):
    """current: today's report with team, name, status, detail and (if known) player_id."""
    log = load(path, league)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    stamp, day = now.isoformat(timespec="minutes"), now.tz_convert("America/New_York").strftime("%Y-%m-%d")
    last = {}
    for r in log.itertuples(index=False):
        last[_key(r.team, r.player_id, r.name)] = r
    new, seen = [], set()
    for r in current.itertuples(index=False):
        pid = getattr(r, "player_id", None)
        k = _key(r.team, pid, r.name)
        seen.add(k)
        status = str(r.status) if isinstance(r.status, str) else "Listed"
        detail = r.detail if isinstance(getattr(r, "detail", None), str) else None
        prev = last.get(k)
        if prev is None or prev.status != status or (prev.team != r.team):
            new.append([stamp, day, r.team, pid, r.name, status, detail])
    for k, prev in last.items():
        if k not in seen and prev.status != "Cleared":
            new.append([stamp, day, prev.team, prev.player_id, prev.name, "Cleared", None])
    if new:
        log = pd.concat([log, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    print(f"[injuries] status log: {len(new)} changes recorded ({len(log):,} rows in all)")
    return log


def status_before(log: pd.DataFrame, when):
    """Every listed player's status as of `when` (UTC): key -> row (Cleared players left out)."""
    when = pd.Timestamp(when)
    t = pd.to_datetime(log.seen_utc, utc=True)
    out = {}
    for r, ts in zip(log.itertuples(index=False), t):
        if ts < when:
            out[_key(r.team, r.player_id, r.name)] = r
    return {k: r for k, r in out.items() if r.status != "Cleared"}


def play_rates(log: pd.DataFrame, games: pd.DataFrame, played: pd.DataFrame):
    """How often players with each status played, from logged games.
    games: game_id, start_utc, home, away (finished games); played: game_id, player_id (minutes > 0).
    Returns {status: (n, rate)}."""
    if log.empty:
        return {}
    log = log.copy()
    log["ts"] = pd.to_datetime(log.seen_utc, utc=True)
    log = log.sort_values("ts")
    first = log.ts.min()
    g = games[pd.to_datetime(games.start_utc, utc=True) > first]
    if g.empty:
        return {}
    who = set(zip(played.game_id, played.player_id.astype(int)))
    res = {}
    for gm in g.itertuples(index=False):
        st = status_before(log, pd.Timestamp(gm.start_utc))
        for r in st.values():
            if r.team not in (gm.home, gm.away) or r.player_id is None or pd.isna(r.player_id):
                continue
            n, k = res.get(r.status, (0, 0))
            res[r.status] = (n + 1, k + int((gm.game_id, int(float(r.player_id))) in who))
    return {s: (n, k / n) for s, (n, k) in res.items() if n}
