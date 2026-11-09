"""
Builds everything the app shows: features for every game (played and upcoming), weekly player
ratings, and walk-forward models. The build is cached in data/cache/ and redone only when the
data or the code changes.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import features as F
import model as M
import ratings as R
from paths import CACHE, CODE, DATA, current_season, today

TRAIN_START = pd.Timestamp("2016-11-01")    # first game the model trains on (ratings need a season to warm up)
ROTATION = 11                                # players per team counted in an upcoming game's lineup


def _fingerprint(data_dir):
    h = hashlib.sha1()
    for f in sorted(Path(data_dir).glob("*.csv*")) + sorted(CODE.glob("*.py")):
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{int(st.st_mtime)}".encode())
    h.update(today().isoformat().encode())
    return h.hexdigest()[:16]


def _status_off(s):
    s = str(s or "").lower()
    return any(k in s for k in ("out", "doubtful", "suspen", "inactive"))


def upcoming_lineups(games, box, last, data_dir, day):
    """Expected rotation for each team in each unplayed game from `day` on: who's available
    (roster minus the injury report's Out/Doubtful), each with projected minutes."""
    up = games[~games.played & (games.date >= pd.Timestamp(day))]
    if up.empty:
        return pd.DataFrame(columns=["game_id", "team", "player_id", "proj", "on", "status", "note"]), pd.DataFrame()
    names = box.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "pos"]]
    # rosters: ESPN's current rosters if we have them, else each player's most recent team
    rf, inf = Path(data_dir) / "rosters.csv", Path(data_dir) / "injuries.csv"
    if rf.exists():
        ro = pd.read_csv(rf)
        roster = ro[["team", "player_id"]].astype({"player_id": int})
    else:
        recent = last[last.last_date >= pd.Timestamp(day) - pd.Timedelta(days=120)]
        roster = recent.reset_index()[["team", "player_id"]]
    inj = pd.read_csv(inf) if inf.exists() else pd.DataFrame(columns=["player_id", "status", "detail", "comment"])
    inj = inj[inj.player_id.notna()].astype({"player_id": int}).drop_duplicates("player_id", keep="first").set_index("player_id")
    rows = []
    d0 = pd.Timestamp(day)
    for t, sub in roster.groupby("team"):
        pids = [p for p in sub.player_id if p in last.index]
        for p in pids:
            lr = last.loc[p]
            proj = float(lr.proj_next)
            gap = (d0 - lr.last_date).days
            if gap > 21:                       # hasn't played in 3+ weeks: assume limited minutes on return
                proj = min(proj, 18.0)
            st = inj.loc[p] if p in inj.index else None
            status = None if st is None else str(st.status)
            note = None if st is None else (st.get("detail") if isinstance(st.get("detail"), str) else None)
            rows.append({"team": t, "player_id": p, "proj": proj, "status": status, "note": note,
                         "on": not _status_off(status), "gap": gap})
    pool = pd.DataFrame(rows)
    # keep a realistic rotation: the top ROTATION available players by projected minutes
    pool["rank"] = pool[pool.on].groupby("team").proj.rank(ascending=False, method="first")
    pool["in_rot"] = pool.on & (pool["rank"] <= ROTATION)
    pool = pool.join(names, on="player_id")
    out = []
    for g in up.itertuples(index=False):
        for t in (g.home, g.away):
            s = pool[pool.team == t]
            out.append(s.assign(game_id=g.game_id))
    allp = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    used = allp[allp.in_rot][["game_id", "team", "player_id", "proj"]]
    return used, allp


class Engine:
    def __init__(self, data_dir=DATA):
        self.data_dir = Path(data_dir)
        self.ready = False

    def ensure(self, verbose=True):
        if self.ready:
            return self
        t0 = time.time()
        key = _fingerprint(self.data_dir)
        cf = CACHE / f"build_{key}.pkl"
        if cf.exists():
            try:
                with open(cf, "rb") as fh:
                    self.__dict__.update(pickle.load(fh))
                self.ready = True
                if verbose:
                    print(f"[engine] loaded from cache in {time.time() - t0:.1f}s")
                return self
            except Exception:  # noqa: BLE001
                pass
        games = D.load_games(self.data_dir)
        box = D.load_box(self.data_dir, games)
        tday = pd.Timestamp(today())
        fits = R.all_fits(games, box, start=pd.Timestamp("2016-10-01"), end=tday)
        proj, last = R.projected_minutes(box)
        used, pool = upcoming_lineups(games, box, last, self.data_dir, tday)
        df, players, _ = F.build(games, box, fits, upcoming=used)
        odds = D.load_odds(self.data_dir)
        df = df.merge(odds, on="game_id", how="left")
        st = {"df": df, "players": players, "pool": pool, "last": last, "fits": fits,
              "box": box, "season": int(current_season())}
        self.__dict__.update(st)
        self.ready = True
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            for old in CACHE.glob("build_*.pkl"):
                old.unlink()
            with open(cf, "wb") as fh:
                pickle.dump(st, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001
            pass
        if verbose:
            print(f"[engine] built in {time.time() - t0:.1f}s")
        return self

    @lru_cache(maxsize=256)
    def fit_before(self, day):
        d = self.df
        tr = d[d.played & (d.date < pd.Timestamp(day)) & (d.date >= TRAIN_START)]
        return M.MarginModel(M.STAT_FEATS).fit(tr)


ENGINE = Engine()
