"""
Builds everything the app shows: features for every game (played and upcoming), weekly player
ratings, and walk-forward models. The build is cached in data/cache/ and redone only when the
data or the code changes.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import features as F
import model as M
import ratings as R
import stints as ST
from paths import CACHE, CODE, DATA, current_season, today

sys.path.append(str(Path(__file__).resolve().parents[2] / "shared"))   # repo layout; a copy next to this file wins

TRAIN_START = pd.Timestamp("2016-11-01")    # first game the model trains on (ratings need a season to warm up)
ROTATION = 11                                # players per team counted in an upcoming game's lineup
STATUS_MIN_N = 150       # logged player-games a status needs before its play rate replaces the in/out rule
STATUS_PRIOR = {"out": 0.0, "doubtful": 0.1, "suspension": 0.0}   # shrinkage targets (others: 0.5)


def status_play_rates(data_dir, games, box):
    """How often players with each injury status actually played, from the injury-status log
    (shared/statuslog.py) and finished games. {status: (n, rate)}; empty until there is a log."""
    try:
        import statuslog
    except ImportError:
        return {}
    f = Path(data_dir) / "injury_log.csv"
    if not f.exists():
        return {}
    log = pd.read_csv(f)
    done = games[games.played & games.start_utc.notna()][["game_id", "start_utc", "home", "away"]]
    try:
        return statuslog.play_rates(log, done, box[["game_id", "player_id"]])
    except Exception:  # noqa: BLE001
        return {}


def status_weight(status, rates):
    """Chance a listed player plays: the logged rate once STATUS_MIN_N games are logged for his status
    (shrunk a little toward a prior); before that, None (use the Out/Doubtful rule)."""
    if not isinstance(status, str):
        return 1.0
    n, r = rates.get(status, (0, None))
    if n < STATUS_MIN_N:
        return None
    prior = next((v for k, v in STATUS_PRIOR.items() if k in status.lower()), 0.5)
    return (n * r + 20 * prior) / (n + 20)


def _fingerprint(data_dir):
    h = hashlib.sha1()
    for f in sorted(Path(data_dir).glob("*.csv*")) + sorted(CODE.glob("*.py")):
        st = f.stat()
        h.update(f"{f.name}{st.st_size}{int(st.st_mtime)}".encode())
    h.update(today().isoformat().encode())
    return h.hexdigest()[:16]


def _norm(n):
    import unicodedata
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in n if ch.isalnum())


def _status_off(s):
    s = str(s or "").lower()
    return any(k in s for k in ("out", "doubtful", "suspen", "inactive"))


def upcoming_lineups(games, box, last, data_dir, day, rates=None):
    """Expected rotation for each team in each unplayed game from `day` on: who's available
    (roster minus the injury report's Out/Doubtful), each with projected minutes."""
    up = games[~games.played & (games.date >= pd.Timestamp(day))]
    if up.empty:
        return pd.DataFrame(columns=["game_id", "team", "player_id", "proj", "on", "status", "note"]), pd.DataFrame()
    names = box.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "pos"]]
    # rosters: ESPN's current rosters if we have them, else each player's most recent team
    rf, inf = Path(data_dir) / "rosters.csv", Path(data_dir) / "injuries.csv"
    recent = last[last.last_date >= pd.Timestamp(day) - pd.Timedelta(days=120)].reset_index()[["team", "player_id"]]
    roster = recent
    if rf.exists():
        ro = pd.read_csv(rf)
        ro = ro[ro.player_id.notna()].astype({"player_id": int})
        ro["team"] = ro.team.astype(str).str.upper().map(lambda t: {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}.get(t, t))
        ok = set(ro.team) & set(games.home)
        # ESPN's rosters where we have them (they catch trades); each player's latest team otherwise
        roster = pd.concat([ro[ro.team.isin(ok)][["team", "player_id"]], recent[~recent.team.isin(ok)]], ignore_index=True)
    inj = pd.read_csv(inf) if inf.exists() else pd.DataFrame(columns=["player_id", "name", "status", "detail", "comment"])
    # match the injury report by ESPN player id, or by name when the id is missing
    by_name = {_norm(n): p for p, n in names.name.items()}
    inj["player_id"] = [p if pd.notna(p) else by_name.get(_norm(n)) for p, n in zip(inj.player_id, inj.get("name", [None] * len(inj)))]
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
            w = status_weight(status, rates or {})
            on = (not _status_off(status)) if w is None else (w >= 0.25)
            rows.append({"team": t, "player_id": p, "proj": proj, "status": status, "note": note,
                         "on": on, "p_play": 1.0 if (w is None or not on) else w, "gap": gap})
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
    # an available player who might sit (logged play rate < 1) counts for his expected minutes
    used = allp[allp.in_rot].assign(proj=lambda d: d.proj * d.p_play)[["game_id", "team", "player_id", "proj"]]
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
        st = ST.load(self.data_dir)
        rc_file = CACHE / "rapm_fits.pkl"      # past weeks' play-by-play fits never change: keep them
        try:
            rapm_cache = pickle.load(open(rc_file, "rb")) if rc_file.exists() else {}
        except Exception:  # noqa: BLE001
            rapm_cache = {}
        last_cut = max([c for c in R._cutoffs(games) if c <= tday], default=None)
        rapm_cache = {k: v for k, v in rapm_cache.items() if k[0] != last_cut}   # this week's refit uses new games
        fits = R.all_fits(games, box, start=pd.Timestamp("2016-10-01"), end=tday, stints=st if len(st) else None,
                          rapm_cache=rapm_cache)
        proj, last = R.projected_minutes(box)
        rates = status_play_rates(self.data_dir, games, box)
        used, pool = upcoming_lineups(games, box, last, self.data_dir, tday, rates)
        df, players, _ = F.build(games, box, fits, upcoming=used)
        odds = D.load_odds(self.data_dir)
        df = df.merge(odds, on="game_id", how="left")
        st = {"df": df, "players": players, "pool": pool, "last": last, "fits": fits, "status_rates": rates,
              "box": box, "season": int(current_season())}
        self.__dict__.update(st)
        self.ready = True
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            with open(rc_file, "wb") as fh:
                pickle.dump(rapm_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
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
