"""
Player ratings, refit every week, plus each game's expected lineup and starting goalie.

Skaters: impact rating (every forward and defenseman)
-----------------------------------------------------
A regularised plus-minus on time-on-ice shares. For each team-game, the team's 5-on-5
expected goals per 60 minutes is modelled as

    xGF/60 = sum(own skaters' offense rating x TOI share) + sum(opposing skaters' defense rating x TOI share) + home

and solved with ridge regression over the last ~3 seasons (recent games weighted more).
Linemates and opponents are adjusted for automatically. A rating is xG per 60 minutes
on the ice compared with an average player: offense = chances created, defense =
chances prevented (positive is always good), impact = offense + defense.
Players who always share the ice (a set line or D pair) split credit, so read trends,
not decimals. Ratings are refit before every week since the first season downloaded,
which is what produces the trend lines.

Goalies: GSAx rating (the hockey version of the NFL algo's QB rating)
---------------------------------------------------------------------
Goals saved above expected per unblocked shot, recency-weighted and heavily shrunk
toward a slightly-below-average prior (goalie results are very noisy; about 2,000
shots of evidence count as much as the prior). Converted to goals per game by
multiplying by the shots a team faces in a typical game.

Lineups and injuries
--------------------
For each game and team, the expected lineup is everyone who played in the team's last
5 games, weighted by their usual TOI share, minus players who are out:
  * games already played: the players who actually dressed (lineups are known at
    warm-ups, before puck drop, so this is pre-game information);
  * upcoming games: the current NHL roster (players traded, waived or sent to the AHL
    are removed) and ESPN's injury list (Out / IR / suspended removed; day-to-day
    counts at 50%).
Injury cost is each missing regular's impact compared with the team's own depth
players, in goals per game.

Starting goalie
---------------
Played games use the actual starter. Upcoming games default to the team's No. 1
(most starts in its last 10 games) unless he's out, or the backup if it's the second
night of a back-to-back. Starters are confirmed on game day; switch goalies in the HUD.
"""
from __future__ import annotations

import time
import unicodedata
from bisect import bisect_left
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

import features as F
from paths import today

HALF_LIFE_DAYS = 365.0         # recency weighting for impact ratings
WINDOW_DAYS = 3 * 365
RIDGE_LAMBDA = 40.0            # shrinkage toward average (weights are in 48-minute games)
DTD_WEIGHT = 0.5               # day-to-day players count at 50%
RECENT_GAMES = 5               # TOI-share memory for expected lineups
GOALIE_PRIOR_SHOTS = 2000.0    # shrinkage for goalie GSAx/shot
GOALIE_PRIOR_SHIFT = -0.002    # unknown goalies are usually below average
GOALIE_HALF_LIFE = 40.0        # games
SKATER_PRIOR_HOURS = 8.0       # shrinkage for skater production rates


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())


# ------------------------------------------------------------------ impact ratings
class ImpactRatings:
    """Weekly-refit ridge plus-minus over TOI shares."""

    def __init__(self, skaters, tstats, games, lam=RIDGE_LAMBDA, half_life=HALF_LIFE_DAYS, window=WINDOW_DAYS):
        self.lam, self.half_life, self.window = lam, half_life, window
        gd = games.set_index("game_id")
        obs = tstats[tstats.game_id.isin(gd.index) & (tstats.toi_5v5 > 600)].copy()
        obs["opp"] = np.where(obs.team == obs.game_id.map(gd.home_team), obs.game_id.map(gd.away_team),
                              obs.game_id.map(gd.home_team))
        obs["gameday"] = obs.game_id.map(gd.gameday)
        obs["home"] = (obs.team == obs.game_id.map(gd.home_team)).astype(float) * (1 - obs.game_id.map(gd.neutral).astype(float))
        obs = obs.sort_values("gameday").reset_index(drop=True)
        obs["y"] = obs.ev_xgf * 3600.0 / obs.toi_5v5
        self.obs = obs

        sk = skaters[skaters.toi > 0][["game_id", "team", "player_id", "share"]]
        ids = pd.Index(sorted(sk.player_id.unique()))
        self.ids = ids
        n = len(ids)
        self.n = n
        key = pd.Series(np.arange(len(obs)), index=pd.MultiIndex.from_arrays([obs.game_id, obs.team]))
        okey = pd.Series(np.arange(len(obs)), index=pd.MultiIndex.from_arrays([obs.game_id, obs.opp]))
        mi = pd.MultiIndex.from_arrays([sk.game_id, sk.team])
        r1 = key.reindex(mi).values
        r2 = okey.reindex(mi).values
        m1, m2 = ~np.isnan(r1), ~np.isnan(r2)
        pidx = ids.get_indexer(sk.player_id)
        rows = np.concatenate([r1[m1], r2[m2], np.arange(len(obs))]).astype(int)
        cols = np.concatenate([pidx[m1], n + pidx[m2], np.full(len(obs), 2 * n)])
        vals = np.concatenate([sk.share.values[m1], sk.share.values[m2], obs.home.values])
        self.X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(obs), 2 * n + 1))
        self.y = obs.y.values
        self.w_min = (obs.toi_5v5.values / 60.0) / F.EV_MIN
        self.days = obs.gameday.values.astype("datetime64[D]").astype(np.int64)
        self._x0 = None

    def fit(self, cutoff):
        """Ratings using only team-games strictly before `cutoff` (Timestamp)."""
        c = np.datetime64(pd.Timestamp(cutoff).date(), "D").astype(np.int64)
        idx = np.flatnonzero((self.days < c) & (self.days >= c - self.window))
        if len(idx) < 200:
            return None
        age = c - self.days[idx]
        rec = 0.5 ** (age / self.half_life)
        w = self.w_min[idx] * rec
        y = self.y[idx]
        mu = np.average(y, weights=w)
        Xs = self.X[idx]
        A = (Xs.T @ sparse.diags(w) @ Xs).tocsr() + sparse.identity(Xs.shape[1]) * self.lam
        A = A.tolil()
        A[2 * self.n, 2 * self.n] -= self.lam * 0.99   # barely shrink the home term
        A = A.tocsr()
        b = Xs.T @ (w * (y - mu))
        sol, _ = cg(A, b, x0=self._x0, maxiter=400)
        self._x0 = sol
        expo = np.asarray((sparse.diags(rec) @ Xs).sum(axis=0)).ravel()   # recency-weighted full-game equivalents
        n = self.n
        return {"mu": mu, "off": sol[:n], "def": sol[n:2 * n], "home": sol[2 * n],
                "off_expo": expo[:n], "def_expo": expo[n:2 * n]}


# ------------------------------------------------------------------ goalie + skater production
def goalie_games(shots, goalies, games):
    """Per goalie per game: unblocked shots faced, xG faced, goals allowed (goalie in net only)."""
    s = shots[shots.ev.isin(["G", "S", "M"]) & (shots.opp_g == 1) & shots.goalie.notna()]
    s = s.assign(is_goal=(s.ev == "G").astype(float))
    g = s.groupby(["game_id", "goalie"]).agg(shots=("xg", "size"), xga=("xg", "sum"), ga=("is_goal", "sum")).reset_index()
    g = g.rename(columns={"goalie": "player_id"})
    g["player_id"] = g.player_id.astype("int64")
    gl = goalies[["game_id", "team", "player_id", "starter", "toi"]].copy()
    out = gl.merge(g, on=["game_id", "player_id"], how="left").fillna({"shots": 0, "xga": 0.0, "ga": 0.0})
    gd = games.set_index("game_id")
    out["gameday"] = out.game_id.map(gd.gameday)
    out["season"] = out.game_id.map(gd.season)
    return out.dropna(subset=["gameday"])


def roll_goalies(gg, half_life_games=GOALIE_HALF_LIFE, prior_n=GOALIE_PRIOR_SHOTS, prior_shift=GOALIE_PRIOR_SHIFT):
    """Pre-game GSAx-per-shot rating for each goalie-game, plus one 'latest' row per goalie."""
    lam = 0.5 ** (1 / half_life_games)
    rows = []
    for pid, t in gg.sort_values("gameday").groupby("player_id", sort=False):
        s_g = s_n = 0.0
        for r in t.itertuples(index=False):
            rows.append((pid, r.game_id, r.gameday, r.season, r.team, (s_g + prior_n * prior_shift) / (s_n + prior_n), s_n))
            if r.shots > 0:
                s_g = s_g * lam + (r.xga - r.ga)
                s_n = s_n * lam + r.shots
        rows.append((pid, None, t.gameday.max() + pd.Timedelta(days=1), None, t.team.iloc[-1],
                     (s_g + prior_n * prior_shift) / (s_n + prior_n), s_n))
    return pd.DataFrame(rows, columns=["pid", "game_id", "gameday", "season", "team", "rating", "volume"])


def roll_skaters(skaters, shots, games, seasons_back=3, half_life_games=30.0, prior_hours=SKATER_PRIOR_HOURS):
    """Recency-weighted, shrunk individual xG/60 and points/60 per skater (for the Players tab)."""
    last = int(games.season.max())
    sk = skaters[skaters.season >= last - seasons_back + 1]
    ix = shots[shots.ev.isin(["G", "S", "M"]) & (shots.opp_g == 1)].groupby(["game_id", "shooter"]).xg.sum()
    sk = sk.assign(ixg=[ix.get((g, p), 0.0) for g, p in zip(sk.game_id, sk.player_id)],
                   pts=sk.goals + sk.assists, hours=sk.toi / 3600.0)
    gd = games.set_index("game_id").gameday
    sk = sk.assign(gameday=sk.game_id.map(gd)).dropna(subset=["gameday"]).sort_values("gameday")
    lg_ixg = sk.ixg.sum() / max(sk.hours.sum(), 1e-9)
    lg_pts = sk.pts.sum() / max(sk.hours.sum(), 1e-9)
    lam = 0.5 ** (1 / half_life_games)
    rows = []
    for pid, t in sk.groupby("player_id", sort=False):
        a = b = h = 0.0
        for r in t.itertuples(index=False):
            a, b, h = a * lam + r.ixg, b * lam + r.pts, h * lam + r.hours
            rows.append((pid, r.game_id, r.gameday, r.season, (a + prior_hours * lg_ixg) / (h + prior_hours) - lg_ixg,
                         (b + prior_hours * lg_pts) / (h + prior_hours) - lg_pts, h))
    return pd.DataFrame(rows, columns=["pid", "game_id", "gameday", "season", "ixg60", "pts60", "hours"])


# ------------------------------------------------------------------ availability for upcoming games
def _load_current(data_dir):
    d = Path(data_dir)
    ros = pd.read_csv(d / "rosters.csv") if (d / "rosters.csv").exists() else None
    inj = pd.read_csv(d / "injuries.csv") if (d / "injuries.csv").exists() else None
    if ros is not None:
        ros["team"] = ros["team"].replace(F.TEAM_FIX)
    if inj is not None:
        inj["team"] = inj["team"].replace(F.TEAM_FIX)
    return ros, inj


def _injury_lookup(inj):
    out = {}
    if inj is None or inj.empty:
        return out
    for r in inj.itertuples(index=False):
        out[(r.team, _norm(r.name))] = r
    return out


def availability(pid, team, name, gameday, ros_by_team, inj_map, roster_ok=True):
    """(weight, reason) for an upcoming game."""
    if roster_ok and ros_by_team is not None and team in ros_by_team and pid not in ros_by_team[team]:
        return 0.0, "not on roster"
    r = inj_map.get((team, _norm(name)))
    if r is None:
        return 1.0, None
    st = str(getattr(r, "status", "") or "")
    what = getattr(r, "injury", None)
    what = what if isinstance(what, str) and what else None
    tag = f" ({what.lower()})" if what else ""
    if "day" in st.lower():
        return DTD_WEIGHT, "day-to-day" + tag
    rd = getattr(r, "return_date", None)
    if isinstance(rd, str) and rd:
        try:
            if pd.Timestamp(rd[:10]) <= gameday:
                return DTD_WEIGHT, "may return" + tag
        except ValueError:
            pass
    if "susp" in st.lower():
        return 0.0, "suspended"
    return 0.0, (st.lower() or "out") + tag


# ------------------------------------------------------------------ the whole pipeline
def build(data_dir, games, skaters, goalies, shots, tstats, verbose=True):
    """Weekly impact ratings, goalie ratings, per-game lineup values and projected goalies."""
    t0 = time.time()
    gd = games.set_index("game_id")
    skaters = skaters[skaters.game_id.isin(gd.index)].copy()
    goalies = goalies[goalies.game_id.isin(gd.index)].copy()
    tot = skaters.groupby(["game_id", "team"]).toi.transform("sum")
    skaters["share"] = np.where(tot > 0, skaters.toi * 5.0 / tot, 0.0)
    skaters["gameday"] = skaters.game_id.map(gd.gameday)

    # player directory: full names where we have them
    pdir = Path(data_dir) / "players.csv"
    full = pd.read_csv(pdir) if pdir.exists() else pd.DataFrame(columns=["player_id", "name", "pos"])
    ros, inj = _load_current(data_dir)
    ap = pd.concat([skaters[["player_id", "name", "pos", "team", "gameday"]],
                    goalies.assign(pos="G", gameday=goalies.game_id.map(gd.gameday))[["player_id", "name", "pos", "team", "gameday"]]])
    info = ap.sort_values("gameday").groupby("player_id").agg(name=("name", "last"), pos=("pos", "last"), team=("team", "last"))
    fn = full.dropna(subset=["name"]).drop_duplicates("player_id", keep="last").set_index("player_id")
    info["name"] = [fn["name"].get(p, n) if isinstance(fn["name"].get(p, None), str) and fn["name"].get(p) else n
                    for p, n in zip(info.index, info["name"])]
    if "pos" in fn:
        info["pos"] = [fn.pos.get(p, q) if isinstance(fn.pos.get(p, None), str) else q for p, q in zip(info.index, info.pos)]
    if ros is not None:
        rr = ros.drop_duplicates("player_id", keep="last").set_index("player_id")
        info["team"] = [rr.team.get(p, t) for p, t in zip(info.index, info.team)]
        extra = rr[~rr.index.isin(info.index)][["name", "pos", "team"]]
        info = pd.concat([info, extra])
    info.index.name = "player_id"

    IR = ImpactRatings(skaters, tstats, games)
    if verbose:
        print(f"[players] {len(IR.ids):,} skaters, {len(IR.obs):,} team-games ({time.time() - t0:.1f}s)", flush=True)

    # weekly checkpoints: the Monday of every week with games, plus today for upcoming games
    tday = pd.Timestamp(today())
    days = pd.Series(sorted(games.gameday.unique()))
    mondays = sorted(set((days - pd.to_timedelta(days.dt.weekday, unit="D")).tolist()))
    mondays = [m for m in mondays if m <= tday]
    cps = mondays + ([tday] if tday not in mondays else [])
    fits = {}
    hist = []
    for cp in cps:
        f = IR.fit(cp)
        fits[cp] = f
        if f is None:
            continue
        keep = (f["off_expo"] > 1.0) | (f["def_expo"] > 1.0)
        hist.append(pd.DataFrame({"cp": cp, "pid": IR.ids[keep], "off": np.round(f["off"][keep], 4),
                                  "def": np.round(-f["def"][keep], 4), "expo": np.round(f["off_expo"][keep], 2)}))
    history = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame(columns=["cp", "pid", "off", "def", "expo"])
    history["impact"] = history["off"] + history["def"]
    if verbose:
        print(f"[players] {len([v for v in fits.values() if v is not None])} weekly refits ({time.time() - t0:.1f}s)", flush=True)

    def fit_for(day):
        if day >= tday:
            return fits.get(cps[-1])
        m = day - pd.Timedelta(days=day.weekday())
        return fits.get(m)

    # ---- lineups
    team_games = {}
    for (team, gid), t in skaters.sort_values("gameday").groupby(["team", "game_id"], sort=False):
        team_games.setdefault(team, []).append((t.gameday.iloc[0], gid, dict(zip(t.player_id, t.share))))
    for team in team_games:
        team_games[team].sort(key=lambda x: x[0])
    team_days = {t: [x[0] for x in v] for t, v in team_games.items()}
    game_lineup = {(x[1], t): x[2] for t, v in team_games.items() for x in v}
    ros_by_team = {t: set(x.player_id) for t, x in ros.groupby("team")} if ros is not None else None
    inj_map = _injury_lookup(inj)
    pid_loc = {p: i for i, p in enumerate(IR.ids)}
    ev_scale = F.EV_MIN / 60.0

    rows, detail = [], {}
    for g in games.itertuples(index=False):
        f = fit_for(g.gameday)
        if f is None:
            continue
        lo, ld = f["off_expo"], f["def_expo"]
        sel = (lo > 1) & (lo < 8)
        r_off = np.average(f["off"][sel], weights=lo[sel]) if sel.any() else 0.0
        r_def = np.average(f["def"][sel], weights=ld[sel]) if sel.any() else 0.0
        upcoming = not g.played
        for team, side in ((g.home_team, "home"), (g.away_team, "away")):
            tg = team_games.get(team, [])
            k = bisect_left(team_days.get(team, []), g.gameday)
            prior = tg[max(0, k - RECENT_GAMES):k]
            if not prior:
                continue
            seen = {}
            for _, _, pl in prior:
                for pid, sh in pl.items():
                    seen.setdefault(pid, []).append(sh)
            n_prior = len(prior)
            e = {p: float(np.mean(v)) * len(v) / n_prior for p, v in seen.items()}
            app = {p: len(v) / n_prior for p, v in seen.items()}
            if not upcoming:
                this = game_lineup.get((g.game_id, team))
                if this is None:
                    continue
                avail = {p: (1.0, None) if p in this else (0.0, "did not play") for p in e}
            else:
                avail = {p: availability(p, team, info["name"].get(p), g.gameday, ros_by_team, inj_map) for p in e}
            # team's own depth (extras / call-ups) is the replacement for a missing regular
            depth = [(e[p], p) for p in e if app[p] < 0.6 and p in pid_loc]
            wsum = sum(s for s, _ in depth)
            t_off = sum(s * f["off"][pid_loc[p]] for s, p in depth) / wsum if wsum > 0.3 else r_off
            t_def = sum(s * f["def"][pid_loc[p]] for s, p in depth) / wsum if wsum > 0.3 else r_def
            res = {"game_id": g.game_id, "team": team, "side": side}
            off_val = def_val = off_cost = def_cost = 0.0
            det = []
            for p, share in e.items():
                j = pid_loc.get(p)
                bo, bd = (f["off"][j], f["def"][j]) if j is not None else (r_off, r_def)
                a, why = avail[p]
                off_val += share * (a * bo + (1 - a) * t_off)
                def_val += share * (a * bd + (1 - a) * t_def)
                off_cost += share * (1 - a) * (bo - t_off)
                def_cost += share * (1 - a) * (bd - t_def)
                if upcoming or g.season >= int(games.season.max()):
                    det.append({"pid": p, "share": round(share, 3), "avail": a, "why": why,
                                "off": round(bo, 3), "def": round(-bd, 3), "app": round(app[p], 2)})
            res["off_val"] = off_val * ev_scale
            res["def_val"] = def_val * ev_scale
            res["injury_rel"] = (off_cost - def_cost) * ev_scale   # goals/game lost vs own depth (positive hurts)
            rows.append(res)
            if det:
                detail[(g.game_id, team)] = det
    lineups = pd.DataFrame(rows)
    if verbose:
        print(f"[players] lineups done ({time.time() - t0:.1f}s)", flush=True)

    # ---- goalies
    gg = goalie_games(shots, goalies, games)
    gprod = roll_goalies(gg)
    fen = gg[gg.starter].shots.mean() if len(gg) else 40.0
    fen = float(fen) if np.isfinite(fen) and fen > 10 else 40.0
    starters = gg[gg.starter].sort_values("gameday")
    team_starts = {t: list(zip(x.gameday, x.game_id, x.player_id)) for t, x in starters.groupby("team")}
    g_last = gprod[gprod.game_id.isna()].set_index("pid")

    goalie_proj = {}
    cur_season = int(games.season.max())
    team_sched = {}
    for g in games.itertuples(index=False):
        team_sched.setdefault(g.home_team, []).append(g.gameday)
        team_sched.setdefault(g.away_team, []).append(g.gameday)
    for g in games[~games.played].itertuples(index=False):
        for team in (g.home_team, g.away_team):
            hist_ = [x for x in team_starts.get(team, []) if x[0] < g.gameday]
            last10 = [p for _, _, p in hist_[-10:]]
            season_starts = {}
            for d_, gid, p in hist_:
                if gd.season.get(gid) == g.season:
                    season_starts[p] = season_starts.get(p, 0) + 1
            cands = []
            for p in dict.fromkeys(list(reversed(last10)) + [p for _, _, p in reversed(hist_[-40:])]):
                cands.append(p)
            if ros is not None:
                for p in ros[(ros.team == team) & (ros.pos == "G")].player_id:
                    if p not in cands:
                        cands.append(p)
            opts = []
            for p in cands:
                ok_w, why = availability(p, team, info["name"].get(p), g.gameday, ros_by_team, inj_map)
                opts.append({"pid": int(p), "name": info["name"].get(p), "available": ok_w > 0, "status": why,
                             "starts10": last10.count(p), "season_starts": season_starts.get(p, 0),
                             "rating": float(g_last.rating.get(p, GOALIE_PRIOR_SHIFT)), "note": None})
            opts = [o for o in opts if o["available"] or o["season_starts"]]
            avail_ = [o for o in opts if o["available"]]
            if not avail_:
                continue
            avail_.sort(key=lambda o: (-o["starts10"], -o["season_starts"]))
            no1 = avail_[0]
            prev_day = max([d_ for d_ in team_sched.get(team, []) if d_ < g.gameday], default=None)
            b2b = prev_day is not None and (g.gameday - prev_day).days == 1
            proj = no1
            if b2b and len(avail_) > 1:
                proj = avail_[1]
                proj["note"] = "back-to-back: backup projected"
            goalie_proj[(g.game_id, team)] = {**proj, "options": opts[:5], "b2b": b2b}
    if verbose:
        print(f"[players] goalies done ({time.time() - t0:.1f}s)", flush=True)

    sprod = roll_skaters(skaters, shots, games)
    lat = fits.get(cps[-1]) if cps else None
    if lat is None:
        lat = next((fits[c] for c in reversed(cps) if fits.get(c) is not None), None)
    if verbose:
        print(f"[players] done in {time.time() - t0:.1f}s", flush=True)
    return {"lineups": lineups, "lineup_detail": detail, "history": history, "goalie_prod": gprod,
            "goalie_games": gg, "skater_prod": sprod, "info": info, "goalie_proj": goalie_proj,
            "fen_per_game": fen, "latest_fit": lat, "ids": IR.ids, "skaters": skaters,
            "cur_season": cur_season}


def game_features(pl, df):
    """Attach lineup + goalie features to the per-game feature table from features.py."""
    lu = pl["lineups"]
    df = df.copy()
    if len(lu):
        h = lu[lu.side == "home"].set_index("game_id")
        a = lu[lu.side == "away"].set_index("game_id")
        for pre, t in (("h_", h), ("a_", a)):
            for c in ("off_val", "def_val", "injury_rel"):
                df[pre + c] = df.game_id.map(t[c])
    else:
        for pre in ("h_", "a_"):
            for c in ("off_val", "def_val", "injury_rel"):
                df[pre + c] = np.nan
    df["d_lineup"] = (df.h_off_val - df.a_def_val) - (df.a_off_val - df.h_def_val)
    df["d_injury_rel"] = df.h_injury_rel - df.a_injury_rel

    gg = pl["goalie_games"]
    st = gg[gg.starter]
    starter = {(g, t): p for g, t, p in zip(st.game_id, st.team, st.player_id)}
    gp = pl["goalie_prod"]
    by_game = {(p, g): r for p, g, r in zip(gp.pid, gp.game_id, gp.rating) if g is not None and not (isinstance(g, float) and np.isnan(g))}
    info = pl["info"]
    fen = pl["fen_per_game"]
    for side, tcol, pre in (("home", "home_team", "h_"), ("away", "away_team", "a_")):
        ids, names, rats, srcs = [], [], [], []
        for gid, team, played in zip(df.game_id, df[tcol], df.played):
            if played:
                p = starter.get((gid, team))
                r = by_game.get((p, gid), GOALIE_PRIOR_SHIFT) if p is not None else np.nan
                src = "actual"
            else:
                pj = pl["goalie_proj"].get((gid, team))
                p = pj["pid"] if pj else None
                r = pj["rating"] if pj else np.nan
                src = "projected" if pj else "unknown"
            ids.append(p)
            names.append(info["name"].get(p) if p is not None else None)
            rats.append(r)
            srcs.append(src)
        df[f"{side}_goalie_id"] = ids
        df[f"{side}_goalie"] = names
        df[f"{pre}goalie_rating"] = rats
        df[f"{side}_goalie_source"] = srcs
    df["d_goalie"] = (df.h_goalie_rating - df.a_goalie_rating) * fen
    return df
