"""
Player ratings, refit every week, plus each game's expected lineup strength.

Two kinds of rating
-------------------
1. Impact rating (every player, every position)
   A regularised plus-minus. For each team-game, the offense's EPA/play is modelled as

       EPA/play = sum(offensive players' rating x snap share) + sum(defenders' rating x snap share) + home

   and solved with ridge regression over the last ~3 seasons (recent games weighted
   more). Teammates and opponents are adjusted for automatically. Snap counts exist
   for every player since 2016, so linemen and defensive backs get ratings too, but
   players who always share the field (an offensive line) can't be separated well,
   so their ratings drift together. Ratings are converted to points per game at full
   snaps (x ~62 plays). Defense is sign-flipped so positive always means good.

2. Production ratings (QB, RB, WR, TE)
   Shrunk, recency-weighted EPA per dropback / per carry / per target from
   play-by-play. QB EPA/dropback is the single most predictive player number.

Both are refit before every week from 2016 on, which is what produces the trend lines.

Lineups
-------
For each game and team, the expected lineup is everyone who played in recent games,
weighted by their recent snap share, minus players who are out:
  * games already played: the players who actually suited up (inactives are announced
    90 minutes before kickoff, so this is pre-game information);
  * upcoming games: the weekly roster (active players only) minus anyone listed
    Out or Doubtful on the injury report; Questionable counts at 75%.
Lineup strength is the snap-weighted sum of impact ratings, and the injury cost is
the gap between that and the same team at full strength.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

import features as F

PLAYS_PER_GAME = 62.0          # converts EPA/play to points-per-game scale
HALF_LIFE_DAYS = 400.0         # recency weighting for impact ratings
WINDOW_DAYS = 3 * 365
RIDGE_LAMBDA = 2500.0          # shrinkage toward average (tuned in tune_players.py)
QUESTIONABLE_WEIGHT = 0.75
RECENT_GAMES = 4               # snap-share memory for expected lineups


# ------------------------------------------------------------------ loading
def _fix(s):
    return s.replace(F.TEAM_FIX)


def load_snaps(data_dir, seasons):
    fr = [pd.read_csv(Path(data_dir) / f"snaps_{s}.csv.gz") for s in seasons if (Path(data_dir) / f"snaps_{s}.csv.gz").exists()]
    s = pd.concat(fr, ignore_index=True)
    s["team"] = _fix(s["team"])
    s["opponent"] = _fix(s["opponent"])
    s = s[s["pfr_player_id"].notna()]
    s = s.drop_duplicates(["game_id", "pfr_player_id"])
    return s


def load_players(data_dir):
    p = pd.read_csv(Path(data_dir) / "players.csv.gz", low_memory=False)
    return p


def team_game_obs(pbp, p: F.Params = F.Params()):
    """One row per (game, offense): EPA/play overall, on passes, on runs."""
    d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    d = d[(d["qb_kneel"].fillna(0) == 0) & (d["qb_spike"].fillna(0) == 0)]
    d = d[d["wp"].between(p.wp_floor, 1 - p.wp_floor) | d["wp"].isna()]
    d["posteam"] = _fix(d["posteam"]); d["defteam"] = _fix(d["defteam"])
    d["is_pass"] = (d["play_type"] == "pass").astype(float)
    d["pass_epa"] = np.where(d.is_pass == 1, d.epa, np.nan)
    d["run_epa"] = np.where(d.is_pass == 0, d.epa, np.nan)
    g = d.groupby(["game_id", "posteam", "defteam"]).agg(
        epa=("epa", "mean"), n=("epa", "size"), pass_rate=("is_pass", "mean"),
        pass_epa=("pass_epa", "mean"), run_epa=("run_epa", "mean"),
        n_pass=("pass_epa", "count"), n_run=("run_epa", "count")).reset_index()
    return g.rename(columns={"posteam": "team", "defteam": "opp"})


# ------------------------------------------------------------------ impact ratings
class ImpactRatings:
    """Weekly-refit ridge plus-minus over snap shares."""

    def __init__(self, snaps, obs, games, lam=RIDGE_LAMBDA, half_life=HALF_LIFE_DAYS, window=WINDOW_DAYS):
        self.lam, self.half_life, self.window = lam, half_life, window
        gd = games.set_index("game_id")
        obs = obs[obs.game_id.isin(gd.index)].copy()
        obs["gameday"] = obs.game_id.map(gd.gameday)
        obs["home"] = (obs.team == obs.game_id.map(gd.home_team)).astype(float)
        obs = obs.sort_values("gameday").reset_index(drop=True)
        self.obs = obs

        ids = pd.Index(sorted(snaps.pfr_player_id.unique()))
        self.ids = ids
        n = len(ids)
        self.n = n
        # columns: [0, n) offense rating, [n, 2n) defense rating, 2n = home
        off = snaps[snaps.offense_pct > 0][["game_id", "team", "pfr_player_id", "offense_pct"]]
        dfn = snaps[snaps.defense_pct > 0][["game_id", "team", "pfr_player_id", "defense_pct"]]
        key = pd.Series(np.arange(len(obs)), index=pd.MultiIndex.from_arrays([obs.game_id, obs.team]))
        okey = pd.MultiIndex.from_arrays([obs.game_id, obs.opp])
        okey = pd.Series(np.arange(len(obs)), index=okey)
        r1 = key.reindex(pd.MultiIndex.from_arrays([off.game_id, off.team])).values
        m1 = ~np.isnan(r1)
        r2 = okey.reindex(pd.MultiIndex.from_arrays([dfn.game_id, dfn.team])).values
        m2 = ~np.isnan(r2)
        rows = np.concatenate([r1[m1], r2[m2], np.arange(len(obs))]).astype(int)
        cols = np.concatenate([ids.get_indexer(off.pfr_player_id[m1]), n + ids.get_indexer(dfn.pfr_player_id[m2]),
                               np.full(len(obs), 2 * n)])
        vals = np.concatenate([off.offense_pct.values[m1], dfn.defense_pct.values[m2], obs.home.values])
        self.X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(obs), 2 * n + 1))
        self.y = obs.epa.values
        self.w_plays = obs.n.values.astype(float)
        self.days = obs.gameday.values.astype("datetime64[D]").astype(np.int64)
        has = np.asarray(self.X[:, :2 * n].sum(axis=1)).ravel() > 5
        self.ok = has  # team-games with snap data

    def fit(self, cutoff):
        """Ratings using only team-games strictly before `cutoff` (Timestamp)."""
        c = np.datetime64(pd.Timestamp(cutoff).date(), "D").astype(np.int64)
        sel = (self.days < c) & (self.days >= c - self.window) & self.ok
        idx = np.flatnonzero(sel)
        if len(idx) < 50:
            return None
        age = c - self.days[idx]
        w = self.w_plays[idx] * 0.5 ** (age / self.half_life)
        y = self.y[idx]
        mu = np.average(y, weights=w)
        sw = np.sqrt(w)
        A = sparse.diags(sw) @ self.X[idx]
        b = sw * (y - mu)
        sol = lsqr(A, b, damp=np.sqrt(self.lam), atol=1e-8, btol=1e-8, iter_lim=3000)[0]
        # exposure: recency-weighted full-game equivalents
        expo = np.asarray((sparse.diags(0.5 ** (age / self.half_life)) @ self.X[idx]).sum(axis=0)).ravel()
        n = self.n
        return {"mu": mu, "off": sol[:n], "def": sol[n:2 * n], "home": sol[2 * n],
                "off_expo": expo[:n], "def_expo": expo[n:2 * n]}


# ------------------------------------------------------------------ production ratings
def production_ratings(pbp, games, prior_n=None):
    """Weekly shrunk EPA rates for QBs (per dropback), rushers (per carry), receivers (per target).

    Returns a long table: one row per (player gsis id, kind, season, week) with the rating
    *before* that week's games, plus the cumulative recency-weighted volume.
    """
    prior_n = prior_n or {"qb": 150.0, "rush": 80.0, "rec": 50.0}
    d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    d = d[(d["qb_kneel"].fillna(0) == 0) & (d["qb_spike"].fillna(0) == 0)]
    gd = games.set_index("game_id").gameday
    d["gameday"] = d.game_id.map(gd)
    d = d[d.gameday.notna()]
    qb = d[d.qb_dropback == 1].copy()
    qb["pid"] = qb.passer_player_id.fillna(qb.rusher_player_id)
    qb["name"] = qb.passer_player_name.fillna(qb.rusher_player_name)
    ru = d[(d.play_type == "run") & (d.qb_scramble.fillna(0) == 0)].rename(columns={"rusher_player_id": "pid", "rusher_player_name": "name"})
    rc = d[(d.play_type == "pass") & d.receiver_player_id.notna()].rename(columns={"receiver_player_id": "pid", "receiver_player_name": "name"})
    out = []
    for kind, t in (("qb", qb), ("rush", ru), ("rec", rc)):
        t = t[t.pid.notna()]
        g = t.groupby(["pid", "game_id", "gameday", "season", "week"]).agg(epa=("epa", "sum"), n=("epa", "size"), name=("name", "last")).reset_index()
        g["kind"] = kind
        out.append(g)
    return pd.concat(out, ignore_index=True)


def roll_production(pg, games, half_life_games=16.0, prior_n=None, prior_shift=None):
    """Turn per-game production into pre-game ratings for each (player, kind, game).

    rating = (sum decayed EPA + prior_n * prior) / (sum decayed n + prior_n), measured relative
    to the league rate that season so far. Prior sits slightly below average (unknown
    players are usually backups).
    """
    prior_n = prior_n or {"qb": 150.0, "rush": 80.0, "rec": 50.0}
    prior_shift = prior_shift or {"qb": -0.08, "rush": -0.03, "rec": -0.05}
    lam = 0.5 ** (1 / half_life_games)
    rows = []
    league = pg.groupby("kind").apply(lambda t: t.epa.sum() / t.n.sum())
    for (pid, kind), t in pg.sort_values("gameday").groupby(["pid", "kind"], sort=False):
        s_epa = s_n = 0.0
        base = league[kind]
        pn, ps = prior_n[kind], prior_shift[kind]
        for r in t.itertuples(index=False):
            rate = (s_epa + pn * (base + ps)) / (s_n + pn)
            rows.append((pid, kind, r.game_id, r.gameday, r.season, r.week, r.name, rate - base, s_n))
            s_epa = s_epa * lam + r.epa
            s_n = s_n * lam + r.n
        # rating after the last game (for upcoming games)
        rows.append((pid, kind, None, t.gameday.max() + pd.Timedelta(days=1), None, None, t.name.iloc[-1],
                     (s_epa + pn * (base + ps)) / (s_n + pn) - base, s_n))
    return pd.DataFrame(rows, columns=["pid", "kind", "game_id", "gameday", "season", "week", "name", "rating", "volume"])


# ------------------------------------------------------------------ the whole pipeline
def _id_maps(data_dir, season):
    """pfr id <-> gsis id, from the player directory and current rosters."""
    p = load_players(data_dir)
    m = p.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    rp = Path(data_dir) / f"rosters_{season}.csv.gz"
    if rp.exists():
        r = pd.read_csv(rp, low_memory=False).dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
        m = pd.concat([r.set_index("pfr_id").gsis_id, m])
        m = m[~m.index.duplicated()]
    return m


def availability_for_upcoming(data_dir, season, week, team_players):
    """{(team, pfr_id): (weight, reason)} for an upcoming week, from rosters + injury report."""
    out = {}
    pf2gs = _id_maps(data_dir, season)
    rp = Path(data_dir) / f"rosters_{season}.csv.gz"
    ros = None
    if rp.exists():
        r = pd.read_csv(rp, low_memory=False)
        r = r[(r.week == week)]
        if not r.empty:
            r["team"] = _fix(r["team"])
            ros = {(t, g): (s, d) for t, g, s, d in zip(r.team, r.gsis_id, r.status, r.status_description_abbr)}
    ip = Path(data_dir) / f"injuries_{season}.csv.gz"
    inj = {}
    if ip.exists():
        i = pd.read_csv(ip, low_memory=False)
        i = i[(i.week == week) & i.report_status.notna()]
        i["team"] = _fix(i["team"])
        inj = {(t, g): (s, pi) for t, g, s, pi in zip(i.team, i.gsis_id, i.report_status, i.report_primary_injury)}
    for team, pids in team_players.items():
        for pid in pids:
            gs = pf2gs.get(pid)
            w, why = 1.0, None
            if ros is not None and gs is not None:
                st = ros.get((team, gs))
                if st is None:
                    w, why = 0.0, "not on roster"
                elif st[0] != "ACT":
                    w, why = 0.0, {"RES": "injured reserve", "DEV": "practice squad", "INA": "inactive",
                                   "CUT": "released", "RET": "retired"}.get(st[0], st[0].lower())
            if w > 0 and gs is not None and (team, gs) in inj:
                s, part = inj[(team, gs)]
                part = part if isinstance(part, str) else ""
                if s in ("Out", "Doubtful"):
                    w, why = 0.0, f"{s.lower()}" + (f" ({part.lower()})" if part else "")
                elif s == "Questionable":
                    w, why = QUESTIONABLE_WEIGHT, "questionable" + (f" ({part.lower()})" if part else "")
            out[(team, pid)] = (w, why)
    return out


def _roster_qbs(data_dir, season):
    rp = Path(data_dir) / f"rosters_{season}.csv.gz"
    out = {}
    if rp.exists():
        r = pd.read_csv(rp, low_memory=False)
        r = r[r.position == "QB"]
        r["team"] = _fix(r["team"])
        for t, w, g, st in zip(r.team, r.week, r.gsis_id, r.status):
            out.setdefault((t, w), {})[g] = st
    return out


def _roster_pfr(data_dir, season):
    """{(team, week): [pfr ids of ACTIVE players]} for the current season."""
    rp = Path(data_dir) / f"rosters_{season}.csv.gz"
    out = {}
    if rp.exists():
        r = pd.read_csv(rp, low_memory=False)
        r = r[(r.status == "ACT") & r.pfr_id.notna()]
        r["team"] = _fix(r["team"])
        for (t, w), g in r.groupby(["team", "week"]):
            out[(t, w)] = list(g.pfr_id)
    return out


def _injury_history(data_dir, season):
    ip = Path(data_dir) / f"injuries_{season}.csv.gz"
    if not ip.exists():
        return {}
    i = pd.read_csv(ip, low_memory=False)
    i["team"] = _fix(i["team"])
    return {(t, w, g): s for t, w, g, s in zip(i.team, i.week, i.gsis_id, i.report_status) if isinstance(s, str)}


def _missed_injured(inj, team, gsis, season, week):
    """Was this player listed Out/Doubtful for this team in one of the last 3 weeks?"""
    return any(inj.get((team, w, gsis)) in ("Out", "Doubtful") for w in range(max(1, week - 3), week))


def build(data_dir, first_season=2016, verbose=True):
    """Compute weekly impact ratings, production ratings and per-game lineup values.

    Returns dict with:
      lineups      one row per (game_id, team): off/def lineup value, full-strength value, injury cost
      lineup_detail {(game_id, team): list of player dicts} for the HUD
      history      weekly impact ratings per player (for trends)
      production   per-game production ratings (QB/rush/rec)
      latest       latest impact ratings per player
    """
    t0 = time.time()
    games = F.load_games(data_dir)
    games = games[games.season >= first_season].reset_index(drop=True)
    seasons = sorted(games.season.unique())
    pbp = F.load_pbp(data_dir, seasons)
    snaps = load_snaps(data_dir, seasons)
    snaps = snaps[snaps.game_id.isin(games.game_id)]
    obs = team_game_obs(pbp)
    IR = ImpactRatings(snaps, obs, games)
    info = snaps.sort_values(["season", "week"]).groupby("pfr_player_id").agg(
        name=("player", "last"), pos=("position", "last"), team=("team", "last"))
    if verbose:
        print(f"[players] data loaded in {time.time() - t0:.1f}s; {len(IR.ids):,} players")

    gday = games.set_index("game_id").gameday
    snaps = snaps.assign(gameday=snaps.game_id.map(gday)).sort_values("gameday")
    # per team: ordered list of (game_id, {pid: (off_pct, def_pct)})
    team_games = {}
    for (team, gid), t in snaps.groupby(["team", "game_id"], sort=False):
        team_games.setdefault(team, []).append((gday[gid], gid, dict(zip(t.pfr_player_id, zip(t.offense_pct, t.defense_pct)))))
    for team in team_games:
        team_games[team].sort(key=lambda x: x[0])

    # weekly checkpoints
    current = int(games.loc[games.played, "season"].max())
    wk = games.groupby(["season", "week"]).gameday.min().reset_index()
    first_unplayed = games[~games.played & (games.season == current)].gameday.min()
    fits, history = {}, []
    pid_index = IR.ids
    upcoming_fit = None  # fit for the first week that starts after the last played game
    for r in wk.itertuples(index=False):
        future = pd.notna(first_unplayed) and r.gameday > first_unplayed
        if future and upcoming_fit is not None:
            fits[(r.season, r.week)] = upcoming_fit  # nothing new to learn from yet
            continue
        f = IR.fit(r.gameday)
        fits[(r.season, r.week)] = f
        if future:
            upcoming_fit = f
        if f is None:
            continue
        keep = (f["off_expo"] > 1.0) | (f["def_expo"] > 1.0)
        history.append(pd.DataFrame({
            "season": r.season, "week": r.week, "pid": pid_index[keep],
            "off": np.round(f["off"][keep] * PLAYS_PER_GAME, 3), "def": np.round(-f["def"][keep] * PLAYS_PER_GAME, 3),
            "off_expo": np.round(f["off_expo"][keep], 2), "def_expo": np.round(f["def_expo"][keep], 2)}))
    history = pd.concat(history, ignore_index=True)
    if verbose:
        print(f"[players] {len(fits)} weekly refits in {time.time() - t0:.1f}s")

    # replacement level: recency-weighted mean of low-exposure players at the last fit
    lat = upcoming_fit if upcoming_fit is not None else \
        next(fits[k] for k in sorted(fits, reverse=True) if fits[k] is not None)

    # lineups
    rows, detail = [], {}
    upcoming_weeks = set()
    for g in games.itertuples(index=False):
        f = fits.get((g.season, g.week))
        if f is None:
            continue
        lo = f["off_expo"]; ld = f["def_expo"]
        r_off_repl = np.average(f["off"][(lo > 0.5) & (lo < 4)], weights=lo[(lo > 0.5) & (lo < 4)])
        r_def_repl = np.average(f["def"][(ld > 0.5) & (ld < 4)], weights=ld[(ld > 0.5) & (ld < 4)])
        for team, side in ((g.home_team, "home"), (g.away_team, "away")):
            tg = team_games.get(team, [])
            prior = [x for x in tg if x[0] < g.gameday][-RECENT_GAMES:]
            if not prior:
                continue
            exp_off, exp_def = {}, {}
            for _, _, pl in prior:
                for pid, (o, d) in pl.items():
                    if o > 0: exp_off.setdefault(pid, []).append(o)
                    if d > 0: exp_def.setdefault(pid, []).append(d)
            # a player's expected share: average over the recent games he played, scaled by how often he played
            n_prior = len(prior)
            e_off = {p: np.mean(v) * len(v) / n_prior for p, v in exp_off.items()}
            e_def = {p: np.mean(v) * len(v) / n_prior for p, v in exp_def.items()}
            if g.played:
                this = next((x[2] for x in tg if x[1] == g.game_id), None)
                if this is None:
                    continue
                avail = {p: (1.0, None) if p in this else (0.0, "did not play") for p in set(e_off) | set(e_def)}
            else:
                upcoming_weeks.add((g.season, g.week))
                avail = None  # filled below per week
            rows.append({"game_id": g.game_id, "season": g.season, "week": g.week, "team": team, "side": side,
                         "played": g.played, "_e_off": e_off, "_e_def": e_def, "_avail": avail,
                         "_f": (g.season, g.week), "_repl": (r_off_repl, r_def_repl)})

    # availability for upcoming weeks from rosters + injury report
    for (s, w) in upcoming_weeks:
        tp = {}
        for r in rows:
            if r["_avail"] is None and r["season"] == s and r["week"] == w:
                tp.setdefault(r["team"], set()).update(r["_e_off"]); tp[r["team"]].update(r["_e_def"])
        av = availability_for_upcoming(data_dir, s, w, tp)
        for r in rows:
            if r["_avail"] is None and r["season"] == s and r["week"] == w:
                r["_avail"] = {p: av.get((r["team"], p), (1.0, None)) for p in set(r["_e_off"]) | set(r["_e_def"])}

    # roster-based lineup: who is actually on the team now (handles free agency, trades,
    # rookies and departures), each player at his recent snap share (with this team if he has
    # one, otherwise with his previous team), rated by his own impact rating.
    gday_of_all = dict(zip(games.game_id, games.gameday))
    gday_of = gday_of_all
    hist = {}
    for pid, t in snaps[["pfr_player_id", "gameday", "offense_pct", "defense_pct"]].groupby("pfr_player_id"):
        hist[pid] = (t.gameday.values, t.offense_pct.values, t.defense_pct.values)
    roster_wk = _roster_pfr(data_dir, current)

    def recent_elsewhere(pid, day):
        h = hist.get(pid)
        if h is None:
            return 0.0, 0.0
        k = np.searchsorted(h[0], np.datetime64(day))
        if k == 0:
            return 0.0, 0.0
        lo = max(0, k - RECENT_GAMES)
        return float(np.mean(h[1][lo:k])), float(np.mean(h[2][lo:k]))

    new_avail_req = {}
    for r in rows:
        day = gday_of_all[r["game_id"]]
        if r["played"]:
            tg = team_games.get(r["team"], [])
            this = next((x[2] for x in tg if x[1] == r["game_id"]), {})
            cands = {p: 1.0 for p in this}
        else:
            cands = {p: a for p, (a, _) in r["_avail"].items()}
            for p in roster_wk.get((r["team"], r["week"]), ()):
                if p not in cands:
                    cands[p] = None  # new arrival / returning player: availability looked up below
                    new_avail_req.setdefault((r["season"], r["week"]), {}).setdefault(r["team"], set()).add(p)
        r["_cands"] = cands
        r["_day"] = day
    for (s_, w_), tp in new_avail_req.items():
        av = availability_for_upcoming(data_dir, s_, w_, tp)
        for r in rows:
            if not r["played"] and r["season"] == s_ and r["week"] == w_:
                for p, a in r["_cands"].items():
                    if a is None:
                        r["_cands"][p] = av.get((r["team"], p), (1.0, None))[0]
    for r in rows:
        f = fits[r["_f"]]
        ro, rd = r["_repl"]
        for unit, e_team, coef, repl, k in (("off", r["_e_off"], f["off"], ro, 0), ("def", r["_e_def"], f["def"], rd, 1)):
            shares = []
            for p, a in r["_cands"].items():
                if not a:
                    continue
                e = e_team.get(p)
                if e is None:
                    e = recent_elsewhere(p, r["_day"])[k]
                if e <= 0.02:
                    continue
                j = pid_index.get_loc(p) if p in pid_index else None
                rp = coef[j] if j is not None else repl
                shares.append((e * a, rp))
            shares.sort(key=lambda x: -x[0])
            tot, val = 0.0, 0.0
            for e, rp in shares:
                take = min(e, 11.0 - tot)
                if take <= 0:
                    break
                val += take * rp
                tot += take
            val += (11.0 - tot) * repl
            r[f"_roster_{unit}"] = val * PLAYS_PER_GAME

    # projected starting QB for upcoming games.
    # Default: whoever started the team's last game, unless he's on IR / off the roster /
    # listed Out or Doubtful this week. Starters who missed recent games injured and aren't
    # ruled out are offered as "possible return"; the HUD lets you switch QBs.
    qb_proj = {}
    pf2gs = _id_maps(data_dir, current)
    gs2pf = {v: k for k, v in pf2gs.items()}
    pos = info.pos
    roster_qbs = _roster_qbs(data_dir, current)
    inj_hist = _injury_history(data_dir, current)
    for r in rows:
        if r["played"]:
            continue
        team, week = r["team"], r["week"]
        tg = [x for x in team_games.get(team, []) if x[0] < gday_of[r["game_id"]]]
        starters = []
        for _, gid, pl_ in tg:
            qbs = [(o, p) for p, (o, _) in pl_.items() if pos.get(p) == "QB"]
            o, p = max(qbs) if qbs else (0, None)
            starters.append((gid, p if o >= 0.5 else None))
        season_starts = {}
        for gid, p in starters:
            if p and gid.startswith(str(r["season"])):
                season_starts[p] = season_starts.get(p, 0) + 1
        ros = roster_qbs.get((team, week), {})  # gsis -> status
        def avail(p):
            gs = pf2gs.get(p)
            if ros and gs not in ros:
                return False, "not on roster"
            if ros and ros.get(gs) in ("RES", "CUT", "RET", "DEV"):
                return False, {"RES": "injured reserve", "DEV": "practice squad"}.get(ros[gs], ros[gs].lower())
            st = inj_hist.get((team, week, gs))
            if st in ("Out", "Doubtful"):
                return False, st.lower()
            return True, ("questionable" if st == "Questionable" else None)
        last6 = [p for _, p in starters[-6:]]
        def entry(p, note=None):
            gs = pf2gs.get(p)
            ok, why = avail(p)
            return {"pfr": p, "gsis": gs, "name": info.name.get(p), "new": float(last6.count(p) <= 1),
                    "season_starts": season_starts.get(p, 0), "available": ok, "status": why, "note": note}
        options, seen = [], set()
        recent = [p for _, p in reversed(starters) if p]
        for p in recent:
            if p not in seen:
                seen.add(p)
                options.append(entry(p))
        # rostered QBs who haven't started recently (backups, rookies)
        for gs, st in ros.items():
            p = gs2pf.get(gs)
            if p and p not in seen and st == "ACT":
                seen.add(p)
                options.append(entry(p))
        options = [o for o in options if o["available"] or o["season_starts"]]
        proj = next((o for o in options if o["available"] and o["pfr"] in recent[:1]), None) \
            or next((o for o in options if o["available"] and o["pfr"] in recent), None)
        last_starter = recent[0] if recent else None
        for o in options:
            if o["available"] and o["pfr"] != last_starter and o["season_starts"] and \
                    _missed_injured(inj_hist, team, o["gsis"], r["season"], week):
                o["note"] = "possible return from injury"
        if proj:
            qb_proj[(r["game_id"], team)] = {**proj, "options": options[:5]}

    out = []
    for r in rows:
        f = fits[r["_f"]]
        ro, rd = r["_repl"]
        res = {"game_id": r["game_id"], "team": r["team"], "side": r["side"],
               "roster_off": r.get("_roster_off"), "roster_def": r.get("_roster_def")}
        det = []
        for unit, e, coef, repl in (("off", r["_e_off"], f["off"], ro), ("def", r["_e_def"], f["def"], rd)):
            tot = sum(e.values()) or 1.0
            scale = 11.0 / tot
            full = val = 0.0
            # team-specific replacement: this team's own reserves (low recent snap share)
            bk = [(share, coef[pid_index.get_loc(p)]) for p, share in e.items() if share < 0.35]
            trepl = (sum(s_ * c_ for s_, c_ in bk) / sum(s_ for s_, _ in bk)) if bk and sum(s_ for s_, _ in bk) > 0.5 else repl
            cost_rel = 0.0
            for p, share in e.items():
                j = pid_index.get_loc(p)
                rp = coef[j]
                a, why = r["_avail"].get(p, (1.0, None))
                full += share * rp
                val += share * (a * rp + (1 - a) * repl)
                cost_rel += share * (1 - a) * (rp - trepl)
                if r["season"] >= current or not r["played"]:
                    det.append({"pid": p, "unit": unit, "share": round(share, 3), "avail": a, "why": why,
                                "pts": round((rp if unit == "off" else -rp) * PLAYS_PER_GAME, 2)})
            res[f"{unit}_val"] = val * scale * PLAYS_PER_GAME
            res[f"{unit}_full"] = full * scale * PLAYS_PER_GAME
            res[f"{unit}_cost_rel"] = cost_rel * PLAYS_PER_GAME
        res["injury_cost"] = (res["off_full"] - res["off_val"]) - (res["def_full"] - res["def_val"])
        # points lost vs. the team's own backups (positive = hurts this team)
        res["injury_rel"] = res["off_cost_rel"] - res["def_cost_rel"]
        out.append(res)
        if det:
            detail[(r["game_id"], r["team"])] = det
    lineups = pd.DataFrame(out)

    prod = roll_production(production_ratings(pbp, games), games)
    if verbose:
        print(f"[players] lineups + production done in {time.time() - t0:.1f}s")

    latest = pd.DataFrame({"pid": pid_index, "off": lat["off"] * PLAYS_PER_GAME, "def": -lat["def"] * PLAYS_PER_GAME,
                           "off_expo": lat["off_expo"], "def_expo": lat["def_expo"]})
    latest = latest.join(info, on="pid")
    return {"lineups": lineups, "lineup_detail": detail, "history": history, "production": prod,
            "latest": latest, "info": info, "fits": fits, "qb_proj": qb_proj}


def game_features(pl, games_feats):
    """Attach lineup + QB production features to the per-game feature table from features.py."""
    lu = pl["lineups"]
    h = lu[lu.side == "home"].set_index("game_id")
    a = lu[lu.side == "away"].set_index("game_id")
    df = games_feats.copy()
    for pre, t in (("h_", h), ("a_", a)):
        for c in ("off_val", "def_val", "off_full", "def_full", "injury_cost", "injury_rel", "roster_off", "roster_def"):
            df[pre + c] = df.game_id.map(t[c])
    # expected points edge from lineups: home offense vs away defense minus the reverse
    # (def values are "EPA allowed" scale: positive = worse defense)
    # def values are on the "EPA allowed" scale (positive = worse defense), so the opponent's
    # defense value adds to your offense
    df["d_lineup"] = (df.h_off_val + df.a_def_val) - (df.a_off_val + df.h_def_val)
    df["d_lineup_full"] = (df.h_off_full + df.a_def_full) - (df.a_off_full + df.h_def_full)
    df["d_roster"] = (df.h_roster_off + df.a_roster_def) - (df.a_roster_off + df.h_roster_def)
    df["d_injury"] = df.h_injury_cost - df.a_injury_cost
    df["d_injury_rel"] = df.h_injury_rel - df.a_injury_rel

    # upcoming games: use the projected starter (handles QBs returning from injury)
    df["home_qb_source"] = np.where(df.played, "actual", "schedule")
    df["away_qb_source"] = df["home_qb_source"]
    for side, team_col, pre in (("home", "home_team", "h_"), ("away", "away_team", "a_")):
        for i in df.index[~df.played]:
            pj = pl["qb_proj"].get((df.at[i, "game_id"], df.at[i, team_col]))
            if pj and pj["gsis"]:
                df.at[i, f"{side}_qb_source"] = "last game's starter"
                df.at[i, f"{side}_qb_id"] = pj["gsis"]
                df.at[i, f"{side}_qb"] = pj["name"]
                df.at[i, f"{pre}qb_new"] = pj["new"]
    df["d_qb_new"] = df.h_qb_new - df.a_qb_new

    # QB production rating of the listed starter, pre-game, in points per game (x 35 dropbacks)
    pr = pl["production"]
    q = pr[pr.kind == "qb"]
    by_game = {(p, g): r for p, g, r in zip(q.pid, q.game_id, q.rating) if g is not None}
    last = q.sort_values("gameday").groupby("pid").rating.last()
    def qb_val(pid, gid, played):
        if not isinstance(pid, str):
            return np.nan
        if played and (pid, gid) in by_game:
            return by_game[(pid, gid)]
        if pid in last.index and not played:
            return last[pid]
        return -0.08  # unknown QB: backup-level prior
    df["h_qb_rating"] = [qb_val(p, g, pl_) for p, g, pl_ in zip(df.home_qb_id, df.game_id, df.played)]
    df["a_qb_rating"] = [qb_val(p, g, pl_) for p, g, pl_ in zip(df.away_qb_id, df.game_id, df.played)]
    df["d_qb_rating"] = (df.h_qb_rating - df.a_qb_rating) * 35.0

    # how far into the season each team is (roster info matters most early on)
    gp, h_gp, a_gp = {}, [], []
    for r in df.sort_values("gameday").itertuples():
        h_gp.append((r.Index, gp.get((r.season, r.home_team), 0)))
        a_gp.append((r.Index, gp.get((r.season, r.away_team), 0)))
        if r.played:
            gp[(r.season, r.home_team)] = gp.get((r.season, r.home_team), 0) + 1
            gp[(r.season, r.away_team)] = gp.get((r.season, r.away_team), 0) + 1
    df["h_gp"] = pd.Series(dict(h_gp)); df["a_gp"] = pd.Series(dict(a_gp))
    df["early"] = np.clip(1 - (df.h_gp + df.a_gp) / 2 / 6.0, 0, 1)
    df["d_roster_early"] = df["d_roster"].fillna(0) * df["early"]

    # final regular-season week: teams resting starters / nothing to play for
    last_reg = df[df.game_type == "REG"].groupby("season").week.transform("max")
    df["final_week"] = ((df.game_type == "REG") & (df.week == last_reg.reindex(df.index))).astype(float)
    for c in ("d_elo", "d_matchup", "d_qb_rating"):
        df[f"fw_x_{c[2:]}"] = df["final_week"] * df[c].fillna(0)
    return df
