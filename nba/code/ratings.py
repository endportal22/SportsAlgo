"""
Team and player ratings, each computed only from games before the one being predicted.

  Elo            classic NBA Elo with margin of victory
  net rating     exponentially weighted, opponent- and home-adjusted point margin
  player ratings refit every week:
    - plus-minus (ridge): each game's margin (per 48 min, home-adjusted) explained by who played and
      how much: home players' minute shares minus away players'. Shrunk toward 0 = average.
    - box score: minutes-weighted, exponentially weighted Hollinger game score per 36 minutes,
      shrunk toward a bench-level value for players with few minutes.
  lineups        for a game, every available player's projected minutes (recent average,
                 scaled so the team adds up to 240) times his ratings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


class Params:
    elo_k = 20.0
    elo_home = 100.0
    elo_carry = 0.75
    net_halflife = 20.0      # games
    net_prior = 4.0          # games of shrinkage toward average
    net_carry = 0.35
    hca = 2.5                # points, for adjusting margins inside the ratings
    pm_halflife = 500.0      # days
    pm_window = 3 * 365      # days of games used per fit
    pm_alpha = 25.0          # ridge penalty
    box_halflife = 300.0     # days
    box_prior_min = 600.0    # minutes of shrinkage
    box_repl = -0.30         # bench level for the shrinkage, per minute, relative to average
    min_halflife = 4.0       # games, for projected minutes
    rookie_min = 12.0
    rapm_alpha = 1000.0      # ridge penalty for the play-by-play plus-minus (in minutes of floor time)
    rapm_halflife = 500.0    # days
    rapm_prior = 0.3         # weight of the box-score rating as the plus-minus prior


P = Params()


# ------------------------------------------------------------------ team ratings
def elo(games: pd.DataFrame, p=P) -> pd.DataFrame:
    r, season = {}, {}
    out_h, out_a = np.zeros(len(games)), np.zeros(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        for t in (g.home, g.away):
            if t not in r:
                r[t], season[t] = 1500.0, g.season
            elif season[t] != g.season:
                r[t] = p.elo_carry * r[t] + (1 - p.elo_carry) * 1505.0
                season[t] = g.season
        rh, ra = r[g.home], r[g.away]
        out_h[i], out_a[i] = rh, ra
        if not g.played:
            continue
        hadv = 0.0 if g.neutral else p.elo_home
        exp = 1.0 / (1.0 + 10 ** (-(rh + hadv - ra) / 400.0))
        mov = abs(g.margin)
        win = 1.0 if g.margin > 0 else 0.0
        dd = (rh + hadv - ra) if win else (ra - rh - hadv)
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * dd)
        delta = p.elo_k * mult * (win - exp)
        r[g.home] += delta
        r[g.away] -= delta
    return pd.DataFrame({"game_id": games.game_id.values, "elo_h": out_h, "elo_a": out_a})


def net_rating(games: pd.DataFrame, p=P) -> pd.DataFrame:
    """Opponent-adjusted margin, exponentially weighted by game; shrunk toward 0 for small samples."""
    lam = 0.5 ** (1.0 / p.net_halflife)
    num, den, season = {}, {}, {}

    def rating(t):
        return num.get(t, 0.0) / (den.get(t, 0.0) + p.net_prior)

    out_h, out_a = np.zeros(len(games)), np.zeros(len(games))
    # games on the same day can't see each other
    for day, idx in games.groupby("date", sort=True).indices.items():
        sub = games.iloc[idx]
        for t in set(sub.home) | set(sub.away):
            s = sub.season.iloc[0]
            if t in season and season[t] != s:
                num[t] *= p.net_carry
                den[t] *= p.net_carry
            season[t] = s
        pre = {t: rating(t) for t in set(sub.home) | set(sub.away)}
        for j, g in zip(idx, sub.itertuples(index=False)):
            out_h[j], out_a[j] = pre[g.home], pre[g.away]
        for g in sub.itertuples(index=False):
            if not g.played:
                continue
            h = 0.0 if g.neutral else p.hca
            for t, o, m in ((g.home, g.away, g.margin - h), (g.away, g.home, -g.margin + h)):
                num[t] = num.get(t, 0.0) * lam + (m + pre[o])
                den[t] = den.get(t, 0.0) * lam + 1.0
    return pd.DataFrame({"game_id": games.game_id.values, "net_h": out_h, "net_a": out_a})


# ------------------------------------------------------------------ player ratings
def _cutoffs(games):
    """Refit dates: every Monday with games around it (ratings used for games from that day on)."""
    d = games.date.drop_duplicates().sort_values()
    mondays = (d - pd.to_timedelta(d.dt.weekday, unit="D")).drop_duplicates()
    return list(mondays)


def fit_players(games, box, cutoff, p=P):
    """Ratings from games strictly before `cutoff`. Returns DataFrame(player_id, pm, box, minutes)."""
    lo = cutoff - pd.Timedelta(days=p.pm_window)
    g = games[games.played & (games.date < cutoff) & (games.date >= lo)]
    b = box[box.game_id.isin(g.game_id)]
    if g.empty or b.empty:
        return pd.DataFrame(columns=["player_id", "pm", "box", "minutes"])
    age = (cutoff - b.date).dt.days.values

    # box score: game score per minute, relative to the minutes-weighted league average
    wb = 0.5 ** (age / p.box_halflife)
    lg = np.average(b.gmsc / b["min"], weights=b["min"] * wb)
    bx = pd.DataFrame({"pid": b.player_id.values, "w": wb * b["min"].values, "v": (b.gmsc.values - lg * b["min"].values) * wb,
                       "m": b["min"].values * wb})
    agg = bx.groupby("pid").agg(v=("v", "sum"), m=("m", "sum"))
    boxr = (agg.v + p.box_repl * p.box_prior_min) / (agg.m + p.box_prior_min) * 36.0   # per 36 min, vs average

    # plus-minus ridge: one row per game, home shares minus away shares
    tot = b.groupby(["game_id", "team"])["min"].transform("sum")
    share = 5.0 * b["min"] / tot
    pids = np.sort(b.player_id.unique())
    col = np.searchsorted(pids, b.player_id.values)
    gidx = pd.Index(g.game_id)
    row = gidx.get_indexer(b.game_id.values)
    sign = np.where(b.is_home.values, 1.0, -1.0)
    X = sparse.csr_matrix((share.values * sign, (row, col)), shape=(len(g), len(pids)))
    home = (~g.neutral.values).astype(float)
    X = sparse.hstack([X, sparse.csr_matrix(home.reshape(-1, 1) * 3.0)]).tocsr()   # home court (scaled so it's barely penalised)
    gm = b.groupby("game_id")["min"].sum().reindex(g.game_id).values
    y = g.margin.values * 480.0 / np.where(gm > 0, gm, 480.0)   # per 48 minutes
    w = 0.5 ** ((cutoff - g.date).dt.days.values / p.pm_halflife)
    rg = Ridge(alpha=p.pm_alpha, fit_intercept=False, solver="auto").fit(X, y, sample_weight=w)
    pm = pd.Series(rg.coef_[:-1], index=pids)
    mins = b.groupby("player_id")["min"].sum()
    out = pd.DataFrame({"player_id": pids, "pm": pm.reindex(pids).values})
    out["box"] = boxr.reindex(pids).values
    out["minutes"] = mins.reindex(pids).values
    out.attrs["hca"] = float(rg.coef_[-1] * 3.0)
    return out


def fit_rapm(games, stints, cutoff, box_prior=None, p=P):
    """Plus-minus from who was actually on the floor together (play-by-play stints before `cutoff`):
    each stint's margin per 48 minutes explained by the ten players on the floor, weighted by
    minutes, shrunk toward 0 (or toward p.rapm_prior x the box-score rating)."""
    lo = cutoff - pd.Timedelta(days=p.pm_window)
    g = games[games.played & (games.date < cutoff) & (games.date >= lo)][["game_id", "date", "neutral"]]
    s = stints.merge(g, on="game_id", how="inner")
    s = s[s.secs >= 5]
    if s.empty:
        return pd.Series(dtype=float)
    hcols, acols = [f"h{k}" for k in range(1, 6)], [f"a{k}" for k in range(1, 6)]
    ids = s[hcols + acols].values
    pids = np.unique(ids)
    col = np.searchsorted(pids, ids)
    n = len(s)
    rows = np.repeat(np.arange(n), 10)
    vals = np.tile(np.r_[np.ones(5), -np.ones(5)], n)
    X = sparse.csr_matrix((vals, (rows, col.ravel())), shape=(n, len(pids)))
    home = (~s.neutral.values).astype(float).reshape(-1, 1)
    X = sparse.hstack([X, sparse.csr_matrix(home * 10.0)]).tocsr()
    mins = s.secs.values / 60.0
    y = (s.home_pts.values - s.away_pts.values) * 48.0 / mins
    w = mins * 0.5 ** ((cutoff - s.date).dt.days.values / p.rapm_halflife)
    prior = np.zeros(len(pids))
    if box_prior is not None and p.rapm_prior:
        prior = p.rapm_prior * box_prior.reindex(pids).fillna(0.0).values
        y = y - X[:, :-1] @ prior
    rg = Ridge(alpha=p.rapm_alpha, fit_intercept=False, solver="sparse_cg", tol=1e-5).fit(X, y, sample_weight=w)
    return pd.Series(rg.coef_[:-1] + prior, index=pids)


def all_fits(games, box, p=P, start=None, end=None, verbose=False, stints=None, rapm_cache=None):
    """{cutoff: ratings} for every weekly refit date from `start` through `end` (today);
    games after the last refit use the latest ratings."""
    fits = {}
    for c in _cutoffs(games):
        if start is not None and c < start:
            continue
        if end is not None and c > end:
            continue
        if not (games.date >= c).any():
            continue
        fits[c] = fit_players(games, box, c, p)
        if stints is not None and len(stints):
            key = (c, p.rapm_alpha, p.rapm_halflife, p.rapm_prior, p.pm_window)
            if rapm_cache is not None and key in rapm_cache:
                r = rapm_cache[key]
            else:
                r = fit_rapm(games, stints, c, fits[c].set_index("player_id").box, p)
                if rapm_cache is not None:
                    rapm_cache[key] = r
            fits[c]["rapm"] = r.reindex(fits[c].player_id).values
    if verbose:
        print(f"[ratings] {len(fits)} weekly player fits")
    return fits


def projected_minutes(box, p=P):
    """Each player's projected minutes going into each of his games (EW over his previous games)."""
    b = box[["game_id", "date", "player_id", "min"]].sort_values(["player_id", "date"])
    ew = b.groupby("player_id")["min"].transform(lambda s: s.shift(1).ewm(halflife=p.min_halflife, min_periods=1).mean())
    b = b.assign(proj=ew.fillna(p.rookie_min))
    last = box.sort_values("date").groupby("player_id").agg(last_min=("min", "last"), last_date=("date", "last"), team=("team", "last"))
    # minutes going into the *next* game (for upcoming games)
    nxt = box.sort_values(["player_id", "date"]).groupby("player_id")["min"].apply(
        lambda s: s.ewm(halflife=p.min_halflife).mean().iloc[-1])
    last["proj_next"] = nxt
    return b[["game_id", "player_id", "proj"]], last


def lineup_value(players: pd.DataFrame, rat: pd.DataFrame, box_default: float):
    """players: player_id, proj (minutes) for one team in one game -> (pm, box) lineup totals."""
    if players.empty:
        return 0.0, 0.0
    r = rat.reindex(players.player_id.values) if rat is not None and len(rat) else None
    pm = np.nan_to_num(r.pm.values, nan=0.0) if r is not None else np.zeros(len(players))
    bx = np.nan_to_num(r.box.values, nan=box_default) if r is not None else np.full(len(players), box_default)
    m = players.proj.values.astype(float)
    share = 5.0 * m / m.sum()
    return float(share @ pm), float(share @ bx)
