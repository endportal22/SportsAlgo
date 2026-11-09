"""
Shared team-rating builder used by train.py, predict.py, backtest.py and the HUD.

One place builds features for past and upcoming games alike, so training and
prediction can never drift apart.

How it works
------------
Games are walked in date order. Each team keeps a running "rating state":

  * ev_xgf60 / ev_xga60  5-on-5 expected goals for / against per 60 minutes,
                         exponentially weighted and opponent-adjusted
  * st_net               special-teams xG edge per game (power play + penalty kill:
                         how often they get / give power plays, and what they do with them)
  * elo                  margin-of-victory Elo (FiveThirtyEight's NHL version)
  * last game date       for rest days and back-to-backs

For every game the features are taken *before* the game is played (a snapshot of
both teams' current state), then the state is updated with the result. Upcoming
games just take the latest snapshot, so the most recent game is always included.

Between seasons each rating is pulled back toward league average and its weight
is reset to a few games' worth, so October isn't driven by last April.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from paths import MODELS

# --------------------------------------------------------------------------- params


@dataclass
class Params:
    decay: float = 0.95          # per-game decay of older games in the EW means (half-life ~14 games)
    carry_weight: float = 10.0   # last season's rating counts as this many games in October
    regress: float = 0.65        # keep this share of last season's rating (rest -> league avg)
    opp_adjust: bool = True
    elo_k: float = 6.0
    elo_hfa: float = 50.0
    elo_regress: float = 0.3

    @classmethod
    def load(cls):
        """Defaults, overridden by models/params.json if experiments.py --tune --save wrote one."""
        p = cls()
        f = MODELS / "params.json"
        if f.exists():
            try:
                for k, v in json.loads(f.read_text()).items():
                    if k in {x.name for x in fields(cls)}:
                        setattr(p, k, type(getattr(p, k))(v))
            except (ValueError, TypeError):
                pass
        return p


TEAM_FIX = {"ARI": "UTA", "PHX": "UTA"}  # Arizona moved to Utah in 2024 (same roster)
EV_MIN = 48.0       # typical 5-on-5 minutes per game (re-estimated from the data)


# --------------------------------------------------------------------------- loading

def load_games(data_dir) -> pd.DataFrame:
    g = pd.read_csv(Path(data_dir) / "games.csv", low_memory=False)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g.sort_values(["gameday", "start_utc", "game_id"], na_position="last").reset_index(drop=True)
    played = g["state"].isin(["OFF", "FINAL"]) & g["home_score"].notna() & g["away_score"].notna()
    g["played"] = played
    g["margin"] = np.where(played, g["home_score"] - g["away_score"], np.nan)
    g["home_win"] = np.where(played, (g["margin"] > 0).astype(float), np.nan)
    g["extra_time"] = g["last_period"].isin(["OT", "SO"]) & played
    g["neutral"] = g["neutral"].fillna(False).astype(bool)
    return g


def load_kind(data_dir, kind, seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        f = Path(data_dir) / f"{kind}_{s}.csv.gz"
        if f.exists():
            d = pd.read_csv(f, low_memory=False)
            d["season"] = s
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No {kind}_*.csv.gz files in {data_dir}. Press Update data (or run update_data.py).")
    out = pd.concat(frames, ignore_index=True)
    for c in ("team", "opp"):
        if c in out:
            out[c] = out[c].replace(TEAM_FIX)
    return out


def team_game_stats(shots: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): 5v5 and special-teams xG, goals and minutes."""
    s = shots[shots.opp_g == 1]
    unb = s[s.ev.isin(["G", "S", "M"])]
    ev = unb[unb.strength == "5v5"]
    st = unb[unb.strength != "5v5"]
    pp = unb[unb.strength == "PP"]

    def by(df, col, val, how="sum"):
        return df.groupby(["game_id", col])[val].agg(how)

    out = pd.DataFrame({
        "ev_xgf": by(ev, "team", "xg"), "st_xgf": by(st, "team", "xg"), "pp_xgf": by(pp, "team", "xg"),
        "gf": by(unb.assign(g=(unb.ev == "G").astype(float)), "team", "g"),
        "xgf": by(unb, "team", "xg"),
        "cf5": by(s[s.strength == "5v5"].assign(one=1.0), "team", "one"),
    })
    out.index.names = ["game_id", "team"]
    against = pd.DataFrame({
        "ev_xga": by(ev, "opp", "xg"), "st_xga": by(st, "opp", "xg"), "pk_xga": by(pp, "opp", "xg"),
        "ga": by(unb.assign(g=(unb.ev == "G").astype(float)), "opp", "g"),
        "xga": by(unb, "opp", "xg"),
        "ca5": by(s[s.strength == "5v5"].assign(one=1.0), "opp", "one"),
    })
    against.index.names = ["game_id", "team"]
    st_ = states.set_index(["game_id", "team"])
    keep = [c for c in ("toi_5v5", "toi_pp", "toi_pk", "toi_ev_other", "toi_en") if c in st_.columns]
    out = st_[keep].join(out, how="left").join(against, how="left").fillna(0.0).reset_index()
    for c in ("toi_5v5", "toi_pp", "toi_pk"):
        if c not in out:
            out[c] = 0.0
    return out


# --------------------------------------------------------------------------- rating engine

@dataclass
class TeamState:
    ev_xgf60: float = 2.5
    ev_xga60: float = 2.5
    st_net: float = 0.0
    pp60: float = 6.5
    pk60: float = 6.5
    pp_min: float = 5.5
    pk_min: float = 5.5
    gd: float = 0.0          # goal differential per game (non-empty-net), for experiments
    fin: float = 0.0         # goals minus xG per game (finishing + goaltending), for experiments
    w: float = 0.0
    elo: float = 1505.0
    season: int | None = None
    last_day: pd.Timestamp | None = None


@dataclass
class League:
    teams: dict = field(default_factory=dict)
    mu_ev60: float = 2.5
    mu_pp60: float = 6.5
    ev_min: float = EV_MIN
    n: int = 0

    def get(self, t) -> TeamState:
        if t not in self.teams:
            self.teams[t] = TeamState(ev_xgf60=self.mu_ev60, ev_xga60=self.mu_ev60, pp60=self.mu_pp60, pk60=self.mu_pp60)
        return self.teams[t]


def _new_season(ts: TeamState, season: int, lg: League, p: Params):
    if ts.season is None:
        ts.w = p.carry_weight          # a brand-new team starts at league average, worth a few games
    elif ts.season != season:
        r = p.regress
        ts.ev_xgf60 = lg.mu_ev60 + r * (ts.ev_xgf60 - lg.mu_ev60)
        ts.ev_xga60 = lg.mu_ev60 + r * (ts.ev_xga60 - lg.mu_ev60)
        ts.pp60 = lg.mu_pp60 + r * (ts.pp60 - lg.mu_pp60)
        ts.pk60 = lg.mu_pp60 + r * (ts.pk60 - lg.mu_pp60)
        avg_min = 0.5 * (ts.pp_min + ts.pk_min)
        ts.pp_min = avg_min + r * (ts.pp_min - avg_min)
        ts.pk_min = avg_min + r * (ts.pk_min - avg_min)
        ts.st_net *= r
        ts.gd *= r
        ts.fin *= r
        ts.w = min(ts.w, p.carry_weight)
        ts.elo = 1505 + (1 - p.elo_regress) * (ts.elo - 1505)
        ts.last_day = None
    ts.season = season


def _ew(old, x, w_old, decay, wt=1.0):
    w_new = w_old * decay + wt
    return old + (x - old) * wt / w_new, w_new


def ev_edge(h: TeamState, a: TeamState, lg: League) -> float:
    """Expected 5-on-5 xG margin per game for h vs a (offense of each against the other's defense)."""
    mu = lg.mu_ev60
    h_for = h.ev_xgf60 + a.ev_xga60 - mu
    a_for = a.ev_xgf60 + h.ev_xga60 - mu
    return (h_for - a_for) * lg.ev_min / 60.0


def st_edge(h: TeamState, a: TeamState, lg: League) -> float:
    return h.st_net - a.st_net


def build_features(games: pd.DataFrame, tstats: pd.DataFrame, p: Params = None, return_league: bool = False):
    """Return one row per game (past and future) with pre-game features."""
    p = p or Params.load()
    stats = {(r.game_id, r.team): r for r in tstats.itertuples(index=False)}
    lg = League()
    rows = []
    for g in games.itertuples(index=False):
        h, a = lg.get(g.home_team), lg.get(g.away_team)
        _new_season(h, g.season, lg, p)
        _new_season(a, g.season, lg, p)
        neutral = bool(g.neutral)
        day = g.gameday
        rest_h = (day - h.last_day).days if h.last_day is not None else 5
        rest_a = (day - a.last_day).days if a.last_day is not None else 5
        rest_h, rest_a = min(max(rest_h, 0), 5), min(max(rest_a, 0), 5)
        rows.append({
            "game_id": g.game_id, "season": g.season, "game_type": g.game_type, "gameday": day,
            "start_utc": g.start_utc, "home_team": g.home_team, "away_team": g.away_team,
            "played": g.played, "margin": g.margin, "home_win": g.home_win, "extra_time": g.extra_time,
            "h_ev_xgf60": h.ev_xgf60, "h_ev_xga60": h.ev_xga60, "a_ev_xgf60": a.ev_xgf60, "a_ev_xga60": a.ev_xga60,
            "d_elo": h.elo - a.elo,
            "d_ev": ev_edge(h, a, lg),
            "d_st": st_edge(h, a, lg),
            "d_gd": h.gd - a.gd, "d_fin": h.fin - a.fin,
            "h_rest": rest_h, "a_rest": rest_a,
            "rest_diff": float(np.clip(rest_h - rest_a, -3, 3)),
            "h_b2b": float(rest_h == 1), "a_b2b": float(rest_a == 1),
            "d_b2b": float(rest_h == 1) - float(rest_a == 1),
            "home_field": 0.0 if neutral else 1.0,
            "playoff": float(g.game_type == "P"),
            "h_games": h.w, "a_games": a.w,
        })
        if not g.played:
            continue
        h.last_day = a.last_day = day

        # ---- Elo update (margin of victory, FiveThirtyEight NHL style)
        hfa = 0 if neutral else p.elo_hfa
        dr = h.elo + hfa - a.elo
        exp_h = 1 / (1 + 10 ** (-dr / 400))
        res = 1.0 if g.margin > 0 else 0.0
        mov = max(abs(g.margin), 1)
        win_diff = dr if g.margin > 0 else -dr
        mult = (0.6686 * np.log(mov) + 0.8048) * 2.05 / (win_diff * 0.001 + 2.05)
        delta = p.elo_k * mult * (res - exp_h)
        h.elo += delta
        a.elo -= delta

        # ---- xG updates
        sh, sa = stats.get((g.game_id, g.home_team)), stats.get((g.game_id, g.away_team))
        if sh is None or sa is None:
            continue
        toi5 = max(sh.toi_5v5, 1.0) / 60.0    # minutes
        if toi5 < 10:                         # missing / broken play-by-play
            continue
        wt = min(toi5 / lg.ev_min, 1.25)
        hx, hxa = sh.ev_xgf * 60 / toi5, sh.ev_xga * 60 / toi5
        ax, axa = sa.ev_xgf * 60 / toi5, sa.ev_xga * 60 / toi5
        mu = lg.mu_ev60
        if p.opp_adjust:
            h_o, h_d = hx - (a.ev_xga60 - mu), hxa - (a.ev_xgf60 - mu)
            a_o, a_d = ax - (h.ev_xga60 - mu), axa - (h.ev_xgf60 - mu)
        else:
            h_o, h_d, a_o, a_d = hx, hxa, ax, axa
        for ts, st, o, d_ in ((h, sh, h_o, h_d), (a, sa, a_o, a_d)):
            w0 = ts.w
            ts.ev_xgf60, _ = _ew(ts.ev_xgf60, o, w0, p.decay, wt)
            ts.ev_xga60, _ = _ew(ts.ev_xga60, d_, w0, p.decay, wt)
            ts.st_net, _ = _ew(ts.st_net, st.st_xgf - st.st_xga, w0, p.decay)
            ts.pp_min, _ = _ew(ts.pp_min, st.toi_pp / 60.0, w0, p.decay)
            ts.pk_min, _ = _ew(ts.pk_min, st.toi_pk / 60.0, w0, p.decay)
            if st.toi_pp > 60:
                ts.pp60, _ = _ew(ts.pp60, st.pp_xgf * 3600 / st.toi_pp, w0, p.decay, min(st.toi_pp / 360.0, 1.5))
            if st.toi_pk > 60:
                ts.pk60, _ = _ew(ts.pk60, st.pk_xga * 3600 / st.toi_pk, w0, p.decay, min(st.toi_pk / 360.0, 1.5))
            ts.gd, _ = _ew(ts.gd, st.gf - st.ga, w0, p.decay)
            ts.fin, ts.w = _ew(ts.fin, (st.gf - st.xgf) - (st.ga - st.xga), w0, p.decay)

        lg.n += 2
        k = min(lg.n, 2000)
        lg.mu_ev60 += ((hx + ax) / 2 - lg.mu_ev60) * 2 / k
        lg.ev_min += (toi5 - lg.ev_min) / min(lg.n / 2, 1000)
        for st in (sh, sa):
            if st.toi_pp > 60:
                lg.mu_pp60 += (st.pp_xgf * 3600 / st.toi_pp - lg.mu_pp60) / k

    out = pd.DataFrame(rows)
    return (out, lg) if return_league else out


def build_all(data_dir, shots, states, first_season: int, p: Params = None, return_league: bool = False):
    games = load_games(data_dir)
    games = games[games["season"] >= first_season]
    return build_features(games, team_game_stats(shots, states), p, return_league)
