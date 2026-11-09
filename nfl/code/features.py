"""
Shared feature builder used by train.py, predict.py and backtest.py.

One place builds features for past and upcoming games alike, so training and
prediction can never drift apart.

How it works
------------
Games are walked in date order. Each team keeps a running "rating state":

  * off_epa / def_epa   exponentially-weighted EPA/play (offense, defense allowed),
                        opponent-adjusted, pass+run plays only, garbage time removed
  * off_sr / def_sr     same idea for success rate
  * elo                 margin-of-victory Elo (FiveThirtyEight-style)
  * qbs                 recent starting QBs, to flag a new/backup starter

For every game the features are taken *before* the game is played (a snapshot of
both teams' current state), then the state is updated with the result. Upcoming
games just take the latest snapshot, so the most recent game is always included.

Between seasons each rating is pulled back toward league average and its weight
is reset to a few "games' worth", so Week 1 isn't driven by last December.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- params

@dataclass
class Params:
    decay: float = 0.92          # per-game decay of older games in the EW means
    carry_weight: float = 3.0    # prior from last season counts as this many games
    regress: float = 0.6         # keep this share of last season's rating (rest -> league avg)
    wp_floor: float = 0.05       # drop plays when win prob is outside [floor, 1-floor] (garbage time)
    opp_adjust: bool = True
    elo_k: float = 20.0
    elo_hfa: float = 48.0
    elo_regress: float = 1 / 3


TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # older abbreviations in games.csv

# Model inputs (home minus away unless noted)


# --------------------------------------------------------------------------- loading

def load_games(data_dir: str | Path) -> pd.DataFrame:
    g = pd.read_csv(Path(data_dir) / "games.csv", low_memory=False)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g.sort_values(["gameday", "gametime", "game_id"], na_position="last").reset_index(drop=True)
    played = g["home_score"].notna() & g["away_score"].notna()
    g["played"] = played
    g["margin"] = np.where(played, g["home_score"] - g["away_score"], np.nan)
    g["home_win"] = np.where(played, (g["margin"] > 0).astype(float) + 0.5 * (g["margin"] == 0), np.nan)
    return g


def load_pbp(data_dir: str | Path, seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        f = Path(data_dir) / f"pbp_{s}.csv.gz"
        if f.exists():
            frames.append(pd.read_csv(f, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No pbp_*.csv.gz files in {data_dir}. Run update_data.py first.")
    return pd.concat(frames, ignore_index=True)


def team_game_stats(pbp: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Per (game_id, team) offensive and defensive efficiency from real snaps only."""
    d = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    d = d[(d.get("qb_kneel", 0).fillna(0) == 0) & (d.get("qb_spike", 0).fillna(0) == 0)]
    if p.wp_floor > 0:
        d = d[d["wp"].between(p.wp_floor, 1 - p.wp_floor) | d["wp"].isna()]
    for c in ("posteam", "defteam"):
        d[c] = d[c].replace(TEAM_FIX)

    d["is_pass"] = (d["play_type"] == "pass").astype(float)
    d["p_epa"] = np.where(d["is_pass"] == 1, d["epa"], np.nan)
    d["r_epa"] = np.where(d["is_pass"] == 0, d["epa"], np.nan)
    off = d.groupby(["game_id", "posteam"]).agg(off_epa=("epa", "mean"), off_sr=("success", "mean"),
                                                 off_plays=("epa", "size"), off_pass_rate=("is_pass", "mean"),
                                                 off_pass=("p_epa", "mean"), off_run=("r_epa", "mean"))
    off.index.names = ["game_id", "team"]
    dfn = d.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean"), def_sr=("success", "mean"),
                                                 def_pass=("p_epa", "mean"), def_run=("r_epa", "mean"))
    dfn.index.names = ["game_id", "team"]
    return off.join(dfn, how="inner").reset_index()


# --------------------------------------------------------------------------- rating engine

@dataclass
class TeamState:
    off_epa: float = 0.0
    def_epa: float = 0.0
    off_sr: float = 0.44
    def_sr: float = 0.44
    w: float = 0.0
    elo: float = 1505.0
    season: int | None = None
    pass_off: float = 0.15
    run_off: float = -0.08
    pass_def: float = 0.15
    run_def: float = -0.08
    pass_rate: float = 0.58
    qbs: list = field(default_factory=list)   # starting QB ids of recent games


def _qb_new(ts: "TeamState", qb) -> float:
    """1 if this week's listed starter started at most 1 of the team's last 6 games."""
    if not ts.qbs or qb is None or (isinstance(qb, float) and np.isnan(qb)):
        return 0.0
    return float(ts.qbs[-6:].count(qb) <= 1)


@dataclass
class League:
    teams: dict = field(default_factory=dict)
    mu_epa: float = 0.0
    mu_sr: float = 0.44
    mu_pass: float = 0.15
    mu_run: float = -0.08

    def get(self, t) -> TeamState:
        if t not in self.teams:
            self.teams[t] = TeamState(off_epa=self.mu_epa, def_epa=self.mu_epa,
                                      off_sr=self.mu_sr, def_sr=self.mu_sr)
        return self.teams[t]


def _new_season(ts: TeamState, season: int, lg: League, p: Params):
    if ts.season is not None and ts.season != season:
        r = p.regress
        ts.off_epa = lg.mu_epa + r * (ts.off_epa - lg.mu_epa)
        ts.def_epa = lg.mu_epa + r * (ts.def_epa - lg.mu_epa)
        ts.off_sr = lg.mu_sr + r * (ts.off_sr - lg.mu_sr)
        ts.def_sr = lg.mu_sr + r * (ts.def_sr - lg.mu_sr)
        for k, m in (("pass_off", lg.mu_pass), ("pass_def", lg.mu_pass), ("run_off", lg.mu_run), ("run_def", lg.mu_run),
                     ("pass_rate", 0.58)):
            setattr(ts, k, m + r * (getattr(ts, k) - m))
        ts.w = min(ts.w, p.carry_weight)
        ts.elo = 1505 + (1 - p.elo_regress) * (ts.elo - 1505)
    ts.season = season


def _exp_off(o: "TeamState", d: "TeamState", lg: "League") -> float:
    """Expected EPA/play for offense o against defense d, mixing pass and run by o's pass rate."""
    pr = o.pass_rate
    return pr * (o.pass_off + d.pass_def - lg.mu_pass) + (1 - pr) * (o.run_off + d.run_def - lg.mu_run)


def _ew(old, x, w_old, decay):
    w_new = w_old * decay + 1.0
    return old + (x - old) / w_new, w_new


def build_features(games: pd.DataFrame, tstats: pd.DataFrame, p: Params = Params(), return_league: bool = False):
    """Return one row per game (past and future) with pre-game features."""
    stats = {(r.game_id, r.team): r for r in tstats.itertuples(index=False)}
    lg = League()
    # league means: running mean of team-game values (updated as we go)
    n_lg, sum_epa, sum_sr = 0, 0.0, 0.0
    rows = []

    for g in games.itertuples(index=False):
        h, a = lg.get(g.home_team), lg.get(g.away_team)
        _new_season(h, g.season, lg, p)
        _new_season(a, g.season, lg, p)
        neutral = str(g.location) == "Neutral"

        rest_h = g.home_rest if pd.notna(g.home_rest) else 7
        rest_a = g.away_rest if pd.notna(g.away_rest) else 7
        rows.append({
            "game_id": g.game_id, "season": g.season, "week": g.week, "game_type": g.game_type,
            "gameday": g.gameday, "home_team": g.home_team, "away_team": g.away_team,
            "played": g.played, "margin": g.margin, "home_win": g.home_win,
            "spread_line": g.spread_line,
            "h_off_epa": h.off_epa, "h_def_epa": h.def_epa, "a_off_epa": a.off_epa, "a_def_epa": a.def_epa,
            "d_off_epa": h.off_epa - a.off_epa,
            "d_def_epa": h.def_epa - a.def_epa,
            "d_net_epa": (h.off_epa - h.def_epa) - (a.off_epa - a.def_epa),
            "d_net_sr": (h.off_sr - h.def_sr) - (a.off_sr - a.def_sr),
            "d_elo": h.elo - a.elo,
            "rest_diff": float(np.clip(rest_h - rest_a, -7, 7)),
            "home_field": 0.0 if neutral else 1.0,
            "div_game": float(g.div_game) if pd.notna(g.div_game) else 0.0,
            "home_qb": g.home_qb_name, "away_qb": g.away_qb_name,
            "home_qb_id": g.home_qb_id, "away_qb_id": g.away_qb_id,
            "d_pass_net": (h.pass_off - h.pass_def) - (a.pass_off - a.pass_def),
            "d_run_net": (h.run_off - h.run_def) - (a.run_off - a.run_def),
            "d_matchup": _exp_off(h, a, lg) - _exp_off(a, h, lg),
            "h_qb_new": _qb_new(h, g.home_qb_id), "a_qb_new": _qb_new(a, g.away_qb_id),
            "h_games": h.w, "a_games": a.w,
        })

        rows[-1]["d_qb_new"] = rows[-1]["h_qb_new"] - rows[-1]["a_qb_new"]
        if not g.played:
            continue
        if pd.notna(g.home_qb_id): h.qbs.append(g.home_qb_id)
        if pd.notna(g.away_qb_id): a.qbs.append(g.away_qb_id)

        # ---- Elo update (margin of victory)
        hfa = 0 if neutral else p.elo_hfa
        exp_h = 1 / (1 + 10 ** (-(h.elo + hfa - a.elo) / 400))
        res = 1.0 if g.margin > 0 else 0.0 if g.margin < 0 else 0.5
        win_diff = (h.elo + hfa - a.elo) * (1 if g.margin > 0 else -1)
        mult = np.log(abs(g.margin) + 1) * 2.2 / (win_diff * 0.001 + 2.2) if g.margin != 0 else 1.0
        delta = p.elo_k * mult * (res - exp_h)
        h.elo += delta
        a.elo -= delta

        # ---- efficiency updates
        sh, sa = stats.get((g.game_id, g.home_team)), stats.get((g.game_id, g.away_team))
        if sh is None or sa is None:
            continue
        mu_e, mu_s = lg.mu_epa, lg.mu_sr
        if p.opp_adjust:
            # credit offense relative to how good the defense it faced was (and vice versa)
            h_off = sh.off_epa - (a.def_epa - mu_e); a_off = sa.off_epa - (h.def_epa - mu_e)
            h_def = sh.def_epa - (a.off_epa - mu_e); a_def = sa.def_epa - (h.off_epa - mu_e)
            h_osr = sh.off_sr - (a.def_sr - mu_s); a_osr = sa.off_sr - (h.def_sr - mu_s)
            h_dsr = sh.def_sr - (a.off_sr - mu_s); a_dsr = sa.def_sr - (h.off_sr - mu_s)
        else:
            h_off, a_off, h_def, a_def = sh.off_epa, sa.off_epa, sh.def_epa, sa.def_epa
            h_osr, a_osr, h_dsr, a_dsr = sh.off_sr, sa.off_sr, sh.def_sr, sa.def_sr

        for ts, st, op in ((h, sh, sa), (a, sa, sh)):
            for k, x in (("pass_off", st.off_pass), ("run_off", st.off_run), ("pass_def", st.def_pass),
                         ("run_def", st.def_run), ("pass_rate", st.off_pass_rate)):
                if pd.notna(x):
                    setattr(ts, k, _ew(getattr(ts, k), x, ts.w, p.decay)[0])

        for ts, o, d_, osr, dsr in ((h, h_off, h_def, h_osr, h_dsr), (a, a_off, a_def, a_osr, a_dsr)):
            w0 = ts.w
            ts.off_epa, _ = _ew(ts.off_epa, o, w0, p.decay)
            ts.def_epa, _ = _ew(ts.def_epa, d_, w0, p.decay)
            ts.off_sr, _ = _ew(ts.off_sr, osr, w0, p.decay)
            ts.def_sr, ts.w = _ew(ts.def_sr, dsr, w0, p.decay)

        n_lg += 2
        sum_epa += sh.off_epa + sa.off_epa
        sum_sr += sh.off_sr + sa.off_sr
        lg.mu_epa, lg.mu_sr = sum_epa / n_lg, sum_sr / n_lg
        for x in (sh.off_pass, sa.off_pass):
            if pd.notna(x): lg.mu_pass += (x - lg.mu_pass) / min(n_lg, 500)
        for x in (sh.off_run, sa.off_run):
            if pd.notna(x): lg.mu_run += (x - lg.mu_run) / min(n_lg, 500)

    out = pd.DataFrame(rows)
    return (out, lg) if return_league else out


def build_all(data_dir: str | Path, first_season: int = 2016, p: Params = Params(), return_league: bool = False):
    games = load_games(data_dir)
    games = games[games["season"] >= first_season]
    pbp = load_pbp(data_dir, sorted(games["season"].unique()))
    return build_features(games, team_game_stats(pbp, p), p, return_league)
