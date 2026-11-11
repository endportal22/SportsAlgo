#!/usr/bin/env python3
"""
Package the NBA model's output as static JSON for the phone app.

Writes, into --out (default: <nba folder>/app/data):
  meta.json          season, game days, last update, season record, labels
  day_<date>.json    every game that day, each team's expected rotation, and the model formula
                     fitted before that day (so the phone can recompute when you mark a player in/out)
  power.json         power rankings
  players.json       player ratings, trends, sparklines
  history.json       weekly rating history per player (loaded when you open one)

Set APP_PASSWORD to encrypt everything (the hosted build does).

Usage: python export_app.py [--out DIR] [--ahead 14]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "shared"))
import appdata  # shared/appdata.py in the repo (a copy next to this file takes precedence)
try:
    import livelog
    import matchup
except ImportError:   # older local copies without shared/: no matchup section or live record
    livelog = matchup = None
_LOG = {}


def track(out):
    """Save upcoming predictions; show played games with the prediction saved before them."""
    return livelog.track(_LOG.get("l"), out) if livelog is not None else out
import pipeline as PL
import ratings as R
from features import FEATURE_LABELS
from paths import ROOT, season_label, today

SWITCH = {"lineup": {"pm": "d_pm", "box": "d_box", "delta": "d_lineup_delta", "box_w": 0.5}}


def _short(d):
    t = pd.Timestamp(d)
    return f"{t.strftime('%b')} {t.day}"


def _f(x, d=3):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, d)
    except (TypeError, ValueError):
        return None


def last_name(n):
    parts = [p for p in str(n).split(" ") if p.rstrip(".").upper() not in ("JR", "SR", "II", "III", "IV")]
    return parts[-1] if parts else str(n)


def ml_text(home, away, hml, aml):
    h, a = _f(hml, 0), _f(aml, 0)
    if h is None or a is None:
        return None
    fav, v = (home, h) if h <= a else (away, a)
    return f"{fav} {int(v):+d}"


def impact_weights(model):
    """Points per 48 minutes a player adds, from the model's lineup coefficients."""
    sc, rg = model.pipe.named_steps["standardscaler"], model.pipe.named_steps["ridge"]
    c = dict(zip(model.feats, rg.coef_ / sc.scale_))
    w_pm = c["d_pm"] + c["d_lineup_delta"]
    w_box = c["d_box"] + SWITCH["lineup"]["box_w"] * c["d_lineup_delta"]
    return w_pm, w_box     # a player on the floor all 48 minutes has a share of 1 (a team's shares add up to 5)


def records(df):
    """Each team's W-L going into each game, this season."""
    out = {}
    w, l, season = {}, {}, None
    for g in df.sort_values(["date", "game_id"]).itertuples(index=False):
        if g.season != season:
            w, l, season = {}, {}, g.season
        out[(g.game_id, g.home)] = f"{w.get(g.home, 0)}-{l.get(g.home, 0)}"
        out[(g.game_id, g.away)] = f"{w.get(g.away, 0)}-{l.get(g.away, 0)}"
        if g.played and g.season_type == 2:
            win, lose = (g.home, g.away) if g.margin > 0 else (g.away, g.home)
            w[win] = w.get(win, 0) + 1
            l[lose] = l.get(lose, 0) + 1
    return out


NBA_STATS = [
    {"key": "ppg", "label": "Points per game", "num": "pts", "den": "gp", "better": "high"},
    {"key": "opp", "label": "Points allowed", "num": "opp_pts", "den": "gp", "better": "low"},
    {"key": "net", "label": "Net rating (per 100)", "num": "diff", "den": "poss", "scale": 100, "fmt": "{:+.1f}", "better": "high"},
    {"key": "pace", "label": "Pace (possessions)", "num": "poss", "den": "gp", "better": None},
    {"key": "efg", "label": "Effective FG%", "num": "efg_m", "den": "fga", "scale": 100, "better": "high"},
    {"key": "tp", "label": "3-point %", "num": "tpm", "den": "tpa", "scale": 100, "better": "high"},
    {"key": "oefg", "label": "Opponents' effective FG%", "num": "opp_efg_m", "den": "opp_fga", "scale": 100, "better": "low"},
    {"key": "reb", "label": "Rebounds", "num": "reb", "den": "gp", "better": "high"},
    {"key": "ast", "label": "Assists", "num": "ast", "den": "gp", "better": "high"},
    {"key": "tov", "label": "Turnovers", "num": "tov", "den": "gp", "better": "low"},
]


def build_matchups(E):
    """Head-to-head, form, records and stat comparisons (shared/matchup.py)."""
    if matchup is None:
        return None
    b = E.box
    t = b.groupby(["game_id", "team"]).agg(pts=("pts", "sum"), fgm=("fgm", "sum"), fga=("fga", "sum"), tpm=("tpm", "sum"),
                                           tpa=("tpa", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"), dreb=("dreb", "sum"),
                                           ast=("ast", "sum"), tov=("tov", "sum"), mins=("min", "sum")).reset_index()
    t["reb"] = t.oreb + t.dreb
    t["efg_m"] = t.fgm + 0.5 * t.tpm
    t["own_poss"] = t.fga - t.oreb + t.tov + 0.44 * t.fta
    o = t[["game_id", "team", "pts", "efg_m", "fga", "own_poss"]].rename(
        columns={"team": "opp_team", "pts": "opp_pts", "efg_m": "opp_efg_m", "fga": "opp_fga", "own_poss": "opp_poss"})
    t = t.merge(o, on="game_id")
    t = t[t.team != t.opp_team].copy()
    t["poss"] = 0.5 * (t.own_poss + t.opp_poss)
    t["diff"] = t.pts - t.opp_pts
    ot = t.groupby("game_id").mins.max() > 245          # more than 48 minutes x 5 players
    df = E.df
    g = pd.DataFrame({"game_id": df.game_id, "date": df.date, "season": df.season, "home": df.home, "away": df.away,
                      "home_score": df.home_score, "away_score": df.away_score, "played": df.played,
                      "regular": df.season_type == 2, "extra": df.game_id.map(ot).map({True: "OT"})})
    return matchup.Matchups(g, t, NBA_STATS, season_label)


class Exporter:
    def __init__(self, E):
        self.E = E
        self.df = E.df
        self.rec = records(self.df[self.df.season == E.season])
        self.latest = E.fits[max(E.fits)].set_index("player_id") if E.fits else pd.DataFrame()
        self.names = E.box.drop_duplicates("player_id", keep="last").set_index("player_id")[["name", "pos", "team"]]
        self.mu = build_matchups(E)

    def lineup(self, g, side):
        """Expected rotation (on) plus bench / injured players (off) for an upcoming game."""
        team = getattr(g, side)
        pool = self.E.pool
        s = pool[(pool.game_id == g.game_id) & (pool.team == team)].sort_values("proj", ascending=False)
        rat = self.latest
        out = []
        for r in s.itertuples(index=False):
            if not r.in_rot and r.on and r.proj < 4:
                continue
            pm = rat.pm.get(r.player_id, 0.0) if len(rat) else 0.0
            bx = rat.box.get(r.player_id, np.nan) if len(rat) else np.nan
            out.append({"id": str(r.player_id), "name": r.name, "pos": r.pos, "min": round(max(float(r.proj), 1.0), 2),
                        "pm": round(float(0.0 if pd.isna(pm) else pm), 4),
                        "box": round(float(R.P.box_repl * 36.0 * 0.8 if pd.isna(bx) else bx), 4),
                        "on": bool(r.in_rot), "status": r.status if isinstance(r.status, str) else None,
                        "note": r.note if isinstance(r.note, str) else None})
        return out[:16]

    def game(self, g, model, cw):
        x = pd.DataFrame([g._asdict()])
        margin = float(model.predict_margin(x)[0])
        prob = float(model.prob_from_margin(margin))
        sc, rg = model.pipe.named_steps["standardscaler"], model.pipe.named_steps["ridge"]
        xv = x[model.feats].astype(float).fillna(0.0).values[0]
        contrib = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "value": _f(v, 4),
                    "points": round(float(c * (v - m) / s), 3)}
                   for f, v, c, m, s in zip(model.feats, xv, rg.coef_, sc.mean_, sc.scale_)]
        mp = _f(g.market_home_prob, 4)
        vt = g.details if isinstance(g.details, str) and g.details else ml_text(g.home, g.away, g.home_ml, g.away_ml)
        out = {"id": str(g.game_id), "time": g.start_utc, "away": g.away, "home": g.home, "neutral": bool(g.neutral),
               "played": bool(g.played), "final_week": False, "prob": round(prob, 4), "margin": round(margin, 2),
               "vegas_prob": mp, "vegas_text": vt, "edge": None if mp is None else round(prob - mp, 4),
               "pick": g.home if prob >= 0.5 else g.away, "base": round(float(rg.intercept_), 3), "contributions": contrib,
               "note": g.note if isinstance(g.note, str) else None,
               "home_badges": ["B2B"] if g.rest_h <= 1 else [], "away_badges": ["B2B"] if g.rest_a <= 1 else []}
        for side in ("home", "away"):
            team = getattr(g, side)
            rec = self.rec.get((g.game_id, team), "")
            label = rec
            if not g.played:
                lu = self.lineup(g, side)
                out[f"{side}_lineup"] = lu
                out[f"{side}_typ"] = {"pm": _f(getattr(g, f"pm_typ_{side[0]}"), 4), "box": _f(getattr(g, f"box_typ_{side[0]}"), 4)}
                outs = [p for p in lu if not p["on"] and p["status"] and p["min"] >= 18]
                if outs:
                    label = "w/o " + ", ".join(last_name(p["name"]) for p in outs[:2]) + ("…" if len(outs) > 2 else "")
            out[f"{side}_starter"] = {"name": label, "rating": None, "new": False, "flag": None, "note": None}
        if self.mu is not None:
            out["matchup"] = self.mu.for_game(g.home, g.away, g.season, g.date)
        if g.played:
            hw = g.margin > 0
            out.update({"home_score": int(g.home_score), "away_score": int(g.away_score),
                        "winner": g.home if hw else g.away, "model_correct": bool((prob >= 0.5) == hw),
                        "vegas_correct": None if mp is None else bool((mp >= 0.5) == hw)})
        return out

    def day_file(self, day):
        sub = self.df[self.df.date == pd.Timestamp(day)]
        if sub.empty:
            return None
        model = self.E.fit_before(pd.Timestamp(day))
        cw = impact_weights(model)
        games = [track(self.game(g, model, cw)) for g in sub.sort_values(["start_utc", "game_id"]).itertuples(index=False)]
        return {"key": day, "model": appdata.model_params(model, FEATURE_LABELS, "logistic", SWITCH), "games": games}


def season_record(ex, season):
    d = ex.df[(ex.df.season == season) & ex.df.played].copy()
    if d.empty:
        return {"rows": [], "total": {"n": 0, "model_right": 0, "vegas_n": 0, "vegas_right": 0, "model_ll": None}}
    ps = []
    for day, sub in d.groupby("date"):
        ps.append(pd.Series(ex.E.fit_before(day).predict_proba(sub), index=sub.index))
    d["p"] = pd.concat(ps)
    d["right"] = (d.p >= 0.5) == (d.margin > 0)
    d["ll"] = -np.where(d.margin > 0, np.log(d.p.clip(1e-4)), np.log((1 - d.p).clip(1e-4)))
    d["vn"] = d.market_home_prob.notna()
    d["vr"] = d.vn & ((d.market_home_prob >= 0.5) == (d.margin > 0))
    d["vll"] = -np.where(d.margin > 0, np.log(d.market_home_prob.clip(1e-4)), np.log((1 - d.market_home_prob).clip(1e-4)))
    d["wk"] = d.date - pd.to_timedelta(d.date.dt.weekday, unit="D")
    rows = []
    for wk, s in d.groupby("wk"):
        rows.append({"label": f"Week of {_short(wk)}", "n": int(len(s)), "model_right": int(s.right.sum()),
                     "vegas_n": int(s.vn.sum()), "vegas_right": int(s.vr.sum()), "model_ll": round(float(s.ll.mean()), 4)})
    v = d[d.vn]
    total = {"n": int(len(d)), "model_right": int(d.right.sum()), "vegas_n": int(len(v)), "vegas_right": int(v.vr.sum()),
             "model_ll": round(float(d.ll.mean()), 4), "vegas_ll": round(float(v.vll.mean()), 4) if len(v) else None}
    return {"rows": rows[::-1], "total": total}


def power(ex):
    E = ex.E
    tday = pd.Timestamp(today())
    model = E.fit_before(tday + pd.Timedelta(days=1))
    up = ex.df[~ex.df.played & (ex.df.date >= tday)].sort_values("date")
    rows = []
    for side in ("h", "a"):
        team = up.home if side == "h" else up.away
        rows.append(pd.DataFrame({"team": team.values, "date": up.date.values, "elo": up[f"elo_{side}"].values,
                                  "net": up[f"net_{side}"].values, "pm": up[f"pm_{side}"].values, "box": up[f"box_{side}"].values,
                                  "pm_typ": up[f"pm_typ_{side}"].values, "box_typ": up[f"box_typ_{side}"].values}))
    t = pd.concat(rows).sort_values("date").drop_duplicates("team").set_index("team")
    avg = t[["elo", "net", "pm", "box"]].mean()
    X = pd.DataFrame({"d_elo": t.elo - avg.elo, "d_net": t.net - avg.net, "d_pm": t.pm - avg.pm, "d_box": t.box - avg.box,
                      "d_lineup_delta": (t.pm - t.pm_typ) + 0.5 * (t.box - t.box_typ), "d_b2b": 0.0, "rest_diff": 0.0,
                      "home_field": 0.0}, index=t.index)
    base = float(model.predict_margin(X.iloc[[0]].assign(**{c: 0.0 for c in X.columns}))[0])
    t["power"] = model.predict_margin(X) - base
    t = t.sort_values("power", ascending=False)
    cur = ex.df[(ex.df.season == E.season)]
    last_rec = {}
    for g in cur[cur.played].itertuples(index=False):
        pass
    recs = records(cur.assign(date=cur.date))
    # record through the last played game
    played = cur[cur.played].sort_values("date")
    w, l = {}, {}
    for g in played.itertuples(index=False):
        if g.season_type != 2:
            continue
        win, lose = (g.home, g.away) if g.margin > 0 else (g.away, g.home)
        w[win] = w.get(win, 0) + 1
        l[lose] = l.get(lose, 0) + 1
    pool = E.pool
    outs = pool[~pool.on & (pool.proj >= 18)].drop_duplicates("player_id")
    teams = []
    for i, (team, r) in enumerate(t.iterrows(), 1):
        o = outs[outs.team == team]
        teams.append({"rank": i, "team": team, "power": round(float(r.power), 2),
                      "sub": f"{w.get(team, 0)}-{l.get(team, 0)}" + (" · w/o " + ", ".join(last_name(n) for n in o.name[:3]) if len(o) else "")})
    return {"about": "Points better than an average team on a neutral floor, with tonight's expected lineup (injuries included).",
            "unit": "pts", "teams": teams}


def players_files(ex):
    E = ex.E
    model = E.fit_before(pd.Timestamp(today()) + pd.Timedelta(days=1))
    wpm, wbox = impact_weights(model)
    cuts = sorted(E.fits)
    recent_cuts = [c for c in cuts if c >= cuts[-1] - pd.Timedelta(days=2 * 365)]
    series = []
    for c in recent_cuts:
        f = E.fits[c]
        series.append(pd.DataFrame({"player_id": f.player_id, "cut": c, "imp": wpm * f.pm + wbox * f.box.fillna(0), "box": f.box}))
    s = pd.concat(series, ignore_index=True)
    cur = E.fits[cuts[-1]].set_index("player_id")
    box = E.box
    season_min = box[box.season == E.season].groupby("player_id")["min"].agg(["sum", "count"])
    last_season = box[box.season >= E.season - 1].groupby("player_id")["min"].sum()
    keep = last_season[last_season >= 400].index.union(season_min[season_min["sum"] >= 60].index)
    last = E.last
    rows, hist = [], {}
    piv = s[s.player_id.isin(keep)].pivot(index="player_id", columns="cut", values="imp")
    for pid in keep:
        if pid not in cur.index or pid not in ex.names.index:
            continue
        nm = ex.names.loc[pid]
        imp = float(wpm * cur.pm.get(pid, 0) + wbox * (cur.box.get(pid, 0) if pd.notna(cur.box.get(pid, np.nan)) else 0))
        sp = piv.loc[pid].dropna().values[-12:] if pid in piv.index else []
        tr = float(sp[-1] - sp[-6]) if len(sp) >= 6 else None
        pos = nm.pos if isinstance(nm.pos, str) else "F"
        grp = pos[-1] if pos in ("PG", "SG", "SF", "PF") else pos
        grp = grp if grp in ("G", "F", "C") else "F"
        sm = season_min.loc[pid] if pid in season_min.index else None
        mpg = (sm["sum"] / sm["count"]) if sm is not None and sm["count"] else None
        team = last.team.get(pid, nm.team)
        rows.append({"pid": str(pid), "name": nm["name"], "pos": pos, "g": grp, "team": team, "imp": round(imp, 2),
                     "tr": None if tr is None else round(tr, 2), "prod": _f(cur.box.get(pid), 2),
                     "sub": f"{pos} · {team} · " + (f"{mpg:.0f} min · {int(sm['count'])} GP" if mpg else "no games this season"),
                     "sp": [round(float(v), 2) for v in sp]})
        hs = s[s.player_id == pid]
        hist[str(pid)] = {"imp": [[c.strftime("%Y-%m-%d"), round(float(v), 2)] for c, v in zip(hs.cut, hs.imp)],
                          "prod": [[c.strftime("%Y-%m-%d"), round(float(v), 2)] for c, v in zip(hs.cut, hs.box) if pd.notna(v)]}
    bp = box[box.player_id.isin(keep)].sort_values("date").groupby("player_id").tail(8)
    opp = np.where(bp.is_home, bp.away, bp.home)
    for (pid, sub), _ in zip(bp.assign(opp=opp).groupby("player_id"), range(10 ** 6)):
        hist.setdefault(str(pid), {})["recent"] = [
            [_short(r.date), f"{'vs' if r.is_home else '@'} {r.opp}", f"{r.min:.0f} min · {r.pts:.0f} pts {r.oreb + r.dreb:.0f} reb {r.ast:.0f} ast · {r.pm:+.0f}"]
            for r in sub.itertuples(index=False)][::-1]
    reg = [r for r in rows if r["tr"] is not None and "GP" in r["sub"] and int(r["sub"].split(" GP")[0].split("· ")[-1]) >= 5]
    movers = {}
    for grp in ("G", "F", "C", "ALL"):
        rg = [r for r in reg if grp == "ALL" or r["g"] == grp]
        movers[grp] = {"up": [r["pid"] for r in sorted(rg, key=lambda r: -r["tr"])[:5]],
                       "down": [r["pid"] for r in sorted(rg, key=lambda r: r["tr"])[:5]]}
    return {"season": E.season, "players": rows, "movers": movers,
            "groups": [{"key": "ALL", "label": "All"}, {"key": "G", "label": "Guards"}, {"key": "F", "label": "Forwards"},
                       {"key": "C", "label": "Centers"}],
            "default_group": "ALL", "prod_groups": ["G", "F", "C", "ALL"],
            "labels": {"imp": "Impact", "prod": "Box score"}, "unit": "pts/48",
            "about": "Impact: points per 48 minutes a player adds vs. an average player, from plus-minus and box score, refit weekly. "
                     "Box score: his game score per 36 minutes vs. average.",
            "detail_note": "Plus-minus is noisy: teammates who always share the floor split credit. Read the trend more than the decimals."}, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "app" / "data"))
    ap.add_argument("--ahead", type=int, default=14, help="days of upcoming games to include")
    args = ap.parse_args()
    t0 = time.time()
    W = appdata.Writer(args.out)
    E = PL.ENGINE.ensure()
    if livelog is not None:
        _LOG["l"] = livelog.LiveLog(PL.DATA, "nba")
    ex = Exporter(E)
    df = E.df
    tday = pd.Timestamp(today())
    cur = df[df.season == E.season]
    days = sorted(d for d in cur.date.unique() if pd.Timestamp(d) <= tday + pd.Timedelta(days=args.ahead))
    periods = []
    for d in days:
        d = pd.Timestamp(d)
        key = d.strftime("%Y-%m-%d")
        p = ex.day_file(key)
        if p is None:
            continue
        name = W.write(f"day_{key}.json", p)
        played = bool(cur[cur.date == d].played.all())
        periods.append({"key": key, "label": d.strftime("%a, %b ") + str(d.day), "top": d.strftime("%a").upper(),
                        "bottom": _short(d), "played": played, "file": name})
    upcoming = [p for p in periods if pd.Timestamp(p["key"]) >= tday]
    default = upcoming[0]["key"] if upcoming else (periods[-1]["key"] if periods else None)
    record = season_record(ex, E.season)
    if _LOG.get("l") is not None:
        record["live"] = livelog.live_record(_LOG["l"])
        W.write("live_log.json", _LOG["l"].payload())
        _LOG["l"].save()
    W.write("power.json", power(ex))
    players, hist = players_files(ex)
    W.write("players.json", players)
    W.write("history.json", hist)
    lp = cur[cur.played].date.max()
    W.write("meta.json", {
        "league": "nba", "name": "NBA", "season": E.season, "season_label": season_label(E.season),
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "periods": periods, "default_period": default, "period_word": "day",
        "through": f"thru {_short(lp)}" if pd.notna(lp) else "",
        "injury_period": None,
        "units": {"margin": "pts", "starter": "Lineup", "starter_plural": "Lineups", "rating_unit": "pts/48"},
        "starter_help": "Based on the latest injury report. Tap a player to mark him in or out; the prediction updates instantly.",
        "injury_pending": None, "record": record, "model": "v1",
    })
    W.finish()
    print(f"[app] NBA: wrote {W.summary()} to {W.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
