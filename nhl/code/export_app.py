#!/usr/bin/env python3
"""
Package the NHL model's output as static JSON for the phone app.

Writes, into --out (default: <NHL folder>/app/data):
  meta.json          season, game days, last update, season record, labels
  day_<date>.json    every game that day plus the model formula fitted before it
                     (so the phone can recompute instantly when you switch a goalie)
  power.json         power rankings
  players.json       skater and goalie ratings, trends, sparklines
  history.json       rating history per player (loaded when you open one)

Set APP_PASSWORD to encrypt everything (the hosted build does).

Usage: python export_app.py [--out DIR] [--ahead 14]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "shared"))
import appdata  # shared/appdata.py in the repo (a copy next to this file takes precedence)
import features as F
import hud_data as H
from paths import DATA, ROOT, today

try:
    import livelog
    import matchup
except ImportError:   # older local copies without shared/: no matchup section or live record
    livelog = matchup = None
_LOG = {}


def track(out):
    """Save upcoming predictions; show played games with the prediction saved before them."""
    return livelog.track(_LOG.get("l"), out) if livelog is not None else out

NHL_STATS = [
    {"key": "gf", "label": "Goals per game", "num": "gf", "den": "gp", "fmt": "{:.2f}", "better": "high"},
    {"key": "ga", "label": "Goals against", "num": "ga", "den": "gp", "fmt": "{:.2f}", "better": "low"},
    {"key": "xg", "label": "5-on-5 expected goals share", "num": "ev_xgf", "den": "ev_xgt", "scale": 100, "fmt": "{:.1f}%", "better": "high"},
    {"key": "sf", "label": "Shots per game", "num": "sf", "den": "gp", "better": "high"},
    {"key": "sa", "label": "Shots against", "num": "sa", "den": "gp", "better": "low"},
    {"key": "pp", "label": "Power play (goals per 60 min)", "num": "ppg", "den": "pp_min", "scale": 60, "fmt": "{:.1f}", "better": "high"},
    {"key": "pk", "label": "Penalty kill (goals against per 60)", "num": "pkga", "den": "pk_min", "scale": 60, "fmt": "{:.1f}", "better": "low"},
    {"key": "sh", "label": "Shooting %", "num": "gf_sog", "den": "sf", "scale": 100, "better": "high"},
    {"key": "sv", "label": "Save %", "num": "saves", "den": "sa_ing", "scale": 1000, "fmt": ".{:03.0f}", "better": "high"},
]


def build_matchups():
    """Head-to-head, form, records and stat comparisons (shared/matchup.py)."""
    if matchup is None:
        return None
    E = H.ENGINE
    df = F.load_games(DATA)
    season = int(E.season)
    shots = F.load_kind(DATA, "shots", [season - 1, season])
    states = F.load_kind(DATA, "states", [season - 1, season])
    sog = shots[shots.ev.isin(["S", "G"])].assign(one=1.0, goal=lambda d: (d.ev == "G").astype(float))
    by = lambda d, col, v: d.groupby(["game_id", col])[v].sum().rename_axis(["game_id", "team"])
    ing = sog[sog.opp_g == 1]
    pp = sog[(sog.strength == "PP")]
    t = pd.concat({"sf": by(sog, "team", "one"), "sa": by(sog, "opp", "one"), "gf_sog": by(sog, "team", "goal"),
                   "saves": by(ing.assign(sv=1.0 - ing.goal), "opp", "sv"), "sa_ing": by(ing, "opp", "one"),
                   "ppg": by(pp, "team", "goal"), "pkga": by(pp, "opp", "goal")}, axis=1).fillna(0.0)
    st = states.set_index(["game_id", "team"])
    t = t.join((st[["toi_pp", "toi_pk"]] / 60.0).rename(columns={"toi_pp": "pp_min", "toi_pk": "pk_min"}), how="outer").fillna(0.0)
    ts = E.pl.get("tstats")
    if ts is not None:
        x = ts.set_index(["game_id", "team"])[["ev_xgf", "ev_xga"]]
        t = t.join(x, how="left")
        t["ev_xgt"] = t.ev_xgf + t.ev_xga
    else:
        t["ev_xgf"], t["ev_xgt"] = 0.0, 0.0
    # goals for/against from final scores, without the shootout's extra goal
    p = df[df.played & (df.season >= season - 1)]
    so = p.last_period.eq("SO")
    hs = p.home_score - (so & (p.home_score > p.away_score)).astype(float)
    as_ = p.away_score - (so & (p.away_score > p.home_score)).astype(float)
    sc = pd.concat([pd.DataFrame({"game_id": p.game_id, "team": p.home_team, "gf": hs, "ga": as_}),
                    pd.DataFrame({"game_id": p.game_id, "team": p.away_team, "gf": as_, "ga": hs})])
    t = sc.merge(t.reset_index(), on=["game_id", "team"], how="left").fillna(0.0)
    g = pd.DataFrame({"game_id": df.game_id, "date": df.gameday, "season": df.season, "home": df.home_team,
                      "away": df.away_team, "home_score": df.home_score, "away_score": df.away_score, "played": df.played,
                      "regular": df.game_type.eq("R"), "extra": df.last_period.where(df.last_period.isin(["OT", "SO"]))})
    return matchup.Matchups(g, t, NHL_STATS, lambda s: f"{s}-{str(s + 1)[-2:]}", otl=True)


_MU = {}


def ml_text(home, away, hml, aml):
    if not hml or not aml:
        return None
    try:
        h, a = float(hml), float(aml)
    except ValueError:
        return None
    return f"{home} {hml}" if h <= a else f"{away} {aml}"


def _short(d):
    t = pd.Timestamp(d)
    return f"{t.strftime('%b')} {t.day}"


def starter(gl):
    out = {"name": gl.get("name"), "rating": gl.get("rating"), "new": False, "flag": None, "note": gl.get("note")}
    if gl.get("options"):
        out["options"] = [{"id": str(o["pid"]), "name": o["name"], "rating": o["rating"], "new": False,
                           "available": o["available"], "status": o["status"],
                           "note": f"{o['starts10']} of last 10 starts" if o.get("starts10") is not None else None,
                           "selected": o["selected"]} for o in gl["options"]]
    return out


def day_file(day):
    p = H.day_payload(day)
    if not p["games"]:
        return None
    model = H.ENGINE.fit_before(pd.Timestamp(day))
    games = []
    for g in p["games"]:
        out = {"id": g["game_id"], "time": g["start_utc"], "away": g["away"], "home": g["home"], "neutral": g["neutral"],
               "played": g["played"], "final_week": False, "prob": g["model_prob"], "margin": g["model_margin"],
               "vegas_prob": g["market_prob"], "vegas_text": ml_text(g["home"], g["away"], g["home_ml"], g["away_ml"]),
               "edge": g["edge"], "pick": g["pick"], "base": g["base_points"], "contributions": g["contributions"],
               "home_starter": starter(g["home_goalie"]), "away_starter": starter(g["away_goalie"]),
               "home_injury": g["home_injury"], "away_injury": g["away_injury"],
               "home_abs": (g["home_lineup"] or {}).get("absences", []), "away_abs": (g["away_lineup"] or {}).get("absences", []),
               "home_badges": ["B2B"] if g["home_b2b"] else [], "away_badges": ["B2B"] if g["away_b2b"] else []}
        for side in ("home", "away"):
            for a in out[f"{side}_abs"]:
                a["why"] = a.get("why") or "out"
        if _MU.get("m") is not None:
            out["matchup"] = _MU["m"].for_game(g["home"], g["away"], H.ENGINE.season, pd.Timestamp(day))
        if g["played"]:
            out.update({"home_score": g.get("home_score"), "away_score": g.get("away_score"),
                        "score_note": g.get("last_period"), "winner": g.get("winner"),
                        "model_correct": g.get("model_correct"), "vegas_correct": g.get("market_correct")})
        games.append(track(out))
    return {"key": p["date"], "model": appdata.model_params(model, H.FEATURE_LABELS, "logistic", {"rating": "d_goalie"}),
            "games": games}


def players_files():
    groups = {"F": 350, "D": 220}
    rows, movers, pids = [], {}, []
    for grp, lim in groups.items():
        p = H.players_payload(grp, limit=lim)
        for r in p["players"]:
            rows.append({"pid": str(r["pid"]), "name": r["name"], "pos": r["pos"], "g": grp, "team": r.get("team"),
                         "imp": r.get("impact"), "tr": r.get("trend"), "prod": r.get("pts60"),
                         "sub": f"{r['pos']} · {r.get('team') or ''} · {r.get('toi') or 0:.1f} min · {r.get('games') or 0} GP",
                         "sp": r.get("spark") or []})
            pids.append(r["pid"])
        movers[grp] = {"up": [str(x["pid"]) for x in p["risers"]], "down": [str(x["pid"]) for x in p["fallers"]]}
    allp = H.players_payload(None, limit=1)
    movers["ALL"] = {"up": [str(x["pid"]) for x in allp["risers"]], "down": [str(x["pid"]) for x in allp["fallers"]]}
    gp = H.players_payload("G", limit=90)
    for r in gp["players"]:
        sp = r.get("spark") or []
        tr = round(sp[-1] - sp[-6], 3) if len(sp) >= 6 else None
        sv = r.get("fen_sv")
        rows.append({"pid": str(r["pid"]), "name": r["name"], "pos": "G", "g": "G", "team": r.get("team"),
                     "imp": r.get("rating"), "tr": tr, "prod": r.get("gsax"),
                     "sub": f"G · {r.get('team') or ''} · {r.get('starts') or 0} starts" + (f" · {sv:.3f}".replace("0.", ".", 1) + " sv% (unblocked)" if sv else ""),
                     "sp": sp})
        pids.append(r["pid"])
    gs = [x for x in rows if x["g"] == "G" and x["tr"] is not None and x["imp"] is not None]
    gs_reg = [x for x in gs if "starts" in x["sub"]]
    movers["G"] = {"up": [x["pid"] for x in sorted(gs_reg, key=lambda x: -x["tr"])[:5]],
                   "down": [x["pid"] for x in sorted(gs_reg, key=lambda x: x["tr"])[:5]]}

    hist = {}
    for pid in pids:
        d = H.player_detail(pid)
        if d["kind"] == "goalie":
            imp = [[s["k"], s["v"]] for s in d["series"]]
            rec = [[_short(s["date"]), f"vs {s['opp']}", f"{s['ga']} GA on {s['shots']} · {s['gsax']:+.1f} GSAx"] for s in d["games"]]
        else:
            imp = [[s["k"], s["impact"]] for s in d["series"]]
            rec = [[_short(s["date"]), f"vs {s['opp']}", f"{s['toi']} min · {s['g']}G {s['a']}A"] for s in d["games"]]
        hist[str(pid)] = {"imp": imp, "recent": rec}
    return {"season": H.ENGINE.season, "players": rows, "movers": movers,
            "groups": [{"key": "F", "label": "Forwards"}, {"key": "D", "label": "Defense"}, {"key": "G", "label": "Goalies"},
                       {"key": "ALL", "label": "All skaters"}],
            "default_group": "F", "all_excludes": ["G"], "prod_groups": ["F", "D", "G"],
            "labels": {"imp": "Impact", "prod": "Pts/60"}, "group_labels": {"G": {"imp": "Rating", "prod": "GSAx"}},
            "unit": "xG/60", "units": {"G": "goals/g"},
            "about": "Impact: expected goals per 60 minutes a skater adds at 5-on-5 (offense plus defense) vs. an average skater, refit weekly. "
                     "Goalies: goals saved above expected per game.",
            "detail_note": "Linemates who always play together split credit, so read the trend rather than the decimals."}, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "app" / "data"))
    ap.add_argument("--ahead", type=int, default=14, help="days of upcoming games to include")
    args = ap.parse_args()
    t0 = time.time()
    W = appdata.Writer(args.out)
    E = H.ENGINE
    E.ensure()
    if getattr(E, "error", None):
        raise SystemExit(E.error)
    if livelog is not None:
        _LOG["l"] = livelog.LiveLog(DATA, "nhl")
    try:
        _MU["m"] = build_matchups()
    except Exception as e:  # noqa: BLE001  (the matchup section is extra; never let it stop the build)
        print(f"[app] matchups skipped: {type(e).__name__}: {e}")
    st = H.status({})
    last_day = pd.Timestamp(today()) + pd.Timedelta(days=args.ahead)
    periods = []
    for d in st["days"]:
        day = pd.Timestamp(d["date"])
        if day > last_day:
            continue
        p = day_file(d["date"])
        if p is None:
            continue
        name = W.write(f"day_{d['date']}.json", p)
        periods.append({"key": d["date"], "label": day.strftime("%a, %b ") + str(day.day),
                        "top": day.strftime("%a").upper(), "bottom": f"{day.strftime('%b')} {day.day}",
                        "played": d["played"], "file": name})
    default = st["default_date"]
    if periods and default not in {p["key"] for p in periods}:
        default = periods[-1]["key"]
    rec = H.season_record(E.season)
    record = {"rows": [{"label": "Week of " + pd.Timestamp(r["week"]).strftime("%b ") + str(pd.Timestamp(r["week"]).day),
                        "n": r["n"], "model_right": r["model_right"], "vegas_n": r["n_line"],
                        "vegas_right": r["vegas_right"], "model_ll": r["model_ll"]} for r in rec["weeks"]],
              "total": {"n": rec["total"].get("n", 0), "model_right": rec["total"].get("model_right", 0),
                        "vegas_n": rec["total"].get("n_line", 0), "vegas_right": rec["total"].get("vegas_right", 0),
                        "model_ll": rec["total"].get("model_ll")}}
    inj_log = DATA / "injury_log.csv"
    if inj_log.exists():   # encrypted copy, so the log survives losing the build cache
        W.write("injury_log.json", {"rows": pd.read_csv(inj_log).to_dict("records")})
    if _LOG.get("l") is not None:
        record["live"] = livelog.live_record(_LOG["l"])
        W.write("live_log.json", _LOG["l"].payload())
        _LOG["l"].save()
    pw = H.rankings()
    W.write("power.json", {"about": "Goals per game better than an average team, with each team's No. 1 goalie.",
                           "unit": "goals", "teams": [{"rank": t["rank"], "team": t["team"], "power": t["power"],
                                                       "sub": f"{t['record']} · {t['points']} pts · {t.get('goalie') or 'G ?'}"
                                                              + (f" ({t['goalie_rating']:+.2f})" if t.get("goalie_rating") is not None else "")}
                                                      for t in pw["teams"]]})
    players, hist = players_files()
    W.write("players.json", players)
    W.write("history.json", hist)
    lp = pd.Timestamp(st["last_played"]) if st.get("last_played") else None
    W.write("meta.json", {
        "league": "nhl", "name": "NHL", "season": E.season, "season_label": st.get("label"),
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "periods": periods, "default_period": default, "period_word": "day",
        "through": f"thru {lp.strftime('%b')} {lp.day}" if lp is not None else "",
        "injury_period": None,
        "units": {"margin": "goals", "starter": "Goalie", "starter_plural": "Goalies", "rating_unit": "goals/g"},
        "starter_help": "Projected until teams confirm starters on game day (usually after the morning skate). Switch if a different goalie is starting; the prediction updates instantly.",
        "injury_pending": None, "record": record, "model": "v1",
    })
    W.finish()
    print(f"[app] NHL: wrote {W.summary()} to {W.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
