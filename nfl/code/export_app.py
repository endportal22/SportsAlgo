#!/usr/bin/env python3
"""
Package the NFL model's output as static JSON for the phone app.

Writes, into --out (default: <NFL folder>/app/data):
  meta.json            season, weeks, last update, season record, labels
  week_<season>_<wk>.json  every game of a week plus that week's model formula
                       (so the phone can recompute instantly when you switch a QB)
  power.json           power rankings
  players.json         every regular player: rating, trend, sparkline
  history.json         weekly rating history per player (loaded when you open one)

Set APP_PASSWORD to encrypt everything (the hosted build does).

Usage: python export_app.py [--out DIR]        (run after update_data.py; ~1 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "shared"))
import appdata  # shared/appdata.py in the repo (a copy next to this file takes precedence)
try:
    import evaluate
    import livelog
    import matchup
except ImportError:   # older local copies without shared/: no matchup section, live record or scorecard
    evaluate = livelog = matchup = None
_LOG = {}
_START = {}   # game_id -> kickoff (UTC), for saving pre-game predictions


def track(out):
    """Save upcoming predictions; show played games with the prediction saved before them."""
    return livelog.track(_LOG.get("l"), out) if livelog is not None else out
import hud_data as H
from paths import ROOT

PROD_GROUPS = {"QB", "RB", "WR", "TE"}

NFL_STATS = [
    {"key": "pf", "label": "Points per game", "num": "pts", "den": "gp", "better": "high"},
    {"key": "pa", "label": "Points allowed", "num": "opp_pts", "den": "gp", "better": "low"},
    {"key": "ypp", "label": "Yards per play", "num": "yds", "den": "plays", "better": "high"},
    {"key": "oypp", "label": "Yards per play allowed", "num": "opp_yds", "den": "opp_plays", "better": "low"},
    {"key": "oepa", "label": "Offense (EPA per play)", "num": "epa", "den": "plays", "fmt": "{:+.2f}", "better": "high"},
    {"key": "depa", "label": "Defense (EPA per play allowed)", "num": "opp_epa", "den": "opp_plays", "fmt": "{:+.2f}", "better": "low"},
    {"key": "pepa", "label": "Passing (EPA per dropback)", "num": "p_epa", "den": "drops", "fmt": "{:+.2f}", "better": "high"},
    {"key": "repa", "label": "Rushing (EPA per carry)", "num": "r_epa", "den": "rushes", "fmt": "{:+.2f}", "better": "high"},
    {"key": "to", "label": "Turnover margin per game", "num": "to_margin", "den": "gp", "fmt": "{:+.1f}", "better": "high"},
    {"key": "third", "label": "3rd-down conversion %", "num": "third_conv", "den": "third_att", "scale": 100, "better": "high"},
    {"key": "sacks", "label": "Sacks per game", "num": "sacks", "den": "gp", "better": "high"},
]


def long_run(E):
    """Walk-forward predictions for every game since 2019, scored against Vegas (shared/evaluate.py)."""
    d = E.df[E.df.played & (E.df.season >= 2019) & E.df.home_win.isin([0, 1])]
    parts = []
    for (s, w), sub in d.groupby(["season", "week"]):
        m = E.fit_before(sub.gameday.min())
        vp = [H.vegas_prob(r, m.sigma) for r in sub.itertuples(index=False)]
        parts.append(pd.DataFrame({"p": m.predict_proba(sub), "y": sub.home_win.astype(float).values,
                                   "p_mkt": pd.to_numeric(pd.Series(vp), errors="coerce").values, "block": f"{s}-{w:02d}"}))
    return evaluate.scorecard(pd.concat(parts, ignore_index=True), "2019")


def build_matchups(E):
    """Head-to-head, form, records and stat comparisons (shared/matchup.py)."""
    if matchup is None:
        return None
    import features as F
    games = F.load_games(H.DATA)
    season = int(E.season)
    pbp = F.load_pbp(H.DATA, [season - 1, season])
    for c in ("posteam", "defteam"):
        pbp[c] = pbp[c].replace(F.TEAM_FIX)
    for c in ("interception", "fumble_lost", "third_down_converted", "third_down_failed", "sack"):
        if c not in pbp:
            pbp[c] = pd.NA
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
    sc = pbp[pbp.play_type.isin(["pass", "run"]) & (pbp.qb_kneel.fillna(0) == 0) & (pbp.qb_spike.fillna(0) == 0)].copy()
    sc["drop"] = (sc.qb_dropback.fillna(0) == 1).astype(float)
    sc["rush1"] = 1.0 - sc["drop"]
    sc["p_epa"] = sc.epa.where(sc["drop"] == 1, 0.0)
    sc["r_epa"] = sc.epa.where(sc["rush1"] == 1, 0.0)
    off = sc.groupby(["game_id", "posteam"]).agg(plays=("epa", "size"), yds=("yards_gained", "sum"), epa=("epa", "sum"),
                                                 drops=("drop", "sum"), p_epa=("p_epa", "sum"), rushes=("rush1", "sum"),
                                                 r_epa=("r_epa", "sum"))
    off.index.names = ["game_id", "team"]
    allp = pbp[pbp.posteam.notna()]
    tov = allp.groupby(["game_id", "posteam"]).agg(give=("interception", "sum"), fum=("fumble_lost", "sum"),
                                                   third_conv=("third_down_converted", "sum"), third_fail=("third_down_failed", "sum"))
    tov.index.names = ["game_id", "team"]
    dfn = allp.groupby(["game_id", "defteam"]).agg(sacks=("sack", "sum"))
    dfn.index.names = ["game_id", "team"]
    t = off.join(tov, how="left").join(dfn, how="left").fillna(0.0).reset_index()
    t["giveaways"] = t.give + t.fum
    t["third_att"] = t.third_conv + t.third_fail
    o = t[["game_id", "team", "plays", "yds", "epa", "giveaways"]].rename(
        columns={"team": "opp_team", "plays": "opp_plays", "yds": "opp_yds", "epa": "opp_epa", "giveaways": "takeaways"})
    t = t.merge(o, on="game_id")
    t = t[t.team != t.opp_team].copy()
    t["to_margin"] = t.takeaways - t.giveaways
    p = games[games.played & (games.season >= season - 1)]
    pts = pd.concat([pd.DataFrame({"game_id": p.game_id, "team": p.home_team, "pts": p.home_score, "opp_pts": p.away_score}),
                     pd.DataFrame({"game_id": p.game_id, "team": p.away_team, "pts": p.away_score, "opp_pts": p.home_score})])
    t = pts.merge(t, on=["game_id", "team"], how="left").fillna(0.0)
    g = pd.DataFrame({"game_id": games.game_id, "date": games.gameday, "season": games.season, "home": games.home_team,
                      "away": games.away_team, "home_score": games.home_score, "away_score": games.away_score,
                      "played": games.played, "regular": games.game_type.eq("REG"),
                      "extra": games.overtime.map({1: "OT", 1.0: "OT"})})
    return matchup.Matchups(g, t, NFL_STATS, str, ties=True)


_MU = {}


def spread_text(home, away, s):
    if s is None:
        return None
    if s == 0:
        return "Pick'em"
    return f"{home if s > 0 else away} −{abs(s):.1f}"


def starter(q, return_flag=None):
    out = {"name": q.get("name"), "rating": q.get("rating"), "new": bool(q.get("new")),
           "flag": f"{q['return_flag']} may return from injury" if q.get("return_flag") else None, "note": None}
    if q.get("options"):
        out["options"] = [{"id": o["pfr"], "name": o["name"], "rating": o["rating"], "new": bool(o.get("new")),
                           "available": o["available"], "status": o["status"], "note": o["note"],
                           "selected": o["selected"]} for o in q["options"]]
    return out


def week_file(season, week):
    E = H.ENGINE
    payload = H.week_payload(season, week)
    if not payload["games"]:
        return None
    df = E.df
    wk = df[(df.season == season) & (df.week == week)]
    model = E.fit_before(wk.gameday.min())
    fw = dict(zip(wk.game_id, wk.final_week))
    games = []
    for g in payload["games"]:
        # QB options need the "new starter" flag too, so the phone can recompute both QB features
        for side, team in (("home", g["home"]), ("away", g["away"])):
            pj = E.pl["qb_proj"].get((g["game_id"], team))
            if pj and g[f"{side}_qb"].get("options"):
                newflag = {o["pfr"]: o["new"] for o in pj["options"]}
                for o in g[f"{side}_qb"]["options"]:
                    o["new"] = bool(newflag.get(o["pfr"], 0))
        out = {"id": g["game_id"], "group": g["gameday"], "away": g["away"], "home": g["home"],
               "neutral": g["neutral"], "played": g["played"], "final_week": bool(fw.get(g["game_id"], 0)),
               "prob": g["model_prob"], "margin": g["model_margin"], "vegas_prob": g["vegas_prob"],
               "vegas_text": spread_text(g["home"], g["away"], g["spread"]), "edge": g["edge"], "pick": g["pick"],
               "base": g["base_points"], "contributions": g["contributions"],
               "home_starter": starter(g["home_qb"]), "away_starter": starter(g["away_qb"]),
               "home_injury": g["home_injury"], "away_injury": g["away_injury"],
               "home_abs": (g["home_lineup"] or {}).get("absences", []), "away_abs": (g["away_lineup"] or {}).get("absences", []),
               "home_badges": [], "away_badges": []}
        if _MU.get("m") is not None:
            out["matchup"] = _MU["m"].for_game(g["home"], g["away"], season, wk.loc[wk.game_id == g["game_id"], "gameday"].iloc[0])
        if g["played"]:
            out.update({"home_score": g.get("home_score"), "away_score": g.get("away_score"), "winner": g.get("winner"),
                        "model_correct": g.get("model_correct"), "vegas_correct": g.get("vegas_correct")})
        out["time"] = _START.get(g["game_id"])
        games.append(track(out))
    return {"key": str(week), "model": appdata.model_params(
        model, H.FEATURE_LABELS, "normal",
        {"rating": "d_qb_rating", "new": "d_qb_new", "final_week": "fw_x_qb_rating"}), "games": games}


def players_files(season):
    E = H.ENGINE
    pt = H._players_table()
    t = pt["table"]
    t = t[(t.snap >= 0.2) | (t.games >= 2)].sort_values("impact", ascending=False)
    rows = []
    for r in t.to_dict("records"):
        s = pt["spark"].get(r["pid"])
        prod = r.get("prod")
        sub = f"{r['pos']} · {r['team'] or ''} · {round(r['snap'] * 100)}% snaps"
        rows.append({"pid": r["pid"], "name": r["name"], "pos": r["pos"], "g": r["group"], "team": r["team"],
                     "imp": r["impact"], "tr": r["trend"], "prod": prod if r["group"] in PROD_GROUPS else None,
                     "sub": sub, "sp": [round(float(v), 2) for v in s[r["unit"]].values] if s is not None else []})
    reg = t[(t.snap >= 0.5) & t.trend.notna()]
    movers = {}
    for grp in ["ALL"] + sorted(t.group.unique()):
        g = reg if grp == "ALL" else reg[reg.group == grp]
        movers[grp] = {"up": list(g.nlargest(5, "trend").pid), "down": list(g.nsmallest(5, "trend").pid)}

    pids = set(t.pid)
    hist = E.pl["history"]
    hh = hist[hist.pid.isin(pids) & (hist.season >= season - 2)].sort_values(["season", "week"])
    unit = dict(zip(t.pid, t.unit))
    kinds = dict(zip(t.pid, t.get("prod_kind", [None] * len(t))))
    out = {}
    for pid, g in hh.groupby("pid"):
        u = unit[pid]
        out[pid] = {"imp": [[f"{a}-{b:02d}", round(float(v), 2)] for a, b, v in zip(g.season, g.week, g[u])]}
    ph = pt["pr_hist"]
    ph = ph[ph.season >= season - 2]
    pf2gs = pt["pf2gs"]
    for pid in pids:
        kind, gs = kinds.get(pid), pf2gs.get(pid)
        if isinstance(kind, str) and gs:
            q = ph[(ph.pid == gs) & (ph.kind == kind)]
            out.setdefault(pid, {})["prod"] = [[f"{int(a)}-{int(b):02d}", round(float(v) * H.PROD_SCALE[kind], 2)]
                                               for a, b, v in zip(q.season, q.week, q.rating)]
    sn = H.P.load_snaps(H.DATA, [season - 1, season])
    sn = sn[sn.pfr_player_id.isin(pids)].sort_values(["season", "week"])
    for pid, g in sn.groupby("pfr_player_id"):
        u = unit[pid]
        g = g.tail(8)
        out.setdefault(pid, {})["recent"] = [[f"{int(a)} Wk {int(b)}", f"vs {o}", f"{round(float(x if u == 'off' else y) * 100)}% snaps"]
                                             for a, b, o, x, y in zip(g.season, g.week, g.opponent, g.offense_pct, g.defense_pct)][::-1]
    labels = H.PROD_LABEL
    groups = [{"key": k, "label": k} for k in ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]] + [{"key": "ALL", "label": "All"}]
    players = {"season": season, "players": rows, "movers": movers, "groups": groups, "default_group": "QB",
               "prod_groups": sorted(PROD_GROUPS), "labels": {"imp": "Impact", "prod": "Production"},
               "unit": "pts/g",
               "about": "Impact: points per game a player adds to his unit at full snaps vs. an average player, refit weekly. "
                        "Production (QB/RB/WR/TE): his own EPA per dropback, carry or target.",
               "detail_note": "Linemen who always play together share credit, so read the trend rather than the decimals.",
               "prod_labels": {g: labels.get({"QB": "qb", "RB": "rush", "WR": "rec", "TE": "rec"}[g]) for g in PROD_GROUPS}}
    return players, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "app" / "data"))
    args = ap.parse_args()
    t0 = time.time()
    W = appdata.Writer(args.out)
    E = H.ENGINE
    E.ensure()
    if E.error:
        raise SystemExit(E.error)
    if livelog is not None:
        import features as F
        _LOG["l"] = livelog.LiveLog(H.DATA, "nfl")
        gm = F.load_games(H.DATA)
        for r in gm.itertuples(index=False):
            if isinstance(getattr(r, "gametime", None), str):
                t = pd.Timestamp(f"{pd.Timestamp(r.gameday).date()} {r.gametime}").tz_localize("America/New_York")
                _START[r.game_id] = t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    try:
        _MU["m"] = build_matchups(E)
    except Exception as e:  # noqa: BLE001  (the matchup section is extra; never let it stop the build)
        print(f"[app] matchups skipped: {type(e).__name__}: {e}")
    season = E.season
    st = H.status({})
    periods = []
    for w in st["weeks"]:
        p = week_file(season, w["week"])
        if p is None:
            continue
        name = W.write(f"week_{season}_{w['week']:02d}.json", p)
        short = w["label"] if not w["label"].startswith("Week") else str(w["week"])
        periods.append({"key": str(w["week"]), "label": w["label"], "top": "WK" if w["label"].startswith("Week") else "",
                        "bottom": short, "played": w["played"], "file": name})
    rec = H.season_record(season)
    record = {"rows": [{"label": f"Week {r['week']}", "n": r["n"], "model_right": r["model_right"],
                        "vegas_right": r["vegas_right"], "model_ll": r["model_ll"]} for r in rec["weeks"]],
              "total": rec["total"]}
    if evaluate is not None:
        try:
            record["scorecard"] = evaluate.cached(H.DATA / "cache", "nfl", lambda: long_run(E))
        except Exception as e:  # noqa: BLE001
            print(f"[app] scorecard skipped: {type(e).__name__}: {e}")
    if _LOG.get("l") is not None:
        record["live"] = livelog.live_record(_LOG["l"])
        W.write("live_log.json", _LOG["l"].payload())
        _LOG["l"].save()
    pw = H.rankings()
    W.write("power.json", {"about": "Points better than an average team on a neutral field, with each team's projected QB.",
                           "unit": "pts", "teams": [{"rank": t["rank"], "team": t["team"], "power": t["power"],
                                                     "sub": f"{t['record']} · {t['qb'] or 'QB ?'}"
                                                            + (f" ({t['qb_rating']:+.1f})" if t.get("qb_rating") is not None else "")}
                                                    for t in pw["teams"]]})
    players, hist = players_files(season)
    W.write("players.json", players)
    W.write("history.json", hist)
    inj = st.get("injury_report_week")
    W.write("meta.json", {
        "league": "nfl", "name": "NFL", "season": season, "season_label": str(season),
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "periods": periods, "default_period": str(st["default_week"]), "period_word": "week",
        "through": f"thru Wk {st['last_played_week']}",
        "injury_period": str(inj) if inj is not None else None,
        "units": {"margin": "pts", "starter": "QB", "starter_plural": "QBs", "rating_unit": "pts/g"},
        "starter_help": "Defaults to last game's starter. Change it if you know who's playing; the prediction updates instantly.",
        "injury_pending": "This week's injury report isn't out yet (Wed–Fri). Only IR and roster moves are counted so far.",
        "record": record, "model": "v4",
    })
    W.finish()
    print(f"[app] NFL: wrote {W.summary()} to {W.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
