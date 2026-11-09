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
import time
from datetime import datetime, timezone

import appdata
import hud_data as H
from paths import ROOT

PROD_GROUPS = {"QB", "RB", "WR", "TE"}


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
        if g["played"]:
            out.update({"home_score": g.get("home_score"), "away_score": g.get("away_score"), "winner": g.get("winner"),
                        "model_correct": g.get("model_correct"), "vegas_correct": g.get("vegas_correct")})
        games.append(out)
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
