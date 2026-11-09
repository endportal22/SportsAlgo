"""
Data layer for the HUD (app.py serves it as JSON).

Everything is built once (pipeline.build, cached on disk) and kept in memory until the
data is updated.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

import features as F
import pipeline
from model import STAT_FEATS, MarginModel
from paths import DATA, REPORTS, season_label, today

FEATURE_LABELS = {
    "d_elo": "Overall strength (Elo)",
    "d_ev": "5-on-5 chances (xG)",
    "d_st": "Special teams",
    "d_goalie": "Starting goalies",
    "d_injury_rel": "Injuries / lineup",
    "d_b2b": "Back-to-back",
    "rest_diff": "Rest advantage",
    "home_field": "Home ice",
}
GROUP_OF = {"C": "F", "L": "F", "R": "F", "LW": "F", "RW": "F", "F": "F", "D": "D", "G": "G"}


def _num(x, nd=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def _clean(d):
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in d.items()}


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.df = self.league = self.pl = None
        self.error = None
        self.built_at = None
        self._fits = {}
        self._players = None
        self._scores = {}

    def has_data(self):
        return (DATA / "games.csv").exists() and any(DATA.glob("states_*.csv.gz"))

    def ensure(self):
        with self.lock:
            if self.df is not None or not self.has_data():
                return
            try:
                t = time.time()
                self.df, self.league, self.pl = pipeline.build(DATA, verbose=True)
                self._fits, self._players = {}, None
                g = pd.read_csv(DATA / "games.csv", usecols=["game_id", "home_score", "away_score", "last_period"])
                g = g.dropna(subset=["home_score"])
                self._scores = {r.game_id: (int(r.home_score), int(r.away_score), r.last_period) for r in g.itertuples(index=False)}
                self.error = None
                self.built_at = datetime.now()
                print(f"[engine] ready in {time.time() - t:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()

    def invalidate(self):
        with self.lock:
            self.df = self.league = self.pl = None
            self._fits, self._players = {}, None

    def train_set(self, cutoff):
        df = self.df
        first = int(df.season.min())
        return df[df.played & (df.season > first) & (df.gameday < cutoff)]

    def fit_before(self, cutoff):
        key = pd.Timestamp(cutoff).normalize()
        if key not in self._fits:
            tr = self.train_set(key)
            if len(tr) < 200:   # very first season downloaded: fall back to everything played so far
                tr = self.df[self.df.played & (self.df.gameday < key)]
            if len(tr) < 50:
                tr = self.df[self.df.played]
            self._fits[key] = MarginModel(STAT_FEATS).fit(tr)
        return self._fits[key]

    @property
    def season(self):
        return int(self.df.season.max())

    @property
    def fen(self):
        return float(self.pl["fen_per_game"])


ENGINE = Engine()


# ------------------------------------------------------------ helpers
def contributions(model, row):
    """Goals of home margin from each feature, relative to an average game (base = average home margin)."""
    sc = model.pipe.named_steps["standardscaler"]
    rg = model.pipe.named_steps["ridge"]
    x = row[model.feats].astype(float).fillna(0.0).values
    parts = (x - sc.mean_) / sc.scale_ * rg.coef_
    return float(rg.intercept_), [{"feature": f, "label": FEATURE_LABELS.get(f, f), "value": _num(v), "points": _num(p, 3)}
                                  for f, v, p in zip(model.feats, x, parts)]


def _goalie_rating_now(pid):
    gp = ENGINE.pl["goalie_prod"]
    last = gp[(gp.pid == pid) & gp.game_id.isna()]
    if last.empty:
        return -0.002
    return float(last.rating.iloc[-1])


def _apply_goalie_override(row, side, pid):
    pj = ENGINE.pl["goalie_proj"].get((row.game_id, row.home_team if side == "home" else row.away_team))
    if not pj:
        return row
    opt = next((o for o in pj["options"] if str(o["pid"]) == str(pid)), None)
    if not opt:
        return row
    row = row.copy()
    pre = "h_" if side == "home" else "a_"
    row[f"{side}_goalie"] = opt["name"]
    row[f"{side}_goalie_id"] = opt["pid"]
    row[f"{pre}goalie_rating"] = _goalie_rating_now(opt["pid"])
    row["d_goalie"] = (row["h_goalie_rating"] - row["a_goalie_rating"]) * ENGINE.fen
    row[f"{side}_goalie_source"] = "your pick"
    return row


def _team_lineup(game_id, team):
    det = ENGINE.pl["lineup_detail"].get((game_id, team), [])
    info = ENGINE.pl["info"]
    out_list = []
    for d in det:
        if d["avail"] < 1 and d["share"] >= 0.18 and d["app"] >= 0.6:
            imp = d["off"] + d["def"]
            out_list.append({"pid": int(d["pid"]), "name": info["name"].get(d["pid"]), "pos": info["pos"].get(d["pid"]),
                             "share": d["share"], "impact": round(imp, 2), "toi": round(d["share"] / 5 * 60, 1),
                             "avail": d["avail"], "why": d["why"]})
    out_list.sort(key=lambda x: -x["toi"])
    return {"absences": out_list[:8]}


def _ml_text(ml):
    if ml is None or (isinstance(ml, float) and math.isnan(ml)):
        return None
    ml = float(ml)
    return f"+{ml:.0f}" if ml > 0 else f"{ml:.0f}"


def _game(r, model, played_info=True):
    m = float(model.predict_margin(r.to_frame().T)[0])
    p = float(model.prob_from_margin(m))
    base, parts = contributions(model, r)
    mp = _num(r.get("market_home_prob"))
    g = {
        "game_id": int(r.game_id), "date": r.gameday.strftime("%Y-%m-%d"), "start_utc": r.get("start_utc"),
        "away": r.away_team, "home": r.home_team, "neutral": r.home_field == 0, "playoff": r.game_type == "P",
        "played": bool(r.played), "model_margin": _num(m, 2), "model_prob": _num(p), "market_prob": mp,
        "home_ml": _ml_text(r.get("home_ml")), "away_ml": _ml_text(r.get("away_ml")), "book": r.get("book") if isinstance(r.get("book"), str) else None,
        "edge": _num(p - mp) if mp is not None else None, "pick": r.home_team if p >= 0.5 else r.away_team,
        "base_points": _num(base, 3), "contributions": parts,
        "home_b2b": bool(r.h_b2b), "away_b2b": bool(r.a_b2b), "home_rest": int(r.h_rest), "away_rest": int(r.a_rest),
    }
    for side, team, pre in (("home", r.home_team, "h_"), ("away", r.away_team, "a_")):
        nm = r.get(f"{side}_goalie")
        gl = {"name": nm if isinstance(nm, str) else None,
              "rating": _num(r[f"{pre}goalie_rating"] * ENGINE.fen if pd.notna(r[f"{pre}goalie_rating"]) else None, 2),
              "source": r.get(f"{side}_goalie_source", "actual")}
        pj = ENGINE.pl["goalie_proj"].get((r.game_id, team)) if not r.played else None
        if pj:
            gl["note"] = pj.get("note")
            gl["options"] = [{"pid": o["pid"], "name": o["name"], "available": o["available"], "status": o["status"],
                              "starts10": o["starts10"], "season_starts": o["season_starts"],
                              "rating": _num(_goalie_rating_now(o["pid"]) * ENGINE.fen, 2),
                              "selected": o["name"] == gl["name"]} for o in pj["options"] if o["name"]]
        g[f"{side}_goalie"] = gl
        g[f"{side}_injury"] = _num(r.get(f"{pre}injury_rel"), 2)
        g[f"{side}_lineup"] = _team_lineup(r.game_id, team) if r.season == ENGINE.season else None
    if r.played and played_info:
        hs, as_, lp = ENGINE._scores.get(r.game_id, (None, None, None))
        g["home_score"], g["away_score"] = hs, as_
        g["last_period"] = lp if lp in ("OT", "SO") else None
        winner = r.home_team if r.margin > 0 else r.away_team
        g["winner"] = winner
        g["model_correct"] = winner == g["pick"]
        if mp is not None:
            g["market_correct"] = winner == (r.home_team if mp >= 0.5 else r.away_team)
    return g


def day_payload(day, overrides=None):
    ENGINE.ensure()
    df = ENGINE.df
    d = pd.Timestamp(day).normalize()
    wk = df[df.gameday == d].sort_values(["start_utc", "game_id"])
    if wk.empty:
        return {"date": d.strftime("%Y-%m-%d"), "games": []}
    model = ENGINE.fit_before(d)
    overrides = overrides or {}
    games = []
    for _, r in wk.iterrows():
        if not r.played:
            for side, team in (("home", r.home_team), ("away", r.away_team)):
                if team in overrides:
                    r = _apply_goalie_override(r, side, overrides[team])
        games.append(_game(r, model))
    tr = ENGINE.train_set(d)
    return {"date": d.strftime("%Y-%m-%d"), "season": int(wk.season.iloc[0]), "curve": {"a": round(model.a, 3), "b": round(model.b, 3)},
            "trained_through": str(tr.gameday.max().date()) if len(tr) else None, "n_train": int(len(tr)), "games": games}


def season_record(season):
    ENGINE.ensure()
    df = ENGINE.df
    s = df[(df.season == season) & df.played].copy()
    if s.empty:
        return {"season": season, "weeks": [], "total": {"n": 0}}
    s["monday"] = s.gameday - pd.to_timedelta(s.gameday.dt.weekday, unit="D")
    rows = []
    for mon, wk in s.groupby("monday"):
        m = ENGINE.fit_before(mon)
        p = np.clip(m.predict_proba(wk), 1e-6, 1 - 1e-6)
        y = wk.home_win.values
        has = wk.market_home_prob.notna().values
        vp = np.clip(wk.market_home_prob.fillna(0.5).values, 1e-6, 1 - 1e-6)
        rows.append({"week": str(mon.date()), "n": int(len(wk)), "model_right": int(((p > 0.5) == (y == 1)).sum()),
                     "n_line": int(has.sum()),
                     "model_right_line": int((((p > 0.5) == (y == 1)) & has).sum()),
                     "vegas_right": int((((vp > 0.5) == (y == 1)) & has).sum()),
                     "model_ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))})
    tot = {k: sum(r[k] for r in rows) for k in ("n", "model_right", "n_line", "model_right_line", "vegas_right")}
    tot["model_ll"] = sum(r["model_ll"] * r["n"] for r in rows) / max(tot["n"], 1)
    return {"season": season, "label": season_label(season), "weeks": rows, "total": tot}


def rankings():
    ENGINE.ensure()
    df, lg = ENGINE.df, ENGINE.league
    season = ENGINE.season
    s = df[df.season == season]
    active = sorted(set(s.home_team) | set(s.away_team))
    m = ENGINE.fit_before(pd.Timestamp(today()) + pd.Timedelta(days=1))
    teams = [lg.teams[t] for t in active if t in lg.teams]
    avg = F.TeamState(ev_xgf60=np.mean([t.ev_xgf60 for t in teams]), ev_xga60=np.mean([t.ev_xga60 for t in teams]),
                      st_net=np.mean([t.st_net for t in teams]), elo=np.mean([t.elo for t in teams]))
    proj = {}
    for (gid, team), pj in sorted(ENGINE.pl["goalie_proj"].items()):
        if team not in proj:
            # the team's No. 1, not a back-to-back backup
            opts = [o for o in pj["options"] if o["available"]]
            no1 = max(opts, key=lambda o: (o["starts10"], o["season_starts"])) if opts else pj
            proj[team] = (no1["name"], _goalie_rating_now(no1["pid"]))
    avg_g = np.mean([v[1] for v in proj.values()]) if proj else -0.002
    recs = {}
    for r in s[s.played & (s.game_type == "R")].itertuples(index=False):
        for t, sign in ((r.home_team, 1), (r.away_team, -1)):
            w, l, otl = recs.get(t, (0, 0, 0))
            won = sign * r.margin > 0
            recs[t] = (w + won, l + (not won and not r.extra_time), otl + (not won and bool(r.extra_time)))
    zero = pd.DataFrame([{f: 0.0 for f in STAT_FEATS}])
    rows = []
    for t in active:
        st = lg.teams.get(t)
        if st is None:
            continue
        row = {f: 0.0 for f in STAT_FEATS}
        row.update({"d_elo": st.elo - avg.elo, "d_ev": F.ev_edge(st, avg, lg), "d_st": st.st_net - avg.st_net,
                    "d_goalie": (proj.get(t, (None, avg_g))[1] - avg_g) * ENGINE.fen})
        power = float(m.predict_margin(pd.DataFrame([row]))[0] - m.predict_margin(zero)[0])
        w, l, otl = recs.get(t, (0, 0, 0))
        rows.append({"team": t, "power": round(power, 2),
                     "elo": round(st.elo), "record": f"{w}-{l}-{otl}", "points": 2 * w + otl,
                     "xgf60": round(st.ev_xgf60, 2), "xga60": round(st.ev_xga60, 2),
                     "pp60": round(st.pp60, 2), "pk60": round(st.pk60, 2), "st_net": round(st.st_net, 2),
                     "goalie": proj.get(t, (None, None))[0],
                     "goalie_rating": _num((proj.get(t, (None, avg_g))[1]) * ENGINE.fen, 2)})
    rows.sort(key=lambda r: -r["power"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"season": season, "label": season_label(season), "teams": rows}


# ------------------------------------------------------------ players
def _players_table():
    if ENGINE._players is not None:
        return ENGINE._players
    pl = ENGINE.pl
    season = ENGINE.season
    info = pl["info"]
    hist = pl["history"]
    sk = pl["skaters"]
    cur = sk[sk.season == season]
    if cur.empty:
        cur = sk[sk.season == sk.season.max()]
    cps = sorted(hist.cp.unique())
    rows = []
    if cps:
        h_now = hist[hist.cp == cps[-1]].set_index("pid")
        h_back = hist[hist.cp == cps[max(0, len(cps) - 5)]].set_index("pid")
        last10 = cur.sort_values("gameday").groupby("player_id").tail(10)
        toi = last10.groupby("player_id").toi.mean() / 60.0
        gp = cur.groupby("player_id").game_id.nunique()
        sp = pl["skater_prod"]
        sp_last = sp.sort_values("gameday").groupby("pid").last() if len(sp) else pd.DataFrame()
        for pid in cur.player_id.unique():
            if pid not in h_now.index:
                continue
            now = h_now.loc[pid]
            pos = info["pos"].get(pid)
            grp = GROUP_OF.get(str(pos).upper(), "F")
            prev = h_back.loc[pid] if pid in h_back.index else None
            r = {"pid": int(pid), "name": info["name"].get(pid), "pos": pos, "group": grp, "team": info["team"].get(pid),
                 "impact": round(float(now.impact), 3), "off": round(float(now.off), 3), "def": round(float(now["def"]), 3),
                 "trend": round(float(now.impact - prev.impact), 3) if prev is not None else None,
                 "exposure": round(float(now.expo), 1), "toi": round(float(toi.get(pid, 0.0)), 1), "games": int(gp.get(pid, 0))}
            if pid in sp_last.index:
                r["ixg60"] = round(float(sp_last.loc[pid].ixg60), 3)
                r["pts60"] = round(float(sp_last.loc[pid].pts60), 3)
            rows.append(r)
    tbl = pd.DataFrame(rows)
    recent = cps[-22:]
    spark = {pid: t.sort_values("cp").impact.values for pid, t in hist[hist.cp.isin(recent)].groupby("pid")}

    # goalies
    gg = pl["goalie_games"]
    gp_ = pl["goalie_prod"]
    gcur = gg[gg.season == season]
    glast = gp_[gp_.game_id.isna()].set_index("pid")
    grows = []
    for pid, t in gcur.groupby("player_id"):
        if pid not in glast.index:
            continue
        sh, ga, xga = t.shots.sum(), t.ga.sum(), t.xga.sum()
        grows.append({"pid": int(pid), "name": info["name"].get(pid), "pos": "G", "group": "G", "team": info["team"].get(pid),
                      "rating": round(float(glast.loc[pid].rating) * ENGINE.fen, 3), "evidence": round(float(glast.loc[pid].volume)),
                      "starts": int(t.starter.sum()), "games": int((t.toi > 0).sum()),
                      "gsax": round(float(xga - ga), 1), "fen_sv": round(1 - ga / sh, 3) if sh else None})
    gtbl = pd.DataFrame(grows)
    gspark = {pid: t.sort_values("gameday").rating.values[-30:] * ENGINE.fen
              for pid, t in gp_[gp_.season.notna() & (gp_.season >= season - 1)].groupby("pid")}
    ENGINE._players = {"table": tbl, "gtable": gtbl, "spark": spark, "gspark": gspark, "season": season}
    return ENGINE._players


def players_payload(group=None, team=None, q=None, sort="impact", limit=80):
    ENGINE.ensure()
    pt = _players_table()
    group = group or "ALL"
    if group == "G":
        t = pt["gtable"]
        if t.empty:
            return {"season": pt["season"], "group": "G", "players": [], "risers": [], "fallers": [], "teams": []}
        if team:
            t = t[t.team == team]
        if q:
            t = t[t.name.str.contains(q, case=False, na=False)]
        t = t[t.games >= 1]
        key = {"impact": "rating", "gsax": "gsax", "starts": "starts"}.get(sort, "rating")
        t = t.sort_values(key, ascending=False).head(limit)
        out = []
        for r in t.to_dict("records"):
            s = pt["gspark"].get(r["pid"])
            r["spark"] = [round(float(v), 3) for v in s] if s is not None else []
            out.append(_clean(r))
        teams = sorted(pt["table"].team.dropna().unique().tolist()) if len(pt["table"]) else []
        return {"season": pt["season"], "group": "G", "players": out, "risers": [], "fallers": [], "teams": teams}
    t = pt["table"]
    if t.empty:
        return {"season": pt["season"], "group": group, "players": [], "risers": [], "fallers": [], "teams": []}
    t = t[(t.toi >= 6) | (t.games >= 3)]
    if group in ("F", "D"):
        t = t[t.group == group]
    if team:
        t = t[t.team == team]
    if q:
        t = t[t.name.str.contains(q, case=False, na=False)]
    if sort == "trend":
        t = t[t.trend.notna()].sort_values("trend", ascending=False)
    elif sort == "falling":
        t = t[t.trend.notna()].sort_values("trend")
    elif sort in ("off", "def", "ixg60", "pts60"):
        t = t[t[sort].notna()].sort_values(sort, ascending=False) if sort in t else t
    else:
        t = t.sort_values("impact", ascending=False)
    t = t.head(limit)
    out = []
    for r in t.to_dict("records"):
        s = pt["spark"].get(r["pid"])
        r["spark"] = [round(float(v), 3) for v in s] if s is not None else []
        out.append(_clean(r))
    reg = pt["table"][(pt["table"].toi >= 14) & pt["table"].trend.notna()]
    if group in ("F", "D"):
        reg = reg[reg.group == group]
    pick = lambda d: [{"pid": int(x.pid), "name": x.name, "team": x.team, "pos": x.pos, "trend": x.trend, "impact": x.impact}
                      for x in d.itertuples(index=False)]
    return {"season": pt["season"], "group": group, "players": out,
            "risers": pick(reg.nlargest(5, "trend")), "fallers": pick(reg.nsmallest(5, "trend")),
            "teams": sorted(pt["table"].team.dropna().unique().tolist())}


def player_detail(pid):
    ENGINE.ensure()
    pt = _players_table()
    pl = ENGINE.pl
    info = pl["info"]
    pid = int(pid)
    gd = ENGINE.df.set_index("game_id")
    if str(info["pos"].get(pid, "")) == "G":
        row = pt["gtable"][pt["gtable"].pid == pid] if len(pt["gtable"]) else pd.DataFrame()
        base = row.iloc[0].to_dict() if len(row) else {"pid": pid, "name": info["name"].get(pid), "pos": "G", "team": info["team"].get(pid)}
        gp = pl["goalie_prod"]
        h = gp[(gp.pid == pid) & gp.season.notna() & (gp.season >= ENGINE.season - 2)].sort_values("gameday")
        series = [{"k": str(pd.Timestamp(d).date()), "season": int(s), "v": round(float(r) * ENGINE.fen, 3)}
                  for d, s, r in zip(h.gameday, h.season, h.rating)]
        gg = pl["goalie_games"]
        gg = gg[(gg.player_id == pid) & (gg.toi > 0)].sort_values("gameday").tail(10)
        games = []
        for r in gg.itertuples(index=False):
            g_ = gd.loc[r.game_id] if r.game_id in gd.index else None
            opp = (g_.away_team if g_.home_team == r.team else g_.home_team) if g_ is not None else ""
            games.append({"date": str(pd.Timestamp(r.gameday).date()), "team": r.team, "opp": opp, "starter": bool(r.starter),
                          "shots": int(r.shots), "ga": int(r.ga), "xga": round(float(r.xga), 2), "gsax": round(float(r.xga - r.ga), 2)})
        return {"kind": "goalie", "player": _clean(base), "series": series, "games": games[::-1]}
    row = pt["table"][pt["table"].pid == pid] if len(pt["table"]) else pd.DataFrame()
    base = row.iloc[0].to_dict() if len(row) else {"pid": pid, "name": info["name"].get(pid), "pos": info["pos"].get(pid), "team": info["team"].get(pid)}
    hist = pl["history"]
    h = hist[hist.pid == pid].sort_values("cp")
    h = h[h.cp >= pd.Timestamp(today()) - pd.Timedelta(days=3 * 365)]
    series = [{"k": str(pd.Timestamp(c).date()), "impact": round(float(i), 3), "off": round(float(o), 3), "def": round(float(d), 3)}
              for c, i, o, d in zip(h.cp, h.impact, h.off, h["def"])]
    sk = pl["skaters"]
    sk = sk[sk.player_id == pid].sort_values("gameday").tail(10)
    games = []
    for r in sk.itertuples(index=False):
        g_ = gd.loc[r.game_id] if r.game_id in gd.index else None
        opp = (g_.away_team if g_.home_team == r.team else g_.home_team) if g_ is not None else ""
        games.append({"date": str(pd.Timestamp(r.gameday).date()), "team": r.team, "opp": opp, "toi": round(r.toi / 60, 1),
                      "g": int(r.goals), "a": int(r.assists)})
    return {"kind": "skater", "player": _clean(base), "series": series, "games": games[::-1]}


def status(job_snapshot):
    if ENGINE.df is None and ENGINE.has_data() and not ENGINE.lock.locked():
        threading.Thread(target=ENGINE.ensure, daemon=True).start()
    gpath = DATA / "games.csv"
    st = {"has_data": ENGINE.has_data(), "engine_error": ENGINE.error, "job": job_snapshot, "today": today().isoformat()}
    if gpath.exists():
        st["data_updated"] = datetime.fromtimestamp(gpath.stat().st_mtime).isoformat(timespec="minutes")
    df = ENGINE.df
    if df is not None:
        season = ENGINE.season
        s = df[df.season == season]
        days = [{"date": d.strftime("%Y-%m-%d"), "n": int(len(g)), "played": bool(g.played.all()),
                 "playoff": bool((g.game_type == "P").any())} for d, g in s.groupby("gameday")]
        tday = pd.Timestamp(today())
        upcoming = [x["date"] for x in days if x["date"] >= tday.strftime("%Y-%m-%d")]
        played = s[s.played]
        st.update({"season": season, "label": season_label(season), "days": days,
                   "default_date": upcoming[0] if upcoming else (days[-1]["date"] if days else None),
                   "last_played": played.gameday.max().strftime("%Y-%m-%d") if len(played) else None,
                   "n_played": int(len(played)), "n_games": int(len(s))})
        inj = DATA / "injuries.csv"
        if inj.exists():
            try:
                st["injuries_fetched"] = str(pd.read_csv(inj, usecols=["fetched"]).fetched.iloc[0])
            except (ValueError, IndexError, KeyError):
                pass
        st["lines_upcoming"] = int(s[~s.played].market_home_prob.notna().sum())
    bt = REPORTS / "backtest_results.json"
    if bt.exists():
        st["backtest_updated"] = datetime.fromtimestamp(bt.stat().st_mtime).isoformat(timespec="minutes")
    return st
