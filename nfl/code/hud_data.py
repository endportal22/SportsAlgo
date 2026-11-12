"""
Data layer for the HUD (app.py serves it as JSON).

Everything is built once (pipeline.build, ~20s) and cached until the data is updated.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import features as F
import pipeline
import players as P
from model import STAT_FEATS, MarginModel
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402

HERE = Path(__file__).resolve().parent

FEATURE_LABELS = {
    "d_elo": "Overall strength (Elo)",
    "d_matchup": "Pass/run matchup",
    "d_pass_net": "Passing game edge",
    "d_run_net": "Running game edge",
    "d_qb_rating": "QB rating edge",
    "d_qb_new": "New starting QB",
    "d_injury_rel": "Injuries",
    "rest_diff": "Rest advantage",
    "home_field": "Home field",
    "div_game": "Division game",
    "d_roster": "Roster strength",
    "fw_x_elo": "Final week: strength",
    "fw_x_matchup": "Final week: matchup",
    "fw_x_qb_rating": "Final week: QB",
}

POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE",
             "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL", "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
             "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB",
             "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB"}
OFF_GROUPS = {"QB", "RB", "WR", "TE", "OL"}
PROD_KIND = {"QB": "qb", "RB": "rush", "WR": "rec", "TE": "rec"}
PROD_SCALE = {"qb": 35.0, "rush": 15.0, "rec": 7.0}   # typical per-game volume -> pts/game
PROD_LABEL = {"qb": "EPA per dropback", "rush": "EPA per carry", "rec": "EPA per target"}


def _num(x, nd=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.df = self.league = self.pl = None
        self.error = None
        self.built_at = None
        self._fits = {}
        self._players = None
        self._scores = {}

    # ------------------------------------------------------------ build / cache
    def ensure(self):
        with self.lock:
            if self.df is not None or not (DATA / "games.csv").exists():
                return
            try:
                t = time.time()
                self.df, self.league, self.pl = pipeline.build(DATA, verbose=True)
                self._fits, self._players = {}, None
                g = pd.read_csv(DATA / "games.csv", usecols=["game_id", "home_score", "away_score"]).dropna()
                self._scores = {r.game_id: (int(r.home_score), int(r.away_score)) for r in g.itertuples(index=False)}
                # Vegas win probability from the moneylines (vig removed) when there are any
                gm = pd.read_csv(DATA / "games.csv")
                if {"home_moneyline", "away_moneyline"} <= set(gm.columns):
                    ml = dict(zip(gm.game_id, _devig(gm.home_moneyline, gm.away_moneyline)))
                    self.df["ml_home_prob"] = self.df.game_id.map(ml)
                else:
                    self.df["ml_home_prob"] = np.nan
                self.error = None
                self.built_at = datetime.now()
                print(f"[engine] built in {time.time() - t:.1f}s")
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()

    def invalidate(self):
        with self.lock:
            self.df = self.league = self.pl = None
            self._fits, self._players = {}, None

    def train_set(self, cutoff):
        df = self.df
        return df[df.played & (df.season >= 2017) & (df.gameday < cutoff)]

    def fit_before(self, cutoff):
        key = pd.Timestamp(cutoff)
        if key not in self._fits:
            self._fits[key] = MarginModel(STAT_FEATS).fit(self.train_set(key))
        return self._fits[key]

    @property
    def season(self):
        return int(self.df.loc[self.df.played, "season"].max())


ENGINE = Engine()


# ------------------------------------------------------------ helpers
def contributions(model, row):
    sc = model.pipe.named_steps["standardscaler"]
    rg = model.pipe.named_steps["ridge"]
    x = row[model.feats].astype(float).fillna(0.0).values
    parts = (x - sc.mean_) / sc.scale_ * rg.coef_
    return float(rg.intercept_), [{"feature": f, "label": FEATURE_LABELS.get(f, f), "value": _num(v), "points": _num(p, 2)}
                                  for f, v, p in zip(model.feats, x, parts)]


def _qb_rating_now(gsis):
    pr = ENGINE.pl["production"]
    q = pr[(pr.kind == "qb") & (pr.pid == gsis)]
    if q.empty:
        return -0.08
    return float(q.sort_values("gameday").rating.iloc[-1])


def _apply_qb_override(row, side, pfr):
    """Swap the projected QB for an upcoming game and recompute the QB features."""
    pj = ENGINE.pl["qb_proj"].get((row.game_id, row.home_team if side == "home" else row.away_team))
    if not pj:
        return row, None
    opt = next((o for o in pj["options"] if o["pfr"] == pfr), None)
    if not opt or not opt["gsis"]:
        return row, None
    row = row.copy()
    pre = "h_" if side == "home" else "a_"
    row[f"{side}_qb"] = opt["name"]
    row[f"{side}_qb_id"] = opt["gsis"]
    row[f"{pre}qb_new"] = opt["new"]
    row[f"{pre}qb_rating"] = _qb_rating_now(opt["gsis"])
    row["d_qb_new"] = row["h_qb_new"] - row["a_qb_new"]
    row["d_qb_rating"] = (row["h_qb_rating"] - row["a_qb_rating"]) * 35.0
    row[f"{side}_qb_source"] = "your pick"
    return row, opt


def _team_lineup(game_id, team, unplayed):
    """Key absences and top contributors for one team in one game."""
    det = ENGINE.pl["lineup_detail"].get((game_id, team), [])
    info = ENGINE.pl["info"]
    out_list, top = [], []
    for d in det:
        name = info.name.get(d["pid"]); pos = info.pos.get(d["pid"])
        val = d["pts"] * d["share"]
        item = {"pid": d["pid"], "name": name, "pos": pos, "unit": d["unit"], "share": d["share"],
                "impact": d["pts"], "value": round(val, 2), "avail": d["avail"], "why": d["why"]}
        if d["avail"] < 1 and d["share"] >= 0.25:
            out_list.append(item)
        elif d["avail"] >= 1:
            top.append(item)
    out_list.sort(key=lambda x: -abs(x["value"]))
    top.sort(key=lambda x: -x["value"])
    return {"absences": out_list[:8], "top": top[:5]}


def week_payload(season, week, overrides=None):
    ENGINE.ensure()
    df = ENGINE.df
    wk = df[(df.season == season) & (df.week == week)].sort_values("gameday")
    if wk.empty:
        return {"games": []}
    cutoff = wk.gameday.min()
    model = ENGINE.fit_before(cutoff)
    overrides = overrides or {}
    games = []
    for _, r in wk.iterrows():
        notes = {}
        if not r.played:
            for side, team in (("home", r.home_team), ("away", r.away_team)):
                if team in overrides:
                    r, opt = _apply_qb_override(r, side, overrides[team])
        m = float(model.predict_margin(r.to_frame().T)[0])
        p = float(norm.cdf(m / model.sigma))
        base, parts = contributions(model, r)
        spread = _num(r.spread_line, 1)
        vegas_p = vegas_prob(r, model.sigma)
        g = {
            "game_id": r.game_id, "gameday": r.gameday.strftime("%a %b ") + str(r.gameday.day),
            "date": r.gameday.strftime("%Y-%m-%d"), "away": r.away_team, "home": r.home_team,
            "neutral": r.home_field == 0, "game_type": r.game_type, "played": bool(r.played),
            "model_margin": _num(m, 1), "model_prob": _num(p), "spread": spread, "vegas_prob": vegas_p,
            "edge": _num(p - vegas_p) if vegas_p is not None else None,
            "pick": r.home_team if p >= 0.5 else r.away_team,
            "base_points": _num(base, 2), "contributions": parts,
        }
        for side, team, pre in (("home", r.home_team, "h_"), ("away", r.away_team, "a_")):
            qb = {"name": r[f"{side}_qb"] if isinstance(r[f"{side}_qb"], str) else None,
                  "new": bool(r[f"{pre}qb_new"]),
                  "rating": _num(r[f"{pre}qb_rating"] * 35.0 if pd.notna(r[f"{pre}qb_rating"]) else None, 1),
                  "source": r.get(f"{side}_qb_source", "actual")}
            pj = ENGINE.pl["qb_proj"].get((r.game_id, team)) if not r.played else None
            if pj:
                qb["options"] = [{"pfr": o["pfr"], "name": o["name"], "available": o["available"], "status": o["status"],
                                  "note": o["note"], "season_starts": o["season_starts"],
                                  "rating": _num(_qb_rating_now(o["gsis"]) * 35.0, 1) if o["gsis"] else None,
                                  "selected": o["name"] == qb["name"]} for o in pj["options"] if o["name"]]
                qb["return_flag"] = next((o["name"] for o in pj["options"] if o["note"] and o["name"] != qb["name"]), None)
            g[f"{side}_qb"] = qb
            g[f"{side}_injury"] = _num(r.get(f"{pre}injury_rel"), 1)
            g[f"{side}_lineup"] = _team_lineup(r.game_id, team, not r.played) if (r.season == ENGINE.season) else None
        if r.played:
            hs, as_ = ENGINE._scores.get(r.game_id, (None, None))
            g["home_score"], g["away_score"] = hs, as_
            winner = r.home_team if r.margin > 0 else r.away_team if r.margin < 0 else None
            g["winner"] = winner
            g["model_correct"] = (winner == g["pick"]) if winner else None
            if vegas_p is not None and winner:
                g["vegas_correct"] = winner == (r.home_team if vegas_p >= 0.5 else r.away_team)
        games.append(g)
    tr = ENGINE.train_set(cutoff)
    return {"season": season, "week": week, "sigma": round(model.sigma, 2),
            "trained_through": str(tr.gameday.max().date()), "n_train": int(len(tr)), "games": games}


def _devig(hml, aml):
    """Home win probability from American moneylines, with the bookmaker's margin removed."""
    def imp(x):
        x = pd.to_numeric(x, errors="coerce")
        return np.where(x < 0, -x / (-x + 100.0), 100.0 / (x + 100.0))
    ph, pa = imp(hml), imp(aml)
    s = ph + pa
    return np.where((s > 0.98) & (s < 1.15), ph / s, np.nan)


def vegas_prob(r, sigma):
    """Vegas's home win probability: from the moneylines when posted (more accurate: in 2019-2026
    games their log loss is 0.6098 vs 0.6105-0.6121 for the spread converted with a normal curve),
    else from the spread."""
    ml = getattr(r, "ml_home_prob", None)
    if ml is not None and not pd.isna(ml):
        return float(ml)
    sp = getattr(r, "spread_line", None)
    if sp is None or pd.isna(sp):
        return None
    return float(norm.cdf(sp / sigma))


def season_record(season):
    ENGINE.ensure()
    df = ENGINE.df
    rows = []
    for w in sorted(df[(df.season == season) & df.played].week.unique()):
        wk = df[(df.season == season) & (df.week == w) & df.played & (df.home_win != 0.5)]
        if wk.empty:
            continue
        m = ENGINE.fit_before(df[(df.season == season) & (df.week == w)].gameday.min())
        p = np.clip(m.predict_proba(wk), 1e-6, 1 - 1e-6)
        vp = np.clip([vegas_prob(r, m.sigma) if vegas_prob(r, m.sigma) is not None else 0.5 for r in wk.itertuples(index=False)],
                     1e-6, 1 - 1e-6)
        y = wk.home_win.values
        rows.append({"week": int(w), "n": int(len(wk)),
                     "model_right": int(((p > 0.5) == (y == 1)).sum()), "vegas_right": int(((vp > 0.5) == (y == 1)).sum()),
                     "model_ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                     "vegas_ll": float(-np.mean(y * np.log(vp) + (1 - y) * np.log(1 - vp)))})
    tot = {k: sum(r[k] for r in rows) for k in ("n", "model_right", "vegas_right")}
    if rows:
        tot["model_ll"] = sum(r["model_ll"] * r["n"] for r in rows) / tot["n"]
        tot["vegas_ll"] = sum(r["vegas_ll"] * r["n"] for r in rows) / tot["n"]
    return {"season": season, "weeks": rows, "total": tot}


def rankings():
    """Neutral-field points vs an average opponent, using the current model's weights."""
    ENGINE.ensure()
    df, lg = ENGINE.df, ENGINE.league
    season = ENGINE.season
    s = df[df.season == season]
    active = sorted(set(s.home_team) | set(s.away_team))
    m = ENGINE.fit_before(pd.Timestamp.now() + pd.Timedelta(days=365))
    teams = [lg.teams[t] for t in active]
    avg = F.TeamState(off_epa=np.mean([t.off_epa for t in teams]), def_epa=np.mean([t.def_epa for t in teams]),
                      pass_off=np.mean([t.pass_off for t in teams]), run_off=np.mean([t.run_off for t in teams]),
                      pass_def=np.mean([t.pass_def for t in teams]), run_def=np.mean([t.run_def for t in teams]),
                      pass_rate=np.mean([t.pass_rate for t in teams]), elo=np.mean([t.elo for t in teams]))
    # projected QB rating per team for the next game
    qbr = {}
    for (gid, team), pj in ENGINE.pl["qb_proj"].items():
        if team not in qbr and pj.get("gsis"):
            qbr[team] = (pj["name"], _qb_rating_now(pj["gsis"]))
    avg_qb = np.mean([v[1] for v in qbr.values()]) if qbr else 0.0
    recs = {}
    for r in s[s.played & (s.game_type == "REG")].itertuples(index=False):
        for t, sign in ((r.home_team, 1), (r.away_team, -1)):
            w, l, tie = recs.get(t, (0, 0, 0))
            mm = sign * r.margin
            recs[t] = (w + (mm > 0), l + (mm < 0), tie + (mm == 0))
    rows = []
    for t in active:
        st = lg.teams[t]
        row = {f: 0.0 for f in STAT_FEATS}
        row.update({"d_elo": st.elo - avg.elo,
                    "d_matchup": F._exp_off(st, avg, lg) - F._exp_off(avg, st, lg),
                    "d_pass_net": (st.pass_off - st.pass_def) - (avg.pass_off - avg.pass_def),
                    "d_run_net": (st.run_off - st.run_def) - (avg.run_off - avg.run_def),
                    "d_qb_rating": (qbr.get(t, (None, avg_qb))[1] - avg_qb) * 35.0})
        zero = {f: 0.0 for f in STAT_FEATS}
        power = float(m.predict_margin(pd.DataFrame([row]))[0] - m.predict_margin(pd.DataFrame([zero]))[0])
        w, l, tie = recs.get(t, (0, 0, 0))
        rows.append({"team": t, "power": round(power, 1), "elo": round(st.elo), "record": f"{w}-{l}" + (f"-{tie}" if tie else ""),
                     "pass_off": round(st.pass_off, 3), "run_off": round(st.run_off, 3),
                     "pass_def": round(st.pass_def, 3), "run_def": round(st.run_def, 3),
                     "qb": qbr.get(t, (None, None))[0], "qb_rating": _num((qbr.get(t, (None, avg_qb))[1]) * 35.0, 1)})
    rows.sort(key=lambda r: -r["power"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"season": season, "teams": rows}


# ------------------------------------------------------------ players
def _players_table():
    """Latest rating + trend for every player active this season or last."""
    if ENGINE._players is not None:
        return ENGINE._players
    pl = ENGINE.pl
    season = ENGINE.season
    hist = pl["history"]
    info = pl["info"]
    snaps = P.load_snaps(DATA, [season - 1, season])
    gday = ENGINE.df.set_index("game_id").gameday
    snaps["gameday"] = snaps.game_id.map(gday)
    cur = snaps[snaps.season == season]
    # recent share: last 4 team games
    last_games = cur.groupby("team").apply(lambda t: set(t.sort_values("gameday").game_id.unique()[-4:]))
    recent = cur[cur.apply(lambda r: r.game_id in last_games.get(r.team, set()), axis=1)]
    share = recent.groupby("pfr_player_id").agg(off=("offense_pct", "sum"), dfn=("defense_pct", "sum"), team=("team", "last"))
    share["off"] /= 4; share["dfn"] /= 4
    games_played = cur.groupby("pfr_player_id").game_id.nunique()

    # checkpoints in order
    cps = hist[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    cps = [tuple(x) for x in cps.values]
    last_cp = cps[-1]
    back4 = cps[max(0, len(cps) - 5)]
    h_now = hist[(hist.season == last_cp[0]) & (hist.week == last_cp[1])].set_index("pid")
    h_back = hist[(hist.season == back4[0]) & (hist.week == back4[1])].set_index("pid")

    pf2gs = P._id_maps(DATA, season)
    pr = pl["production"]
    pr_last = pr.sort_values("gameday").groupby(["pid", "kind"]).agg(rating=("rating", "last"), volume=("volume", "last"))
    pr_hist = pr[pr.game_id.notna()].sort_values("gameday")

    rows = []
    for pid in set(share.index) | set(cur.pfr_player_id.unique()):
        if pid not in info.index or pid not in h_now.index:
            continue
        pos = info.pos.get(pid)
        grp = POS_GROUP.get(str(pos).upper())
        if grp is None:
            continue
        unit = "off" if grp in OFF_GROUPS else "def"
        now = h_now.loc[pid]
        val = float(now[unit])
        expo = float(now[f"{unit}_expo"])
        prev = float(h_back.loc[pid][unit]) if pid in h_back.index else None
        sh = share.loc[pid] if pid in share.index else None
        snap = float(sh["off"] if unit == "off" else sh["dfn"]) if sh is not None else 0.0
        team = sh["team"] if sh is not None else info.team.get(pid)
        r = {"pid": pid, "name": info.name.get(pid), "pos": pos, "group": grp, "team": team, "unit": unit,
             "impact": round(val, 2), "trend": round(val - prev, 2) if prev is not None else None,
             "exposure": round(expo, 1), "snap": round(snap, 3), "games": int(games_played.get(pid, 0))}
        kind = PROD_KIND.get(grp)
        gs = pf2gs.get(pid)
        if kind and gs is not None and (gs, kind) in pr_last.index:
            pv = pr_last.loc[(gs, kind)]
            r["prod"] = round(float(pv.rating) * PROD_SCALE[kind], 2)
            r["prod_volume"] = round(float(pv.volume), 0)
            r["prod_kind"] = kind
        rows.append(r)
    tbl = pd.DataFrame(rows)
    # sparkline: last 22 checkpoints of the player's main unit
    recent_cps = cps[-22:]
    hh = hist.merge(pd.DataFrame(recent_cps, columns=["season", "week"]), on=["season", "week"])
    spark = {}
    for pid, t in hh.groupby("pid"):
        spark[pid] = t
    ENGINE._players = {"table": tbl, "spark": spark, "cps": recent_cps, "season": season, "pf2gs": pf2gs,
                       "pr_hist": pr_hist}
    return ENGINE._players


def players_payload(group=None, team=None, q=None, sort="impact", limit=60):
    ENGINE.ensure()
    pt = _players_table()
    t = pt["table"]
    t = t[(t.snap >= 0.2) | (t.games >= 2)]
    if group and group != "ALL":
        t = t[t.group == group]
    if team:
        t = t[t.team == team]
    if q:
        t = t[t.name.str.contains(q, case=False, na=False)]
    if sort == "trend":
        t = t[t.trend.notna()].sort_values("trend", ascending=False)
    elif sort == "falling":
        t = t[t.trend.notna()].sort_values("trend")
    elif sort == "prod":
        t = t[t["prod"].notna()].sort_values("prod", ascending=False) if "prod" in t else t
    else:
        t = t.sort_values("impact", ascending=False)
    t = t.head(limit)
    out = []
    for r in t.to_dict("records"):
        s = pt["spark"].get(r["pid"])
        r["spark"] = [round(float(v), 2) for v in s[r["unit"]].values] if s is not None else []
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    # risers / fallers among regulars
    reg = pt["table"][(pt["table"].snap >= 0.5) & pt["table"].trend.notna()]
    if group and group != "ALL":
        reg = reg[reg.group == group]
    pick = lambda d: [{"pid": x.pid, "name": x.name, "team": x.team, "pos": x.pos, "trend": x.trend, "impact": x.impact}
                      for x in d.itertuples(index=False)]
    return {"season": pt["season"], "players": out,
            "risers": pick(reg.nlargest(5, "trend")), "fallers": pick(reg.nsmallest(5, "trend")),
            "teams": sorted(pt["table"].team.dropna().unique().tolist())}


def player_detail(pid):
    ENGINE.ensure()
    pt = _players_table()
    pl = ENGINE.pl
    info = pl["info"]
    row = pt["table"][pt["table"].pid == pid]
    base = row.iloc[0].to_dict() if len(row) else {"pid": pid, "name": info.name.get(pid), "pos": info.pos.get(pid)}
    unit = base.get("unit", "off")
    hist = pl["history"]
    h = hist[(hist.pid == pid) & (hist.season >= ENGINE.season - 2)].sort_values(["season", "week"])
    series = [{"season": int(a), "week": int(b), "v": round(float(v), 2), "expo": round(float(e), 1)}
              for a, b, v, e in zip(h.season, h.week, h[unit], h[f"{unit}_expo"])]
    prod = []
    kind = base.get("prod_kind")
    gs = pt["pf2gs"].get(pid)
    if kind and gs:
        ph = pt["pr_hist"]
        ph = ph[(ph.pid == gs) & (ph.kind == kind) & (ph.season >= ENGINE.season - 2)]
        prod = [{"season": int(s), "week": int(w), "v": round(float(r) * PROD_SCALE[kind], 2)}
                for s, w, r in zip(ph.season, ph.week, ph.rating)]
    sn = P.load_snaps(DATA, [ENGINE.season - 1, ENGINE.season])
    sn = sn[sn.pfr_player_id == pid].sort_values(["season", "week"])
    snaps = [{"season": int(s), "week": int(w), "team": t, "opp": o, "pct": float(op if unit == "off" else dp)}
             for s, w, t, o, op, dp in zip(sn.season, sn.week, sn.team, sn.opponent, sn.offense_pct, sn.defense_pct)]
    base = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in base.items()}
    return {"player": base, "impact": series, "prod": prod, "prod_label": PROD_LABEL.get(kind), "snaps": snaps[-20:]}


def status(job_snapshot):
    ENGINE.ensure()
    gpath = DATA / "games.csv"
    st = {"has_data": gpath.exists(), "engine_error": ENGINE.error, "job": job_snapshot}
    if gpath.exists():
        st["data_updated"] = datetime.fromtimestamp(gpath.stat().st_mtime).isoformat(timespec="minutes")
    df = ENGINE.df
    if df is not None:
        season = ENGINE.season
        s = df[df.season == season]
        labels = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}
        weeks = [{"week": int(w), "played": bool(g.played.all()), "label": labels.get(g.game_type.iloc[0], f"Week {w}")}
                 for w, g in s.groupby("week")]
        upcoming = [w["week"] for w in weeks if not w["played"]]
        st.update({"season": season, "weeks": weeks, "default_week": upcoming[0] if upcoming else weeks[-1]["week"],
                   "last_played_week": int(s[s.played].week.max())})
        inj = DATA / f"injuries_{season}.csv.gz"
        if inj.exists():
            i = pd.read_csv(inj, usecols=["week"])
            st["injury_report_week"] = int(i.week.max())
    bt = REPORTS / "backtest_results.json"
    if bt.exists():
        st["backtest_updated"] = datetime.fromtimestamp(bt.stat().st_mtime).isoformat(timespec="minutes")
    return st
