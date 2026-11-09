#!/usr/bin/env python3
"""
Package the model's output as static JSON for the phone app (app/ folder).

The phone never runs Python. Everything it shows comes from these files:

  app/data/meta.json             season, weeks, last update, season record
  app/data/week_<season>_<wk>.json  every game of a week, the model formula for that week
                                 (so the phone can recompute instantly when you switch a QB),
                                 QB options, missing players, results
  app/data/power.json            power rankings
  app/data/players.json          every regular player: rating, trend, sparkline
  app/data/history.json          weekly rating history per player (loaded when you open one)

Usage: python export_app.py            (run after update_data.py; takes ~40s)
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import hud_data as H
from paths import ROOT

OUT = ROOT / "app" / "data"

# Password protection: when APP_PASSWORD is set (it is on the hosted build), every data file
# is encrypted with AES-256-GCM using a key derived from the password (PBKDF2-SHA256), and
# lock.json lets the app check a password. Without it (local runs), files are plain JSON.
PBKDF2_ITER = 210_000
_KEY = None


def _setup_encryption():
    global _KEY
    pw = os.environ.get("APP_PASSWORD", "")
    lock = OUT / "lock.json"
    if not pw:
        if lock.exists():
            lock.unlink()
        return
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt = os.urandom(16)
    _KEY = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITER).derive(pw.encode())
    lock.write_text(json.dumps({"v": 1, "salt": base64.b64encode(salt).decode(), "iter": PBKDF2_ITER,
                                "check": _encrypt(b"nflalgo-ok")}))


def _encrypt(data: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = os.urandom(12)
    ct = AESGCM(_KEY).encrypt(iv, data, None)
    return {"iv": base64.b64encode(iv).decode(), "ct": base64.b64encode(ct).decode()}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def _write(name, obj):
    p = OUT / name
    raw = json.dumps(_clean(obj), separators=(",", ":"))
    if _KEY is not None:
        raw = json.dumps(_encrypt(raw.encode()), separators=(",", ":"))
    p.write_text(raw)
    return p.stat().st_size


def model_params(model):
    sc, rg = model.pipe.named_steps["standardscaler"], model.pipe.named_steps["ridge"]
    return {"feats": model.feats, "mean": list(sc.mean_), "scale": list(sc.scale_), "coef": list(rg.coef_),
            "intercept": float(rg.intercept_), "sigma": model.sigma, "labels": [H.FEATURE_LABELS.get(f, f) for f in model.feats]}


def week_file(season, week):
    E = H.ENGINE
    payload = H.week_payload(season, week)
    if not payload["games"]:
        return None
    df = E.df
    wk = df[(df.season == season) & (df.week == week)]
    model = E.fit_before(wk.gameday.min())
    payload["model"] = model_params(model)
    # QB options need the "new starter" flag too, so the phone can recompute both QB features
    fw = dict(zip(wk.game_id, wk.final_week))
    for g in payload["games"]:
        g["final_week"] = bool(fw.get(g["game_id"], 0))
        for side, team in (("home", g["home"]), ("away", g["away"])):
            pj = E.pl["qb_proj"].get((g["game_id"], team))
            if pj and g[f"{side}_qb"].get("options"):
                newflag = {o["pfr"]: o["new"] for o in pj["options"]}
                for o in g[f"{side}_qb"]["options"]:
                    o["new"] = bool(newflag.get(o["pfr"], 0))
        g.pop("date", None)
    return payload


def players_files():
    E = H.ENGINE
    pt = H._players_table()
    t = pt["table"]
    t = t[(t.snap >= 0.2) | (t.games >= 2)].sort_values("impact", ascending=False)
    rows = []
    for r in t.to_dict("records"):
        s = pt["spark"].get(r["pid"])
        rows.append({"pid": r["pid"], "name": r["name"], "pos": r["pos"], "g": r["group"], "team": r["team"],
                     "imp": r["impact"], "tr": r["trend"], "snap": r["snap"], "gp": r["games"],
                     "prod": r.get("prod"), "pk": r.get("prod_kind"),
                     "sp": [round(float(v), 2) for v in s[r["unit"]].values] if s is not None else []})
    reg = t[(t.snap >= 0.5) & t.trend.notna()]
    movers = {}
    for grp in ["ALL"] + sorted(t.group.unique()):
        g = reg if grp == "ALL" else reg[reg.group == grp]
        movers[grp] = {"up": list(g.nlargest(5, "trend").pid), "down": list(g.nsmallest(5, "trend").pid)}

    # history for each exported player
    pids = set(t.pid)
    hist = E.pl["history"]
    season = E.season
    hh = hist[hist.pid.isin(pids) & (hist.season >= season - 2)].sort_values(["season", "week"])
    unit = dict(zip(t.pid, t.unit))
    out = {}
    for pid, g in hh.groupby("pid"):
        u = unit[pid]
        out[pid] = {"imp": [[int(a), int(b), round(float(v), 2)] for a, b, v in zip(g.season, g.week, g[u])]}
    ph = pt["pr_hist"]
    ph = ph[ph.season >= season - 2]
    pf2gs = pt["pf2gs"]
    for r in t.itertuples(index=False):
        kind = getattr(r, "prod_kind", None)
        gs = pf2gs.get(r.pid)
        if isinstance(kind, str) and gs:
            q = ph[(ph.pid == gs) & (ph.kind == kind)]
            out.setdefault(r.pid, {})["prod"] = [[int(a), int(b), round(float(v) * H.PROD_SCALE[kind], 2)]
                                                 for a, b, v in zip(q.season, q.week, q.rating)]
    sn = H.P.load_snaps(H.DATA, [season - 1, season])
    sn = sn[sn.pfr_player_id.isin(pids)].sort_values(["season", "week"])
    for pid, g in sn.groupby("pfr_player_id"):
        u = unit[pid]
        g = g.tail(8)
        out.setdefault(pid, {})["sn"] = [[int(a), int(b), o, round(float(x if u == "off" else y), 2)]
                                        for a, b, o, x, y in zip(g.season, g.week, g.opponent, g.offense_pct, g.defense_pct)]
    labels = {k: v for k, v in H.PROD_LABEL.items()}
    return {"season": season, "players": rows, "movers": movers, "prod_labels": labels}, out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    _setup_encryption()
    E = H.ENGINE
    E.ensure()
    if E.error:
        raise SystemExit(E.error)
    season = E.season
    st = H.status({})
    sizes = {}
    weeks = []
    for w in st["weeks"]:
        p = week_file(season, w["week"])
        if p is None:
            continue
        sizes[f"week_{season}_{w['week']:02d}.json"] = _write(f"week_{season}_{w['week']:02d}.json", p)
        weeks.append({**w, "file": f"week_{season}_{w['week']:02d}.json"})
    rec = H.season_record(season)
    meta = {"season": season, "weeks": weeks, "default_week": st["default_week"],
            "last_played_week": st["last_played_week"], "injury_report_week": st.get("injury_report_week"),
            "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "record": rec,
            "model": "v4"}
    sizes["power.json"] = _write("power.json", H.rankings())
    players, hist = players_files()
    sizes["players.json"] = _write("players.json", players)
    sizes["history.json"] = _write("history.json", hist)
    sizes["meta.json"] = _write("meta.json", meta)
    total = sum(sizes.values())
    print(f"[app] wrote {len(sizes)} files, {total / 1e6:.1f} MB to {OUT} in {time.time() - t0:.0f}s"
          + (" (encrypted)" if _KEY is not None else ""))


if __name__ == "__main__":
    main()
