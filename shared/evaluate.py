"""
Long-run scorecard for the app's Record tab, the same for every league.

Every past game since the backtest start is predicted walk-forward (by the model trained only on
games before it) and scored three ways, next to Vegas:
  log loss   how good the probabilities are (lower is better; a coin flip scores 0.693)
  Brier      mean squared error of the probabilities (lower is better; a coin flip scores 0.25)
  winners    how often the favourite won
plus calibration ("when the model said 60-65%, how often did that team win?"), how the model does
when it disagrees with Vegas, and whether the gap to Vegas is bigger than chance (a block bootstrap
over weeks). The result is cached for the week, since it only changes as games are added.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
BUCKETS = ((.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .75), (.75, .8), (.8, .9), (.9, 1.01))


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _scores(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {"ll": round(float(_ll(p, y).mean()), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "winners": round(float(np.mean((p > 0.5) == (y == 1))), 4)}


def _calibration(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    fav = np.maximum(p, 1 - p)
    won = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for lo, hi in BUCKETS:
        m = (fav >= lo) & (fav < hi)
        if m.sum() >= 20:
            rows.append({"label": f"{int(lo * 100)}–{min(int(hi * 100), 100)}%", "n": int(m.sum()),
                         "said": round(float(fav[m].mean()), 3), "won": round(float(won[m].mean()), 3)})
    return rows


def scorecard(oos: pd.DataFrame, since_label: str):
    """oos: one row per past game with p (model, home win), y (1 if home won), p_mkt (Vegas, may be
    missing), block (week or month, for the significance test)."""
    o = oos.dropna(subset=["p", "y"])
    o = o[o.y.isin([0, 1])]
    if len(o) < 100:
        return None
    v = o[o.p_mkt.notna()]
    out = {"since": since_label, "n": int(len(o)), "model": _scores(o.p, o.y), "calibration": _calibration(o.p, o.y)}
    if len(v) >= 100:
        out["n_vegas"] = int(len(v))
        out["model_vs"] = _scores(v.p, v.y)
        out["vegas"] = _scores(v.p_mkt, v.y)
        out["vegas_calibration"] = _calibration(v.p_mkt, v.y)
        # is the log-loss gap bigger than chance? block bootstrap over weeks/months
        d = pd.DataFrame({"d": _ll(v.p, v.y) - _ll(v.p_mkt, v.y), "g": v.block.astype(str).values})
        gs = d.groupby("g").d.agg(["sum", "count"])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(gs), size=(2000, len(gs)))
        boots = gs["sum"].values[idx].sum(1) / gs["count"].values[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["gap"] = {"mean": round(float(d.d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
        dis = v[(v.p >= 0.5) != (v.p_mkt >= 0.5)]
        if len(dis):
            out["disagree"] = {"n": int(len(dis)), "model_right": round(float(((dis.p >= 0.5) == (dis.y == 1)).mean()), 3)}
        big = v[(v.p - v.p_mkt).abs() >= 0.10]
        if len(big) >= 20:
            # the side Vegas rated higher than the model did
            vegas_side_won = np.where(big.p_mkt > big.p, big.y, 1 - big.y)
            out["big_gap"] = {"n": int(len(big)), "vegas_side_won": round(float(vegas_side_won.mean()), 3),
                              "model_ll": round(float(_ll(big.p, big.y).mean()), 4),
                              "vegas_ll": round(float(_ll(big.p_mkt, big.y).mean()), 4)}
    return out


def cached(cache_dir, league, compute, key_extra=""):
    """Compute once per ISO week (and code version), reuse otherwise."""
    wk = pd.Timestamp.now().strftime("%G-%V")
    f = Path(cache_dir) / f"scorecard_{league}_{wk}{key_extra}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            pass
    res = compute()
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        for old in Path(cache_dir).glob(f"scorecard_{league}_*.json"):
            old.unlink()
        f.write_text(json.dumps(res))
    except Exception:  # noqa: BLE001
        pass
    return res
