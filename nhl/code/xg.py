"""
Expected goals (xG): the chance that an unblocked shot becomes a goal, from where and how
it was taken. This is hockey's version of EPA: it rates chances instead of results, so it
settles down much faster than goals.

Logistic regression on every unblocked shot with a goalie in net (empty-net shots and
shootouts are left out), using: distance and angle to the net (with curves), shot type,
rebound (team's previous attempt <= 3 s earlier), rush (puck was outside the offensive
zone <= 4 s earlier), and power play / short-handed. Whether the shot hit the net is
deliberately not used, since that would leak the result.
Blocked shots get no xG (the NHL doesn't record where they would have gone).

The fitted model is cached in data/cache/ and refit whenever the shot files change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]


def design(s: pd.DataFrame) -> np.ndarray:
    d = s["dist"].astype(float).clip(1, 100).fillna(35.0).values
    a = s["angle"].astype(float).clip(0, 180).fillna(30.0).values
    st = s["shot_type"].astype(str).str.lower()
    cols = [d, np.log(d), d ** 2 / 100.0, a, (a / 90.0) ** 2, d * a / 1000.0, (a > 90).astype(float),
            s["rebound"].fillna(0).astype(float).values, s["rush"].fillna(0).astype(float).values,
            (s["strength"] == "PP").astype(float).values, (s["strength"] == "SH").astype(float).values,
            (s["strength"] == "EV").astype(float).values,
            s["rebound"].fillna(0).astype(float).values * a / 90.0]
    for t in SHOT_TYPES:
        cols.append((st == t).astype(float).values)
    return np.column_stack(cols)


def fit(shots: pd.DataFrame):
    s = shots[shots.ev.isin(["G", "S", "M"]) & (shots.opp_g == 1)]
    y = (s.ev == "G").astype(int).values
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    m.fit(design(s), y)
    return m


def add_xg(shots: pd.DataFrame, model=None) -> pd.DataFrame:
    """Adds an `xg` column: model xG for unblocked shots on a goalie, 0 for blocked shots.
    Empty-net shots get xg = NaN (they're handled separately)."""
    model = model or fit(shots)
    shots = shots.copy()
    unb = shots.ev.isin(["G", "S", "M"])
    on_g = unb & (shots.opp_g == 1)
    xg = np.zeros(len(shots))
    if on_g.any():
        xg[on_g.values] = model.predict_proba(design(shots[on_g]))[:, 1]
    xg[(unb & (shots.opp_g != 1)).values] = np.nan
    shots["xg"] = xg
    return shots, model
