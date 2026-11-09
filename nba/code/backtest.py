#!/usr/bin/env python3
"""
Walk-forward backtest: each month is predicted by a model trained only on games before it.
Compares Elo only, team ratings only and the full model, and the full model vs. the closing line.

Usage: python backtest.py [--start 2018]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import model as M
import pipeline as PL


def walk_forward(df, feats, start):
    d = df[df.played]
    out = []
    tests = d[d.season >= start]
    for _, sub in tests.groupby(tests.date.dt.to_period("M")):
        tr = d[(d.date < sub.date.min()) & (d.date >= PL.TRAIN_START)]
        m = M.MarginModel(feats).fit(tr)
        out.append(pd.DataFrame({"game_id": sub.game_id, "season": sub.season, "p": m.predict_proba(sub), "y": sub.home_win,
                                 "mkt": sub.market_home_prob}))
    return pd.concat(out)


def score(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(), ((p > 0.5) == (y == 1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    args = ap.parse_args()
    E = PL.ENGINE.ensure()
    for name, feats in (("Elo only", M.ELO_FEATS), ("Team ratings", M.TEAM_FEATS), ("Full model", M.STAT_FEATS)):
        r = walk_forward(E.df, feats, args.start)
        ll, acc = score(r.p, r.y)
        print(f"{name:14s} log loss {ll:.4f}  winners {acc:.1%}  ({len(r)} games)")
    by = r.groupby("season").apply(lambda s: pd.Series(score(s.p, s.y), index=["ll", "acc"]))
    print(by.round(4).to_string())
    v = r[r.mkt.notna()]
    print(f"vs closing line on {len(v)} games: model {score(v.p, v.y)[0]:.4f}, market {score(v.mkt, v.y)[0]:.4f}")


if __name__ == "__main__":
    main()
