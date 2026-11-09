#!/usr/bin/env python3
"""
Re-run the context-feature experiment (special teams, travel, time zones, early kickoffs, wind).

Walk-forward 2019-2025: train on seasons before S, predict S. Prints log loss for the
current model and for each addition. None of these beat the current model as of Oct 2026,
so they are not part of STAT_FEATS; re-run after more seasons to see if that changes.

Usage: python experiments.py
"""
import warnings

import numpy as np
from sklearn.metrics import log_loss

import context as C
import features as F
import pipeline
from model import STAT_FEATS, MarginModel
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402

warnings.filterwarnings("ignore")


def run(df, feats, lo, hi):
    ys, ps = [], []
    for s in range(lo, hi + 1):
        base = df[df.played & (df.home_win != 0.5)]
        tr, te = base[(base.season < s) & (base.season >= 2017)], base[base.season == s]
        ys += list(te.home_win)
        ps += list(MarginModel(feats).fit(tr).predict_proba(te))
    return log_loss(ys, ps), float(np.mean((np.array(ps) > 0.5) == (np.array(ys) == 1)))


def main():
    df, _, _ = pipeline.build(DATA)
    games = F.load_games(DATA)
    games = games[games.season >= 2016]
    df = C.add_context(df, games, F.load_pbp(DATA, sorted(games.season.unique())))
    tests = {
        "current model": [],
        "+ special teams": ["d_st"],
        "+ travel distance": ["d_travel"],
        "+ time zones": ["d_tz"],
        "+ early body-clock kickoff": ["d_early"],
        "+ wind x passing": ["wind_x_pass"],
        "+ all of the above": ["d_st", "d_travel", "d_tz", "d_early", "wind_x_pass"],
    }
    print(f"{'':28s} {'2019-25':>16s} {'2019-22':>8s} {'2023-25':>8s}")
    for name, extra in tests.items():
        fs = STAT_FEATS + extra
        a, acc = run(df, fs, 2019, 2025)
        b, _ = run(df, fs, 2019, 2022)
        c, _ = run(df, fs, 2023, 2025)
        print(f"{name:28s} {a:.4f} ({acc:.1%}) {b:8.4f} {c:8.4f}")


if __name__ == "__main__":
    main()
