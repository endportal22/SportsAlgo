#!/usr/bin/env python3
"""
Test features that aren't in the model, and (optionally) tune the team-rating settings.

Walk-forward: for each test season S, train on seasons before S (after the burn-in season)
and predict S. Prints log loss for the current model and for each addition. Lower is better;
differences under ~0.001 are noise.

Usage:
  python experiments.py              # test candidate features
  python experiments.py --tune       # also try other decay / regression settings
  python experiments.py --tune --save   # ... and save the best to models/params.json
                                         (the HUD and every script pick it up automatically)
"""
import argparse
import itertools
import json
import warnings

import numpy as np
from sklearn.metrics import log_loss

import context as C
import features as F
import pipeline
import xg as XG
from model import STAT_FEATS, MarginModel
from paths import DATA, MODELS

warnings.filterwarnings("ignore")


def run(df, feats, seasons, target="target"):
    first = int(df.season.min())
    ys, ps = [], []
    base = df[df.played].copy()
    if target != "target":
        base["target"] = base[target]
    for s in seasons:
        tr, te = base[(base.season < s) & (base.season > first)], base[base.season == s]
        if len(tr) < 300 or te.empty:
            continue
        ys += list(te.home_win)
        ps += list(MarginModel(feats).fit(tr).predict_proba(te))
    if not ys:
        return float("nan"), float("nan")
    return log_loss(ys, ps), float(np.mean((np.array(ps) > 0.5) == (np.array(ys) == 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df, _, _ = pipeline.build(DATA, verbose=True)
    df = C.add_context(df)
    first, last = int(df.season.min()), int(df.season.max())
    seasons = list(range(first + 2, last + 1))
    if not seasons:
        print("Need at least three seasons of data.")
        return
    mid = seasons[: max(1, len(seasons) // 2)]
    late = seasons[len(mid):] or seasons
    tests = {
        "current model": [],
        "+ travel distance": ["d_travel"],
        "+ time zones": ["d_tz"],
        "+ road-trip length": ["road_trip"],
        "+ goal-differential form": ["d_gd"],
        "+ finishing / save luck (G - xG)": ["d_fin"],
        "+ full lineup value": ["d_lineup"],
        "+ playoff game": ["playoff"],
        "- injuries": None,
        "- special teams": None,
    }
    lab = lambda ss: f"{ss[0]}-{str(ss[-1])[-2:]}" if len(ss) > 1 else str(ss[0])
    print(f"{'':36s} {lab(seasons):>16s} {lab(mid):>9s} {lab(late):>9s}")
    for name, extra in tests.items():
        if extra is None:
            drop = {"- injuries": "d_injury_rel", "- special teams": "d_st"}[name]
            fs = [f for f in STAT_FEATS if f != drop]
        else:
            fs = STAT_FEATS + extra
        a, acc = run(df, fs, seasons)
        b, _ = run(df, fs, mid)
        c, _ = run(df, fs, late)
        print(f"{name:36s} {a:.4f} ({acc:.1%}) {b:9.4f} {c:9.4f}")

    if not args.tune:
        return
    # ---- tune team-rating settings: rebuild only the team features for each setting
    print("\nTuning team ratings (decay, regression to the mean, carry-over) ...")
    seas = sorted(df.season.unique())
    games = F.load_games(DATA)
    games = games[games.season >= first].reset_index(drop=True)
    shots, _ = XG.add_xg(F.load_kind(DATA, "shots", seas))
    tstats = F.team_game_stats(shots, F.load_kind(DATA, "states", seas))
    team_cols = ["d_elo", "d_ev", "d_st", "d_b2b", "rest_diff"]
    keep = df.drop(columns=team_cols).set_index("game_id")
    best = None
    grid = itertools.product([0.93, 0.95, 0.97], [0.5, 0.65, 0.8], [6.0, 10.0, 16.0])
    for decay, regress, carry in grid:
        p = F.Params(decay=decay, regress=regress, carry_weight=carry)
        tf = F.build_features(games, tstats, p)[["game_id"] + team_cols].set_index("game_id")
        d2 = keep.join(tf, how="inner").reset_index()
        ll, _ = run(d2, STAT_FEATS, seasons)
        print(f"  decay {decay:.2f}  regress {regress:.2f}  carry {carry:4.0f}   log loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, decay, regress, carry)
    print(f"best: decay {best[1]}, regress {best[2]}, carry {best[3]} (log loss {best[0]:.4f})")
    if args.save:
        MODELS.mkdir(parents=True, exist_ok=True)
        (MODELS / "params.json").write_text(json.dumps({"decay": best[1], "regress": best[2], "carry_weight": best[3]}, indent=2))
        print(f"saved -> {MODELS / 'params.json'}")


if __name__ == "__main__":
    main()
