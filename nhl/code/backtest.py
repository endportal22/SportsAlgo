#!/usr/bin/env python3
"""
Walk-forward backtest: for each test season S, train only on seasons before S, predict
every game in S using features known before puck drop, then score.

Compares:
  home_rate     always predict the historical home-win rate (the floor to beat)
  elo           Elo rating + home ice only
  team          team ratings only: Elo, 5-on-5 xG, special teams, rest (no goalies, no lineups)
  model         team ratings + starting goalie + injuries (the current model)
  market        betting market alone (vig-free moneyline), on games that have a line
  model+market  model features plus the market line

The first season downloaded is a burn-in for the ratings and is never scored or trained on.

Usage: python backtest.py [--data-dir data] [--first-test 2021]
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import pipeline
from model import ELO_FEATS, MARKET_FEATS, STAT_FEATS, TEAM_FEATS, MarginModel
from paths import DATA, REPORTS

warnings.filterwarnings("ignore")
MODELS_ALL = ["home_rate", "elo", "team", "model"]
MODELS_MKT = ["model", "market", "model+market"]


def score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "accuracy": float(np.mean((p > 0.5) == (y == 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--first-test", type=int, default=None)
    ap.add_argument("--out", default=str(REPORTS / "backtest_results.json"))
    args = ap.parse_args()

    df, _, _ = pipeline.build(args.data_dir, verbose=True)
    df = df[df.played].copy()
    df["home_win"] = df.home_win.astype(int)
    first = int(df.season.min())
    first_test = args.first_test or first + 2
    last = int(df.season.max())
    preds = []
    for S in range(first_test, last + 1):
        tr = df[(df.season < S) & (df.season > first)]
        te = df[df.season == S].copy()
        if len(tr) < 300 or te.empty:
            continue
        out = te[["game_id", "season", "gameday", "home_team", "away_team", "home_win", "market_home_prob"]].copy()
        out["home_rate"] = tr.home_win.mean()
        out["elo"] = MarginModel(ELO_FEATS).fit(tr).predict_proba(te)
        out["team"] = MarginModel(TEAM_FEATS).fit(tr).predict_proba(te)
        out["model"] = MarginModel(STAT_FEATS).fit(tr).predict_proba(te)
        mk_tr = tr.dropna(subset=MARKET_FEATS)
        if len(mk_tr) >= 200:
            lr = LogisticRegression(max_iter=1000).fit(mk_tr[MARKET_FEATS], mk_tr.home_win)
            out["market"] = np.where(te.market_logit.notna(), lr.predict_proba(te[MARKET_FEATS].fillna(0))[:, 1], np.nan)
            out["model+market"] = np.where(te.market_logit.notna(),
                                           MarginModel(STAT_FEATS + MARKET_FEATS).fit(mk_tr).predict_proba(te.fillna({"market_logit": 0})),
                                           np.nan)
        else:   # no history of lines to calibrate on: use the vig-free line as is
            out["market"] = te.market_home_prob.values
            out["model+market"] = np.nan
        preds.append(out)
    if not preds:
        print("Not enough seasons to backtest yet (need at least three downloaded).")
        return
    P = pd.concat(preds, ignore_index=True)

    rows = []
    full = [s for s in sorted(P.season.unique()) if s < last] or [last]
    pooled = f"{full[0]}-{full[-1]}"
    for S, grp in list(P.groupby("season")) + [(pooled, P[P.season.isin(full)])]:
        for m in MODELS_ALL:
            rows.append({"season": S, "subset": "all", "model": m, **score(grp.home_win.values, grp[m].values)})
        wl = grp[grp.market.notna()]
        if len(wl) >= 30:
            for m in MODELS_MKT:
                if wl[m].notna().all():
                    rows.append({"season": S, "subset": "with_line", "model": m, **score(wl.home_win.values, wl[m].values)})
    R = pd.DataFrame(rows)
    R["season"] = R.season.astype(str)

    pd.set_option("display.width", 160)
    print(f"\n=== Pooled {pooled}, all games (lower log loss / brier is better; a coin flip is 0.693) ===")
    print(R[(R.season == pooled) & (R.subset == "all")].set_index("model")[["n", "log_loss", "brier", "accuracy"]].round(4))
    wl = R[(R.season == pooled) & (R.subset == "with_line")]
    if len(wl):
        print(f"\n=== Pooled {pooled}, games with a betting line ===")
        print(wl.set_index("model")[["n", "log_loss", "brier", "accuracy"]].round(4))
    print("\n=== Log loss by season (all games) ===")
    print(R[(R.season != pooled) & (R.subset == "all")].pivot(index="season", columns="model", values="log_loss")[MODELS_ALL].round(4))

    REPORTS.mkdir(parents=True, exist_ok=True)
    P.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    meta = {"pooled": pooled, "current_season": str(last), "rows": R.to_dict("records")}
    with open(args.out, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
