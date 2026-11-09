#!/usr/bin/env python3
"""
Walk-forward backtest: for each test season S, train only on seasons < S, predict
every game in S using features known before kickoff, then score.

Compares:
  home_rate      always predict the historical home-win rate (the floor to beat)
  original       your v1 model (L3/L5 rolling diffs, RandomForest + isotonic)
  v2             team ratings + new-QB flag, ridge on point margin -> win prob
  v3             v2 + player ratings: QB rating, injuries, pass/run matchups
  v4             v3 + roster-based strength and final-week adjustment (current model)
  market         Vegas spread alone (the ceiling most models never reach)
  v4+market      v4 features plus the spread

Usage: python backtest.py [--data-dir data] [--first-test 2019]
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import pipeline
from model import MarginModel, STAT_FEATS, MARKET_FEATS, V2_FEATS, V3_FEATS
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402

warnings.filterwarnings("ignore")


# ------------------------------------------------------------- replica of v1 features
def original_features(data_dir, games):
    pbp = F.load_pbp(data_dir, sorted(games["season"].unique()))
    pbp = pbp[pbp["play_type"].notna()].copy()
    for c in ("posteam", "defteam"):
        pbp[c] = pbp[c].replace(F.TEAM_FIX)
    pbp["succ"] = pbp["success"].fillna(0)
    pbp["expl"] = (pbp["yards_gained"] >= 15).astype(float)
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(float)
    o = pbp[pbp.posteam.notna()].groupby(["game_id", "posteam"]).agg(
        off_sr=("succ", "mean"), off_epa=("epa", "mean"), off_ex=("expl", "mean"), off_pr=("is_pass", "mean"))
    o.index.names = ["game_id", "team"]
    d = pbp[pbp.defteam.notna()].groupby(["game_id", "defteam"]).agg(
        def_sr=("succ", "mean"), def_epa=("epa", "mean"), def_ex=("expl", "mean"))
    d.index.names = ["game_id", "team"]
    tg = o.join(d, how="inner").reset_index().merge(games[["game_id", "gameday"]], on="game_id")
    tg = tg.sort_values(["team", "gameday"])
    grp = tg.groupby("team")
    for col, w in [("off_epa", 5), ("off_epa", 3), ("off_sr", 5), ("off_sr", 3), ("off_ex", 5), ("off_pr", 5),
                   ("def_epa", 5), ("def_epa", 3), ("def_sr", 5), ("def_ex", 5)]:
        tg[f"{col}_l{w}"] = grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    tg["rest"] = grp["gameday"].diff().dt.days
    roll = [c for c in tg.columns if c.endswith(("_l3", "_l5"))] + ["rest"]
    t = tg.set_index(["game_id", "team"])[roll]
    gm = games[games.played][["game_id", "season", "home_team", "away_team", "home_win"]]
    hj = t.add_prefix("h_").reset_index().rename(columns={"team": "home_team"})
    aj = t.add_prefix("a_").reset_index().rename(columns={"team": "away_team"})
    m = gm.merge(hj, on=["game_id", "home_team"]).merge(aj, on=["game_id", "away_team"])
    feats = []
    for c in roll:
        m[f"d_{c}"] = m[f"h_{c}"] - m[f"a_{c}"]
        feats.append(f"d_{c}")
    m = m.dropna(subset=["d_off_epa_l3"])
    m["d_rest"] = m["d_rest"].fillna(0)
    return m, feats


def fit_original(tr, feats):
    """Same recipe as v1 train_nfl.py: RF fit on first 80%, isotonic on last 20%."""
    tr = tr.dropna(subset=feats)
    n = int(len(tr) * 0.8)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(tr[feats].iloc[:n], tr["home_win"].iloc[:n])
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic")
    except ImportError:
        cal = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
    cal.fit(tr[feats].iloc[n:], tr["home_win"].iloc[n:])
    return cal


# ------------------------------------------------------------- scoring
def score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(y), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "accuracy": float(np.mean((p > 0.5) == (y == 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--first-test", type=int, default=2019)
    ap.add_argument("--out", default=str(REPORTS / "backtest_results.json"))
    args = ap.parse_args()

    games = F.load_games(args.data_dir)
    games = games[games.season >= 2016]
    feats, _, _ = pipeline.build(args.data_dir)
    feats = feats[feats.played & (feats.home_win != 0.5)]
    feats["home_win"] = feats["home_win"].astype(int)
    orig, ofeats = original_features(args.data_dir, games)
    orig = orig[orig.home_win != 0.5]
    orig["home_win"] = orig["home_win"].astype(int)

    last = int(feats.season.max())
    preds = []
    for S in range(args.first_test, last + 1):
        tr = feats[(feats.season < S) & (feats.season >= 2017)]   # 2016 = burn-in for ratings
        te = feats[feats.season == S].copy()
        out = te[["game_id", "season", "week", "home_team", "away_team", "home_win", "spread_line"]].copy()

        out["home_rate"] = tr.home_win.mean()

        otr, ote = orig[orig.season < S], orig[orig.season == S]
        om = fit_original(otr, ofeats)
        out = out.merge(pd.DataFrame({"game_id": ote.game_id, "original": om.predict_proba(ote[ofeats])[:, 1]}),
                        on="game_id", how="left")

        out["v2"] = MarginModel(V2_FEATS).fit(tr).predict_proba(te)
        out["v3"] = MarginModel(V3_FEATS).fit(tr).predict_proba(te)
        out["v4"] = MarginModel(STAT_FEATS).fit(tr).predict_proba(te)

        mk_tr, mk_te = tr.dropna(subset=MARKET_FEATS), te
        mk = LogisticRegression(max_iter=1000).fit(mk_tr[MARKET_FEATS], mk_tr.home_win)
        out["market"] = mk.predict_proba(mk_te[MARKET_FEATS].fillna(0))[:, 1]

        out["v4+market"] = MarginModel(STAT_FEATS + MARKET_FEATS).fit(mk_tr).predict_proba(
            te.fillna({"spread_line": 0}))
        preds.append(out)

    P = pd.concat(preds, ignore_index=True)
    models = ["home_rate", "original", "v2", "v3", "v4", "market", "v4+market"]
    P = P.dropna(subset=["original"])  # same games for every model

    rows = []
    for S, grp in P.groupby("season"):
        for m in models:
            rows.append({"season": S, "model": m, **score(grp.home_win.values, grp[m].values)})
    hist = P[P.season < last]
    for m in models:
        rows.append({"season": f"{args.first_test}-{last-1}", "model": m, **score(hist.home_win.values, hist[m].values)})
    R = pd.DataFrame(rows)

    pd.set_option("display.width", 160)
    summ = R[R.season == f"{args.first_test}-{last-1}"].set_index("model")[["n", "log_loss", "brier", "accuracy"]]
    print(f"\n=== Pooled {args.first_test}-{last-1} (lower log loss / brier is better) ===")
    print(summ.round(4))
    print("\n=== Log loss by season ===")
    print(R[R.season != f"{args.first_test}-{last-1}"].pivot(index="season", columns="model", values="log_loss")[models].round(4))
    print("\n=== Accuracy by season ===")
    print(R[R.season != f"{args.first_test}-{last-1}"].pivot(index="season", columns="model", values="accuracy")[models].round(3))

    REPORTS.mkdir(exist_ok=True)
    P.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    R.to_json(args.out, orient="records", indent=1)


if __name__ == "__main__":
    main()
