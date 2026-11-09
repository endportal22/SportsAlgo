#!/usr/bin/env python3
"""
Predict upcoming games with the saved model.

By default predicts the next day with games. Output CSV has, per game: projected
starting goalies, injury cost, model goal margin, model home win prob, the betting
market's (vig-free) home win prob, and the gap between them ("edge").

Usage:
  python predict.py                      # next game day
  python predict.py --date 2026-10-20    # a specific day
  python predict.py --days 7             # the next 7 days
  python predict.py --all                # every remaining game this season
"""
import argparse
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pipeline
from model import STAT_FEATS, MarginModel
from paths import DATA, MODELS, PREDICTIONS, today
from train import train_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--model", default=str(MODELS / "margin_model.joblib"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="all remaining games this season")
    ap.add_argument("--out-dir", default=str(PREDICTIONS))
    args = ap.parse_args()

    df, _, pl = pipeline.build(args.data_dir, verbose=True)
    fen = pl["fen_per_game"]
    mp = Path(args.model)
    model = joblib.load(mp) if mp.exists() else MarginModel(STAT_FEATS).fit(train_set(df))
    season = int(df.season.max())
    up = df[(df.season == season) & ~df.played & (df.gameday >= pd.Timestamp(today()))]
    if up.empty:
        print(f"No unplayed games left in {season}.")
        return
    if args.all:
        tag = f"{season}_remaining"
    elif args.days:
        d0 = up.gameday.min()
        up = up[up.gameday < d0 + timedelta(days=args.days)]
        tag = f"{d0.date()}_next_{args.days}_days"
    else:
        d = pd.Timestamp(args.date) if args.date else up.gameday.min()
        up = up[up.gameday == d]
        tag = f"{d.date()}"
        if up.empty:
            print(f"No unplayed games on {d.date()}.")
            return

    out = up[["season", "gameday", "away_team", "home_team", "away_goalie", "home_goalie"]].copy()
    out["gameday"] = out.gameday.dt.date
    out["away_goalie_gsax_per_game"] = (up["a_goalie_rating"] * fen).round(2)
    out["home_goalie_gsax_per_game"] = (up["h_goalie_rating"] * fen).round(2)
    out["away_injury_cost_goals"] = up["a_injury_rel"].round(2)
    out["home_injury_cost_goals"] = up["h_injury_rel"].round(2)
    out["away_back_to_back"] = up["a_b2b"].astype(int)
    out["home_back_to_back"] = up["h_b2b"].astype(int)
    out["model_home_margin"] = model.predict_margin(up).round(2)
    out["model_home_win_prob"] = model.predict_proba(up).round(3)
    out["market_home_win_prob"] = up["market_home_prob"].round(3)
    out["home_moneyline"] = up["home_ml"]
    out["away_moneyline"] = up["away_ml"]
    out["edge"] = (out["model_home_win_prob"] - out["market_home_win_prob"]).round(3)
    out["pick"] = np.where(out["model_home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out = out.sort_values(["gameday", "model_home_win_prob"], ascending=[True, False])

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    csv = od / f"predictions_{tag}.csv"
    out.to_csv(csv, index=False)
    show = out[["gameday", "away_team", "home_team", "model_home_margin", "model_home_win_prob", "market_home_win_prob", "edge"]]
    print(show.to_string(index=False))
    print(f"\nsaved -> {csv.resolve()}")
    if not args.all and not args.days:
        plot(out, od / f"predictions_{tag}.png", f"NHL {tag}")


def plot(out, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping chart)")
        return
    d = out.sort_values("model_home_win_prob").reset_index(drop=True)
    labels = d["away_team"] + " @ " + d["home_team"]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    for i, r in d.iterrows():
        if pd.notna(r.market_home_win_prob):
            ax.plot([r.market_home_win_prob, r.model_home_win_prob], [i, i], color="#bbb", lw=2, zorder=1)
    ax.scatter(d["market_home_win_prob"], y, color="#999", s=36, label="Betting market", zorder=2)
    ax.scatter(d["model_home_win_prob"], y, color="#1f7fbf", s=48, label="Model", zorder=3)
    ax.axvline(0.5, color="#333", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Home win probability")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved -> {Path(path).resolve()}")


if __name__ == "__main__":
    main()
