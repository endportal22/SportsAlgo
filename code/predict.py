#!/usr/bin/env python3
"""
Predict upcoming games with the saved v2 model.

By default predicts the next unplayed week of the current season. Output CSV has,
per game: model home margin, model home win prob, Vegas spread, market-implied
win prob, and the gap between them ("edge").

Usage:
  python predict.py                      # next week
  python predict.py --week 7             # a specific week of the current season
  python predict.py --all                # every remaining game this season
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

import pipeline
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--model", default=str(MODELS / "margin_model.joblib"))
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="all remaining games this season")
    ap.add_argument("--out-dir", default=str(PREDICTIONS))
    args = ap.parse_args()

    model = joblib.load(args.model)
    df, _, _ = pipeline.build(args.data_dir)
    season = args.season or int(df.loc[df.played, "season"].max())
    up = df[(df.season == season) & ~df.played]
    if up.empty:
        print(f"No unplayed games left in {season}.")
        return
    if not args.all:
        week = args.week or int(up.week.min())
        up = up[up.week == week]
        tag = f"{season}_week_{week:02d}"
    else:
        tag = f"{season}_remaining"

    out = up[["season", "week", "gameday", "away_team", "home_team", "away_qb", "home_qb",
              "h_qb_new", "a_qb_new", "spread_line"]].copy()
    out["away_injury_cost_pts"] = up["a_injury_rel"].round(1)
    out["home_injury_cost_pts"] = up["h_injury_rel"].round(1)
    out["model_home_margin"] = model.predict_margin(up).round(1)
    out["model_home_win_prob"] = model.predict_proba(up).round(3)
    out["market_home_win_prob"] = norm.cdf(up["spread_line"] / model.sigma).round(3)
    out["edge"] = (out["model_home_win_prob"] - out["market_home_win_prob"]).round(3)
    out["pick"] = np.where(out["model_home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out = out.rename(columns={"spread_line": "vegas_home_spread", "h_qb_new": "home_new_qb", "a_qb_new": "away_new_qb"})
    out = out.sort_values(["gameday", "model_home_win_prob"], ascending=[True, False])

    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    csv = od / f"predictions_{tag}.csv"
    out.to_csv(csv, index=False)

    show = out[["away_team", "home_team", "model_home_margin", "vegas_home_spread",
                "model_home_win_prob", "market_home_win_prob", "edge"]]
    print(show.to_string(index=False))
    print(f"\nsaved -> {csv.resolve()}")

    if not args.all:
        plot(out, od / f"predictions_{tag}.png", f"{season} Week {week}")


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
    ax.scatter(d["market_home_win_prob"], y, color="#999", s=36, label="Vegas (from spread)", zorder=2)
    ax.scatter(d["model_home_win_prob"], y, color="#1f5fbf", s=48, label="Model", zorder=3)
    ax.axvline(0.5, color="#333", lw=0.8, ls=":")
    ax.set_yticks(y, labels)
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
