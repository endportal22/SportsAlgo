#!/usr/bin/env python3
"""
Train the model on every completed game (after the first, burn-in season) and save it.

Usage: python train.py [--data-dir data] [--out models/margin_model.joblib]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import joblib

import pipeline
from model import STAT_FEATS, MarginModel
from paths import DATA, MODELS


def train_set(df):
    first = int(df.season.min())
    return df[df.played & (df.season > first)]   # first season = burn-in for the ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out", default=str(MODELS / "margin_model.joblib"))
    args = ap.parse_args()

    df, _, _ = pipeline.build(args.data_dir, verbose=True)
    tr = train_set(df)
    model = MarginModel(STAT_FEATS).fit(tr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    coefs, intercept = model.coefficients()
    meta = {
        "trained_on": date.today().isoformat(),
        "n_games": int(len(tr)),
        "seasons": [int(tr.season.min()), int(tr.season.max())],
        "features": STAT_FEATS,
        "home_margin_intercept": round(intercept, 3),
        "goals_per_1sd_of_feature": coefs,
        "win_prob_curve": {"a": round(model.a, 4), "b": round(model.b, 4),
                           "note": "home win prob = 1 / (1 + exp(-(a + b * predicted margin)))"},
        "margin_error_sd_goals": round(model.sigma, 3),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
