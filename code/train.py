#!/usr/bin/env python3
"""
Train the v2 model on every completed game since 2017 and save it.

Usage: python train.py [--data-dir data] [--out models/margin_model.joblib]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import joblib

import pipeline
from model import MarginModel, STAT_FEATS
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out", default=str(MODELS / "margin_model.joblib"))
    args = ap.parse_args()

    df, _, _ = pipeline.build(args.data_dir)
    tr = df[df.played & (df.season >= 2017)]   # 2016 is burn-in for the ratings
    model = MarginModel(STAT_FEATS).fit(tr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    coefs, intercept = model.coefficients()
    meta = {
        "trained_on": date.today().isoformat(),
        "n_games": int(len(tr)),
        "seasons": [int(tr.season.min()), int(tr.season.max())],
        "features": STAT_FEATS,
        "home_margin_intercept": round(intercept, 2),
        "points_per_1sd_of_feature": coefs,
        "sigma_points": round(model.sigma, 2),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
