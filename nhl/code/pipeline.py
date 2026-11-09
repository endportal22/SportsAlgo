"""
One call that builds every per-game feature: expected goals (xg.py), team ratings
(features.py), plus skater ratings, lineups, injuries and goalies (players.py), plus
the betting line. Used by train, predict, backtest and the HUD.

The result is cached in data/cache/ and rebuilt only when the data or the code changes
(Update data always changes the data, so it always triggers a rebuild).
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import players as P
import xg as XG
from paths import CACHE, CODE, DATA, MODELS

CACHE_VERSION = 3


def _seasons(data_dir):
    return sorted(int(f.name.split("_")[1].split(".")[0]) for f in Path(data_dir).glob("states_*.csv.gz"))


def _fingerprint(data_dir):
    h = hashlib.sha1(str(CACHE_VERSION).encode())
    for f in sorted(Path(data_dir).glob("*.csv*")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    for f in sorted(CODE.glob("*.py")):
        h.update(f.read_bytes())
    pj = MODELS / "params.json"
    if pj.exists():
        h.update(pj.read_bytes())
    return h.hexdigest()[:16]


def load_odds(data_dir) -> pd.DataFrame:
    f = Path(data_dir) / "odds.csv"
    if not f.exists():
        return pd.DataFrame(columns=["game_id", "market_home_prob", "home_ml", "away_ml", "book"])
    o = pd.read_csv(f)
    for c in ("market_home_prob", "home_ml", "away_ml", "book"):
        if c not in o:
            o[c] = np.nan
    return o[["game_id", "market_home_prob", "home_ml", "away_ml", "book"]]


def build(data_dir=DATA, verbose=False, use_cache=True):
    """Returns (games_df, league_state, players_bundle)."""
    data_dir = Path(data_dir)
    key = _fingerprint(data_dir)
    cf = CACHE / f"build_{key}.pkl"
    if use_cache and cf.exists():
        try:
            with open(cf, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    seasons = _seasons(data_dir)
    if not seasons:
        raise FileNotFoundError("No game data yet. Press Update data (or run update_data.py).")
    first = seasons[0]
    games = F.load_games(data_dir)
    games = games[games.season >= first].reset_index(drop=True)
    shots = F.load_kind(data_dir, "shots", seasons)
    shots, xg_model = XG.add_xg(shots)
    states = F.load_kind(data_dir, "states", seasons)
    skaters = F.load_kind(data_dir, "skaters", seasons)
    goalies = F.load_kind(data_dir, "goalies", seasons)
    if verbose:
        print(f"[pipeline] loaded {len(shots):,} shots, {len(states) // 2:,} games, xG fit ({time.time() - t0:.1f}s)", flush=True)

    tstats = F.team_game_stats(shots, states)
    df, league = F.build_features(games, tstats, F.Params.load(), return_league=True)
    F.EV_MIN = float(league.ev_min)
    pl = P.build(data_dir, games, skaters, goalies, shots, tstats, verbose=verbose)
    df = P.game_features(pl, df)

    # target: goal margin without empty-net goals or the shootout
    ts = tstats.set_index(["game_id", "team"])
    hg = ts.gf.reindex(pd.MultiIndex.from_arrays([df.game_id, df.home_team])).values
    ag = ts.gf.reindex(pd.MultiIndex.from_arrays([df.game_id, df.away_team])).values
    tgt = hg - ag
    df["target"] = np.where(df.played & ~np.isnan(tgt), tgt, np.where(df.played, df.margin, np.nan))

    odds = load_odds(data_dir)
    df = df.merge(odds, on="game_id", how="left")
    p = df["market_home_prob"].astype(float).clip(0.02, 0.98)
    df["market_logit"] = np.log(p / (1 - p))
    pl["xg_model"] = xg_model
    pl["shots_by_game"] = shots.groupby("game_id").size().to_dict()
    out = (df, league, pl)
    if verbose:
        print(f"[pipeline] built in {time.time() - t0:.1f}s", flush=True)
    if use_cache:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            for old in CACHE.glob("build_*.pkl"):
                old.unlink()
            with open(cf, "wb") as fh:
                pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001
            pass
    return out
