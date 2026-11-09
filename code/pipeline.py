"""
One call that builds every per-game feature: team ratings (features.py) plus player
ratings, lineups and injuries (players.py). Used by train, predict, backtest and the HUD.
"""
from __future__ import annotations

import features as F
import players as P
from paths import DATA, MODELS, PREDICTIONS, REPORTS  # noqa: E402


def build(data_dir=DATA, verbose=False):
    """Returns (games_df, league_state, players_bundle)."""
    df, league = F.build_all(data_dir, return_league=True)
    pl = P.build(data_dir, verbose=verbose)
    df = P.game_features(pl, df)
    return df, league, pl
