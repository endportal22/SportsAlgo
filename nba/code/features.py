"""Game features: everything the model sees about a game, as home-minus-away differences."""
from __future__ import annotations

import numpy as np
import pandas as pd

import ratings as R

FEATURE_LABELS = {
    "d_elo": "Elo", "d_net": "Recent form (net rating)", "d_pm": "Lineup plus-minus", "d_box": "Lineup box score",
    "d_lineup_delta": "Who's in / out", "d_b2b": "Back-to-back", "rest_diff": "Rest", "home_field": "Home court",
}


def rest_days(games):
    """Days since each team's previous game (scheduled or played), capped at 4."""
    rows = pd.concat([games[["game_id", "date", "home"]].rename(columns={"home": "team"}),
                      games[["game_id", "date", "away"]].rename(columns={"away": "team"})]).sort_values(["team", "date", "game_id"])
    rows["rest"] = rows.groupby("team").date.diff().dt.days.clip(upper=4).fillna(4)
    r = rows.set_index(["game_id", "team"]).rest
    h = r.reindex(pd.MultiIndex.from_arrays([games.game_id, games.home])).values
    a = r.reindex(pd.MultiIndex.from_arrays([games.game_id, games.away])).values
    return h, a


def fit_for(fits, day):
    keys = [c for c in fits if c <= day]
    return fits[max(keys)] if keys else None


def team_game_lineups(games, box, fits, proj, upcoming=None):
    """Lineup values for every team in every game. Played games use who actually played;
    upcoming games use `upcoming` (game_id, team, player_id, proj)."""
    b = box[["game_id", "team", "player_id", "date"]].merge(proj, on=["game_id", "player_id"], how="left")
    if upcoming is not None and len(upcoming):
        up = upcoming.merge(games[["game_id", "date"]], on="game_id")
        b = pd.concat([b[~b.game_id.isin(up.game_id)], up[["game_id", "team", "player_id", "date", "proj"]]], ignore_index=True)
    b["proj"] = b.proj.fillna(R.P.rookie_min).clip(lower=1.0)
    # attach each player's ratings from the latest weekly fit before the game
    cut = sorted(fits)
    b["cut"] = pd.Series(pd.cut(b.date, bins=cut + [pd.Timestamp.max], right=False, labels=cut).astype("datetime64[ns]"))
    frames = []
    for c, fr in fits.items():
        frames.append(fr[["player_id", "pm", "box"]].assign(cut=c))
    rat = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["player_id", "pm", "box", "cut"])
    b = b.merge(rat, on=["cut", "player_id"], how="left")
    b["pm"] = b.pm.fillna(0.0)
    b["box"] = b.box.fillna(R.P.box_repl * 36.0 * 0.8)
    b["share"] = 5.0 * b.proj / b.groupby(["game_id", "team"]).proj.transform("sum")
    b["v_pm"], b["v_box"] = b.share * b.pm, b.share * b.box
    t = b.groupby(["game_id", "team"]).agg(pm=("v_pm", "sum"), box=("v_box", "sum")).reset_index()
    return t, b


def build(games, box, fits, upcoming=None):
    g = games.copy()
    el = R.elo(g)
    nt = R.net_rating(g)
    g = g.merge(el, on="game_id").merge(nt, on="game_id")
    proj, last = R.projected_minutes(box)
    t, players = team_game_lineups(g, box, fits, proj, upcoming)
    # a team's usual lineup strength (EW over its previous games) to measure who's missing
    t = t.merge(g[["game_id", "date", "played"]], on="game_id").sort_values(["team", "date", "game_id"])
    for k in ("pm", "box"):
        # EW over the team's *played* games: before each played game (shifted), and through the
        # last played game for upcoming ones
        v = t[k].where(t.played)
        ew = v.groupby(t.team).transform(lambda s: s.ewm(halflife=8, min_periods=1, ignore_na=True).mean())
        prev = ew.where(t.played).groupby(t.team).shift(1)
        prev = prev.where(t.played, ew.where(t.played).groupby(t.team).ffill())
        t[f"{k}_typ"] = prev.where(t.played, ew.groupby(t.team).ffill())
    ti = t.set_index(["game_id", "team"])
    H = pd.MultiIndex.from_arrays([g.game_id, g.home])
    A = pd.MultiIndex.from_arrays([g.game_id, g.away])
    for k in ("pm", "box", "pm_typ", "box_typ"):
        g[f"{k}_h"] = ti[k].reindex(H).values
        g[f"{k}_a"] = ti[k].reindex(A).values
    g["d_elo"] = g.elo_h - g.elo_a
    g["d_net"] = g.net_h - g.net_a
    g["d_pm"] = g.pm_h - g.pm_a
    g["d_box"] = g.box_h - g.box_a
    lh = (g.pm_h - g.pm_typ_h).fillna(0) + 0.5 * (g.box_h - g.box_typ_h).fillna(0)
    la = (g.pm_a - g.pm_typ_a).fillna(0) + 0.5 * (g.box_a - g.box_typ_a).fillna(0)
    g["d_lineup_delta"] = lh - la
    rh, ra = rest_days(g)
    g["rest_h"], g["rest_a"] = rh, ra
    g["d_b2b"] = (rh <= 1).astype(float) - (ra <= 1).astype(float)
    g["rest_diff"] = np.clip(rh, 1, 4) - np.clip(ra, 1, 4)
    g["home_field"] = (~g.neutral).astype(float)
    g["target"] = g.margin
    return g, players, last
