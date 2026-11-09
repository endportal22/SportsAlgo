"""
Game context features: special teams, travel / time zones, and weather.

  d_st        special-teams edge: recency-weighted net EPA per game on kickoffs, punts,
              field goals and extra points (yours minus opponents'), regressed each off-season
  d_travel    extra distance the away team travelled vs the home team (1000s of km)
  d_tz        time zones crossed by the away team minus the home team
  d_early     body-clock kickoff before 11am for one side (e.g. a West-coast team at 1pm ET,
              or anyone in a 9:30am ET international game)
  wind_x_pass wind (mph/10, outdoor games only) x passing-game edge: wind blunts passing
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# stadium_id -> (lat, lon, hours offset from US Eastern during the season)
STADIUMS = {
    "ATL00": (33.755, -84.401, 0), "ATL97": (33.755, -84.401, 0), "BAL00": (39.278, -76.623, 0),
    "BOS00": (42.091, -71.264, 0), "BUF00": (42.774, -78.787, 0), "CAR00": (35.226, -80.853, 0),
    "CHI98": (41.862, -87.617, -1), "CIN00": (39.095, -84.516, 0), "CLE00": (41.506, -81.700, 0),
    "DAL00": (32.748, -97.093, -1), "DEN00": (39.744, -105.020, -2), "DET00": (42.340, -83.046, 0),
    "GNB00": (44.501, -88.062, -1), "HOU00": (29.685, -95.411, -1), "IND00": (39.760, -86.164, 0),
    "JAX00": (30.324, -81.637, 0), "KAN00": (39.049, -94.484, -1), "LAX01": (33.953, -118.339, -3),
    "LAX97": (33.864, -118.261, -3), "LAX99": (34.014, -118.288, -3), "MIA00": (25.958, -80.239, 0),
    "MIN01": (44.974, -93.258, -1), "NAS00": (36.166, -86.771, -1), "NOR00": (29.951, -90.081, -1),
    "NYC01": (40.813, -74.074, 0), "OAK00": (37.751, -122.201, -3), "PHI00": (39.901, -75.168, 0),
    "PHO00": (33.528, -112.263, -3), "PIT00": (40.447, -80.016, 0), "SDG00": (32.783, -117.120, -3),
    "SEA00": (47.595, -122.332, -3), "SFO01": (37.403, -121.970, -3), "TAM00": (27.976, -82.503, 0),
    "VEG00": (36.091, -115.184, -3), "WAS00": (38.908, -76.864, 0),
    # international / neutral
    "LON00": (51.556, -0.280, 5), "LON01": (51.456, -0.342, 5), "LON02": (51.604, -0.066, 5),
    "FRA00": (50.069, 8.645, 6), "GER00": (48.219, 11.625, 6), "MUN01": (48.219, 11.625, 6),
    "MAD01": (40.453, -3.688, 6), "PAR00": (48.924, 2.360, 6), "MEX00": (19.303, -99.150, -1),
    "SAO00": (-23.545, -46.474, 1), "RIO00": (-22.912, -43.230, 1), "MEL00": (-37.820, 144.983, 14),
}
ST_PLAYS = ("kickoff", "punt", "field_goal", "extra_point")


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def special_teams_games(pbp):
    """Net special-teams EPA per (game_id, team)."""
    st = pbp[pbp.play_type.isin(ST_PLAYS) & pbp.epa.notna()].copy()
    for c in ("posteam", "defteam"):
        st[c] = st[c].replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
    a = st.groupby(["game_id", "posteam"]).epa.sum()
    b = st.groupby(["game_id", "defteam"]).epa.sum()
    a.index.names = b.index.names = ["game_id", "team"]
    return a.sub(b, fill_value=0).to_dict()


def add_context(df, games, pbp, decay=0.94, carry=4.0, regress=0.5):
    """Attach context features to the per-game feature table (same game_id keys)."""
    g = games.set_index("game_id")
    # each team's home stadium per season (non-neutral home games)
    home = games[games.location != "Neutral"].groupby(["season", "home_team"]).stadium_id.agg(lambda x: x.mode()[0])
    home_any = games[games.location != "Neutral"].groupby("home_team").stadium_id.agg(lambda x: x.mode()[0])

    def home_site(season, team):
        sid = home.get((season, team), home_any.get(team))
        return STADIUMS.get(sid)

    # ---- special teams: walk games in order, EW mean of net ST EPA per game
    st_games = special_teams_games(pbp)
    state = {}
    d_st = []
    for r in df.itertuples(index=False):
        vals = []
        for t in (r.home_team, r.away_team):
            v, w, s = state.get(t, (0.0, 0.0, r.season))
            if s != r.season:
                v, w = v * regress, min(w, carry)
            state[t] = (v, w, r.season)
            vals.append(v)
        d_st.append(vals[0] - vals[1])
        if r.played:
            for t in (r.home_team, r.away_team):
                x = st_games.get((r.game_id, t))
                if x is None:
                    continue
                v, w, s = state[t]
                w2 = w * decay + 1
                state[t] = (v + (x - v) / w2, w2, s)
    df = df.copy()
    df["d_st"] = d_st

    # ---- travel, time zones, body clock
    trav, tz, early, wind, wxp = [], [], [], [], []
    for r in df.itertuples(index=False):
        gr = g.loc[r.game_id]
        site = STADIUMS.get(gr.stadium_id) or home_site(r.season, r.home_team)
        hs, as_ = home_site(r.season, r.home_team), home_site(r.season, r.away_team)
        if site is None or hs is None or as_ is None:
            trav.append(0.0); tz.append(0.0); early.append(0.0)
        else:
            trav.append((_km(as_, site) - _km(hs, site)) / 1000.0)
            tz.append(abs(site[2] - as_[2]) - abs(site[2] - hs[2]))
            try:
                hh, mm = map(int, str(gr.gametime).split(":"))
                et = hh + mm / 60
            except ValueError:
                et = 13.0
            clock = lambda team_tz: et + team_tz  # kickoff on the team's home clock
            early.append(float(clock(as_[2]) < 11) - float(clock(hs[2]) < 11))
        roof = str(gr.roof)
        w = gr.wind if roof in ("outdoors", "open") else 0.0
        if roof in ("outdoors", "open") and pd.isna(w):
            w = np.nan
        wind.append(w)
    df["d_travel"] = trav
    df["d_tz"] = tz
    df["d_early"] = early
    df["wind"] = wind
    # unknown outdoor wind -> typical outdoor value
    med = np.nanmedian(df["wind"].values) if np.isfinite(np.nanmedian(df["wind"].values)) else 8.0
    df["wind_known"] = df["wind"].notna().astype(float)
    df["wind"] = df["wind"].fillna(med)
    df["wind_x_pass"] = df["wind"] / 10.0 * df["d_pass_net"]
    return df
