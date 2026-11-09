"""
Game context features for experiments.py: travel and time zones.

  d_travel  km the away team travelled since its previous game minus the home team's (1000s of km)
  d_tz      time zones crossed since the previous game, away minus home
  road_trip consecutive road games so far for the away team (minus home team's)
"""
from __future__ import annotations

import math

import numpy as np

# home arena per team: (lat, lon, UTC offset in winter)
SITES = {
    "ANA": (33.808, -117.877, -8), "BOS": (42.366, -71.062, -5), "BUF": (42.875, -78.876, -5),
    "CGY": (51.037, -114.052, -7), "CAR": (35.803, -78.722, -5), "CHI": (41.881, -87.674, -6),
    "COL": (39.749, -105.008, -7), "CBJ": (39.969, -83.006, -5), "DAL": (32.790, -96.810, -6),
    "DET": (42.341, -83.055, -5), "EDM": (53.547, -113.498, -7), "FLA": (26.158, -80.326, -5),
    "LAK": (34.043, -118.267, -8), "MIN": (44.945, -93.101, -6), "MTL": (45.496, -73.569, -5),
    "NSH": (36.159, -86.778, -6), "NJD": (40.734, -74.171, -5), "NYI": (40.712, -73.726, -5),
    "NYR": (40.751, -73.993, -5), "OTT": (45.297, -75.927, -5), "PHI": (39.901, -75.172, -5),
    "PIT": (40.439, -79.989, -5), "SJS": (37.333, -121.901, -8), "SEA": (47.622, -122.354, -8),
    "STL": (38.627, -90.203, -6), "TBL": (27.943, -82.452, -5), "TOR": (43.643, -79.379, -5),
    "VAN": (49.278, -123.109, -8), "VGK": (36.103, -115.178, -8), "WSH": (38.898, -77.021, -5),
    "WPG": (49.893, -97.144, -6), "UTA": (40.768, -111.901, -7), "ARI": (33.532, -112.261, -7),
}


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def add_context(df):
    """Attach travel / time-zone features to the per-game table (games in date order)."""
    df = df.sort_values("gameday").copy()
    last_site, trip = {}, {}
    trav, tz, rt = [], [], []
    for r in df.itertuples(index=False):
        site = SITES.get(r.home_team) if r.home_field else None
        vals = []
        for team, home in ((r.home_team, True), (r.away_team, False)):
            prev = last_site.get((team, r.season), SITES.get(team))
            if site is None or prev is None:
                vals.append((0.0, 0.0))
            else:
                vals.append((_km(prev, site) / 1000.0, abs(site[2] - prev[2])))
        trav.append(vals[1][0] - vals[0][0])
        tz.append(vals[1][1] - vals[0][1])
        rt.append(trip.get(r.away_team, 0) - trip.get(r.home_team, 0))
        if r.played and site is not None:
            last_site[(r.home_team, r.season)] = site
            last_site[(r.away_team, r.season)] = site
            trip[r.home_team] = 0
            trip[r.away_team] = trip.get(r.away_team, 0) + 1
    df["d_travel"] = trav
    df["d_tz"] = tz
    df["road_trip"] = np.clip(rt, -6, 6)
    return df
