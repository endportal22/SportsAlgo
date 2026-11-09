"""
The v2 model: predict the home team's point margin with ridge regression, then
turn that into a win probability using the spread of past prediction errors.

Why margin instead of win/loss? A 30-point win and a 1-point win carry very
different information, and with only ~270 games a season the model needs every
bit of signal it can get. It also gives you a predicted spread for free.
"""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# v4 feature set (see README "Backtest"): team strength, pass/run matchup, QB, injuries,
# roster-based strength (who's actually on the team) and a final-week adjustment
STAT_FEATS = ["d_elo", "d_matchup", "d_pass_net", "d_run_net", "d_qb_rating", "d_qb_new", "d_injury_rel",
              "rest_diff", "home_field", "div_game",
              "d_roster", "fw_x_elo", "fw_x_matchup", "fw_x_qb_rating"]
# v3 feature set, kept for comparison in backtest.py
V3_FEATS = ["d_elo", "d_matchup", "d_pass_net", "d_run_net", "d_qb_rating", "d_qb_new", "d_injury_rel",
            "rest_diff", "home_field", "div_game"]
# v2 feature set, kept for comparison in backtest.py
V2_FEATS = ["d_net_epa", "d_off_epa", "d_net_sr", "d_elo", "rest_diff", "home_field", "div_game", "d_qb_new"]
MARKET_FEATS = ["spread_line"]


class MarginModel:
    def __init__(self, feats, alpha=10.0):
        self.feats = list(feats)
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.sigma = 13.5

    def fit(self, df):
        d = df.dropna(subset=["margin"])
        X = self._X(d)
        self.pipe.fit(X, d["margin"])
        resid = d["margin"] - self.pipe.predict(X)
        self.sigma = float(np.std(resid))
        return self

    def _X(self, df):
        # missing player info (e.g. no listed QB yet) -> neutral
        return df[self.feats].astype(float).fillna(0.0)

    def predict_margin(self, df):
        return self.pipe.predict(self._X(df))

    def predict_proba(self, df):
        return norm.cdf(self.predict_margin(df) / self.sigma)

    def coefficients(self):
        sc, rg = self.pipe.named_steps["standardscaler"], self.pipe.named_steps["ridge"]
        # points of margin per 1 standard deviation of each feature
        return dict(zip(self.feats, np.round(rg.coef_, 3))), float(rg.intercept_)
