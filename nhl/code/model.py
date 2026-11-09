"""
The model: predict the home team's goal margin with ridge regression, then turn that into
a win probability with a logistic curve fit on past results.

Why margin instead of win/loss? A 5-1 win and a shootout win carry different information,
and it gives you a predicted goal line for free. The target leaves out empty-net goals and
shootouts (both mostly noise about who was trailing late). The win probability is then
calibrated directly on wins and losses (overtime and shootouts included), because hockey
margins are small and lumpy and a normal curve doesn't fit them well.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# current feature set (see README "Backtest")
STAT_FEATS = ["d_elo", "d_ev", "d_st", "d_goalie", "d_injury_rel", "d_b2b", "rest_diff", "home_field"]
# team ratings only (no goalies or lineups), kept for comparison in backtest.py
TEAM_FEATS = ["d_elo", "d_ev", "d_st", "d_b2b", "rest_diff", "home_field"]
ELO_FEATS = ["d_elo", "home_field"]
MARKET_FEATS = ["market_logit"]


class MarginModel:
    def __init__(self, feats, alpha=10.0):
        self.feats = list(feats)
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.a, self.b = 0.0, 0.9   # win prob = sigmoid(a + b * margin)
        self.sigma = 2.4

    def fit(self, df):
        d = df.dropna(subset=["target", "home_win"])
        X = self._X(d)
        self.pipe.fit(X, d["target"])
        m = self.pipe.predict(X)
        self.sigma = float(np.std(d["target"] - m))
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(m.reshape(-1, 1), d["home_win"].astype(int))
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def _X(self, df):
        # missing player info (e.g. no starting goalie known yet) -> neutral
        return df[self.feats].astype(float).fillna(0.0)

    def predict_margin(self, df):
        return self.pipe.predict(self._X(df))

    def prob_from_margin(self, m):
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * np.asarray(m, dtype=float))))

    def predict_proba(self, df):
        return self.prob_from_margin(self.predict_margin(df))

    def coefficients(self):
        rg = self.pipe.named_steps["ridge"]
        # goals of margin per 1 standard deviation of each feature
        return dict(zip(self.feats, [round(float(c), 4) for c in rg.coef_])), float(rg.intercept_)
