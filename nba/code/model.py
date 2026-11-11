"""
The model: ridge regression on the home team's point margin, turned into a win probability
with a logistic curve fit on past wins and losses. It never sees the betting line.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# v2 (Nov 10, 2026): play-by-play plus-minus (d_rapm, d_rapm_delta) replaces the game-level one (d_pm)
STAT_FEATS = ["d_elo", "d_net", "d_rapm", "d_box", "d_rapm_delta", "d_lineup_delta", "d_b2b", "rest_diff", "home_field"]
V1_FEATS = ["d_elo", "d_net", "d_pm", "d_box", "d_lineup_delta", "d_b2b", "rest_diff", "home_field"]
TEAM_FEATS = ["d_elo", "d_net", "d_b2b", "rest_diff", "home_field"]
ELO_FEATS = ["d_elo", "home_field"]


class MarginModel:
    def __init__(self, feats=STAT_FEATS, alpha=10.0):
        self.feats = list(feats)
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.a, self.b, self.sigma = 0.0, 0.15, 12.0

    def _X(self, df):
        return df[self.feats].astype(float).fillna(0.0)

    def fit(self, df):
        d = df.dropna(subset=["target", "home_win"])
        X = self._X(d)
        self.pipe.fit(X, d.target)
        m = self.pipe.predict(X)
        self.sigma = float(np.std(d.target - m))
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(m.reshape(-1, 1), d.home_win.astype(int))
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def predict_margin(self, df):
        return self.pipe.predict(self._X(df))

    def prob_from_margin(self, m):
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * np.asarray(m, dtype=float))))

    def predict_proba(self, df):
        return self.prob_from_margin(self.predict_margin(df))
