"""
Saves each game's last prediction made before it starts, so the app can show how the model
actually did live, not just how it does when past games are re-scored afterwards.

Every build calls record() for each upcoming game; a game's row is overwritten until it starts,
so the saved prediction is the last one published before the game. The log is kept in
<league>/data/live_log.csv (kept between hosted builds by the Actions cache), and also published,
encrypted, as live_log.json: if the cached copy is ever lost, the next build gets it back from the
live site (SITE + APP_PASSWORD environment variables).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLS = ["game_id", "start_utc", "saved_utc", "prob", "margin", "vegas_prob", "pick"]


def _utc(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _fetch_published(league):
    """The last published log, decrypted (None if unavailable)."""
    try:
        import appdata
        js = appdata.fetch_published(league, "live_log.json")
    except ImportError:
        js = None
    if not js:
        if os.environ.get("SITE"):
            print("[live] no published log to recover")
        return None
    rows = js["games"]
    print(f"[live] recovered {len(rows)} saved predictions from the published app")
    return pd.DataFrame(rows, columns=COLS)


class LiveLog:
    def __init__(self, data_dir, league, now=None):
        self.path = Path(data_dir) / "live_log.csv"
        self.now = _utc(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype={"game_id": str})
        else:
            self.df = _fetch_published(league)
            if self.df is None:
                self.df = pd.DataFrame(columns=COLS)
        self.df["game_id"] = self.df.game_id.astype(str)
        self.rows = {r["game_id"]: r for r in self.df.to_dict("records")}
        self.results, self.first = [], None

    def record(self, game_id, start_utc, prob, margin, vegas_prob, pick):
        """Save (or update) a prediction for a game that hasn't started yet."""
        start = _utc(start_utc)
        if start is None or self.now >= start:
            return
        self.rows[str(game_id)] = {"game_id": str(game_id), "start_utc": start.isoformat(),
                                   "saved_utc": self.now.isoformat(timespec="minutes"), "prob": round(float(prob), 4),
                                   "margin": round(float(margin), 3),
                                   "vegas_prob": None if vegas_prob is None or pd.isna(vegas_prob) else round(float(vegas_prob), 4),
                                   "pick": pick}

    def get(self, game_id):
        return self.rows.get(str(game_id))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(self.rows.values()), columns=COLS).to_csv(self.path, index=False)

    def payload(self):
        return {"games": [{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
                          for r in self.rows.values()]}


def apply(game, saved):
    """Show a played game with the prediction that was saved before it started."""
    if not saved or not game.get("played"):
        return game
    p = float(saved["prob"])
    hw = game.get("winner") == game["home"]
    game.update({"prob": p, "margin": float(saved["margin"]), "pick": game["home"] if p >= 0.5 else game["away"],
                 "model_correct": bool((p >= 0.5) == hw), "live": True, "saved_utc": saved.get("saved_utc")})
    return game


def summary(results):
    """results: list of (prob, home_won, vegas_prob) for played games with a saved prediction."""
    import math
    if not results:
        return None
    n = len(results)
    right = sum(1 for p, y, _ in results if (p >= 0.5) == y)
    ll = -sum(math.log(max(1e-4, p if y else 1 - p)) for p, y, _ in results) / n
    v = [(q, y) for _, y, q in results if q is not None and not (isinstance(q, float) and math.isnan(q))]
    out = {"n": n, "model_right": right, "model_ll": round(ll, 4), "vegas_n": len(v),
           "vegas_right": sum(1 for q, y in v if (q >= 0.5) == y)}
    out["vegas_ll"] = round(-sum(math.log(max(1e-4, q if y else 1 - q)) for q, y in v) / len(v), 4) if v else None
    return out


def track(log, game):
    """Call on every game written to the app: saves upcoming predictions, and shows played games with
    the prediction saved before them (collecting those for the live record)."""
    if log is None:
        return game
    if game.get("played"):
        saved = log.get(game["id"])
        if saved:
            apply(game, saved)
            vp = game.get("vegas_prob")
            log.results.append((float(game["prob"]), game.get("winner") == game["home"], vp))
            log.first = min(log.first or saved["saved_utc"], saved["saved_utc"])
    else:
        log.record(game["id"], game.get("time"), game["prob"], game["margin"], game.get("vegas_prob"), game["pick"])
    return game


def live_record(log):
    s = summary(log.results) if log is not None else None
    if s:
        t = pd.Timestamp(log.first)
        s["since"] = f"{t.strftime('%b')} {t.day}"
    return s
