"""
Matchup details for the app's game screen, the same for every league:

  records  each team's record going into the game: overall, at home / on the road, last 10, streak
  stats    a side-by-side comparison of season stats, with league ranks
  h2h      the last meetings between the two teams and this season's series
  form     each team's last five results

Everything is computed from games *before* the game's date, so past days show what was known then.
These numbers are for reading only; the models don't use them (head-to-head records were tested
and add nothing once team ratings are in the model).

Each league passes:
  games       game_id, date, season, home, away, home_score, away_score, played,
              regular (bool: counts toward the standings), extra ("OT", "SO" or None)
  team_games  one row per team per played game: game_id, team, and the raw columns the stats use
  specs       the stats to show: {"key", "label", "num", "den", "scale", "fmt", "better"}
              value = scale * sum(num) / sum(den) over the team's games ("gp" counts games);
              better is "high", "low" or None (no winner highlighted, e.g. pace)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _short(d):
    t = pd.Timestamp(d)
    return f"{t.strftime('%b')} {t.day}"


def _long(d):
    t = pd.Timestamp(d)
    return f"{t.strftime('%b')} {t.day}, {t.year}"


def _ordinal(n):
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class Matchups:
    def __init__(self, games, team_games, specs, season_label, ties=False, otl=False, min_games=3, h2h_n=10,
                 recent=5):
        g = games.copy()
        g["date"] = pd.to_datetime(g["date"])
        self.g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
        self.played = self.g[self.g.played].copy()
        self.played["winner"] = np.where(self.played.home_score > self.played.away_score, self.played.home,
                                         np.where(self.played.home_score < self.played.away_score, self.played.away, None))
        tg = team_games.copy()
        tg["gp"] = 1.0
        self.tg = tg.merge(self.played[["game_id", "date", "season", "regular"]], on="game_id", how="inner")
        self.specs = specs
        self.season_label = season_label     # function: season -> "2026-27"
        self.ties, self.otl = ties, otl
        self.min_games, self.h2h_n, self.recent = min_games, h2h_n, recent
        self._tables = {}

    # ------------------------------------------------------------------ stats
    def table(self, season, before):
        """Every team's stats in `season` from games before `before`, with ranks."""
        key = (season, pd.Timestamp(before))
        if key in self._tables:
            return self._tables[key]
        d = self.tg[(self.tg.season == season) & (self.tg.date < key[1]) & self.tg.regular]
        cols = sorted({c for s in self.specs for c in (s["num"], s["den"])})
        sums = d.groupby("team")[cols].sum() if len(d) else pd.DataFrame(columns=cols)
        out = pd.DataFrame(index=sums.index)
        out["gp"] = sums["gp"] if "gp" in sums else 0
        for s in self.specs:
            den = sums[s["den"]].replace(0, np.nan)
            out[s["key"]] = s.get("scale", 1.0) * sums[s["num"]] / den
            asc = s.get("better") == "low"
            out[s["key"] + "_rk"] = out[s["key"]].rank(ascending=asc, method="min")
        self._tables[key] = out
        return out

    def stats(self, home, away, season, date):
        t = self.table(season, date)
        basis = f"{self.season_label(season)} season"
        thin = any(t.gp.get(x, 0) < self.min_games for x in (home, away))
        if thin:   # too early in the season: use last season's full numbers
            prev = self.table(season - 1, pd.Timestamp.max)
            if len(prev) and home in prev.index and away in prev.index:
                t, basis = prev, f"{self.season_label(season - 1)} season (too early for this season's)"
        if home not in t.index or away not in t.index:
            return None
        rows = []
        n = len(t)
        for s in self.specs:
            hv, av = t.at[home, s["key"]], t.at[away, s["key"]]
            if pd.isna(hv) or pd.isna(av):
                continue
            fmt = s.get("fmt", "{:.1f}")
            better = None
            if s.get("better") in ("high", "low") and fmt.format(hv) != fmt.format(av):
                better = "home" if (hv > av) == (s["better"] == "high") else "away"
            rows.append({"label": s["label"], "home": fmt.format(hv), "away": fmt.format(av),
                         "home_rank": _ordinal(t.at[home, s["key"] + "_rk"]), "away_rank": _ordinal(t.at[away, s["key"] + "_rk"]),
                         "better": better})
        return {"basis": basis, "teams": n, "rows": rows}

    # ------------------------------------------------------------------ records
    def _result(self, r, team):
        if r.winner is None:
            return "T"
        if r.winner == team:
            return "W"
        return "OTL" if (self.otl and r.extra in ("OT", "SO")) else "L"

    def _wl(self, rs):
        w, l = rs.count("W"), rs.count("L")
        if self.otl:
            return f"{w}-{l}-{rs.count('OTL')}"
        t = rs.count("T")
        return f"{w}-{l}" + (f"-{t}" if t else "")

    def records(self, team, season, date):
        p = self.played
        d = p[(p.season == season) & (p.date < pd.Timestamp(date)) & p.regular & ((p.home == team) | (p.away == team))]
        res = [self._result(r, team) for r in d.itertuples(index=False)]
        home = [self._result(r, team) for r in d[d.home == team].itertuples(index=False)]
        away = [self._result(r, team) for r in d[d.away == team].itertuples(index=False)]
        streak = ""
        if res:
            last = res[-1]
            k = 0
            for x in reversed(res):
                if x != last:
                    break
                k += 1
            streak = f"{'L' if last == 'OTL' else last}{k}" if last != "OTL" else f"OTL{k}"
        n10 = res[-10:]
        return {"overall": self._wl(res) if res else "0-0", "home": self._wl(home) if home else "0-0",
                "away": self._wl(away) if away else "0-0", "last10": self._wl(n10) if len(res) >= 10 else None,
                "streak": streak, "gp": len(res)}

    # ------------------------------------------------------------------ head to head
    def h2h(self, home, away, season, date):
        p = self.played
        m = p[(p.date < pd.Timestamp(date)) & (((p.home == home) & (p.away == away)) | ((p.home == away) & (p.away == home)))]
        last = m.tail(self.h2h_n)
        games = [{"date": _long(r.date), "home": r.home, "away": r.away, "home_score": int(r.home_score),
                  "away_score": int(r.away_score), "winner": r.winner, "note": r.extra if isinstance(r.extra, str) else None,
                  "playoff": not bool(r.regular)} for r in last.itertuples(index=False)][::-1]
        if not games:
            return {"summary": "First meeting since " + self.season_label(int(p.season.min())), "season": None, "games": []}
        wh = sum(1 for x in games if x["winner"] == home)
        wa = sum(1 for x in games if x["winner"] == away)
        n = len(games)
        lead = f"{home} {wh}-{wa}" if wh >= wa else f"{away} {wa}-{wh}"
        summary = f"{lead} in the last {n} meeting{'s' if n > 1 else ''}" if wh != wa else f"Split {wh}-{wa} in the last {n} meetings"
        cur = m[m.season == season]
        series = None
        if len(cur):
            ch = int((cur.winner == home).sum())
            ca = int((cur.winner == away).sum())
            series = (f"{home} leads the season series {ch}-{ca}" if ch > ca else
                      f"{away} leads the season series {ca}-{ch}" if ca > ch else f"Season series tied {ch}-{ca}")
        # average margin from the home team's point of view, last meetings
        mg = [(x["home_score"] - x["away_score"]) * (1 if x["home"] == home else -1) for x in games]
        return {"summary": summary, "season": series, "avg_margin": round(float(np.mean(mg)), 1), "games": games}

    # ------------------------------------------------------------------ recent form
    def form(self, team, date):
        p = self.played
        d = p[(p.date < pd.Timestamp(date)) & ((p.home == team) | (p.away == team))].tail(self.recent)
        out = []
        for r in d.itertuples(index=False):
            at_home = r.home == team
            us, them = (r.home_score, r.away_score) if at_home else (r.away_score, r.home_score)
            out.append({"r": self._result(r, team), "score": f"{int(us)}-{int(them)}", "opp": r.away if at_home else r.home,
                        "home": bool(at_home), "date": _short(r.date),
                        "note": r.extra if isinstance(r.extra, str) else None})
        return out[::-1]

    def for_game(self, home, away, season, date):
        return {"records": {"home": self.records(home, season, date), "away": self.records(away, season, date)},
                "stats": self.stats(home, away, season, date),
                "h2h": self.h2h(home, away, season, date),
                "form": {"home": self.form(home, date), "away": self.form(away, date)}}
