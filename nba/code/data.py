"""Load the downloaded NBA files into tidy tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load_games(data: Path) -> pd.DataFrame:
    fs = sorted(Path(data).glob("schedule_*.csv"))
    if not fs:
        raise FileNotFoundError("No NBA data yet. Run update_data.py first.")
    g = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("game_id", keep="last")
    g["date"] = pd.to_datetime(g.date)
    g["completed"] = _bool(g.completed)
    g["neutral"] = _bool(g.neutral)
    g["played"] = g.completed & g.home_score.notna() & g.away_score.notna()
    g["margin"] = np.where(g.played, g.home_score - g.away_score, np.nan)
    g["home_win"] = np.where(g.played, (g.margin > 0).astype(float), np.nan)
    # 2020 "bubble" games in Orlando had no home court
    g.loc[(g.date >= "2020-07-30") & (g.date <= "2020-10-15"), "neutral"] = True
    g = g[g.home.isin(TEAMS) & g.away.isin(TEAMS)]   # drops All-Star games and unset NBA Cup knockouts
    return g.sort_values(["date", "start_utc", "game_id"]).reset_index(drop=True)


def load_box(data: Path, games: pd.DataFrame) -> pd.DataFrame:
    fs = sorted(Path(data).glob("box_*.csv.gz"))
    b = pd.concat([pd.read_csv(f, dtype={"pm": str}) for f in fs], ignore_index=True)
    b["pm"] = pd.to_numeric(b.pm.astype(str).str.replace("+", "", regex=False), errors="coerce")
    b = b.drop_duplicates(["game_id", "player_id"], keep="last")
    b = b.merge(games[["game_id", "date", "season", "home", "away"]], on="game_id", how="inner")
    b["min"] = pd.to_numeric(b["min"], errors="coerce").fillna(0.0)
    b = b[b["min"] > 0].copy()
    num = ["pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm"]
    b[num] = b[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    b["player_id"] = b.player_id.astype(int)
    b["is_home"] = b.team == b.home
    # Hollinger game score: a quick all-in-one box score number
    b["gmsc"] = (b.pts + 0.4 * b.fgm - 0.7 * b.fga - 0.4 * (b.fta - b.ftm) + 0.7 * b.oreb + 0.3 * b.dreb
                 + b.stl + 0.7 * b.ast + 0.7 * b.blk - 0.4 * b.pf - b.tov)
    return b.sort_values(["date", "game_id", "team", "min"], ascending=[True, True, True, False]).reset_index(drop=True)


ODDS_COLS = ["game_id", "market_home_prob", "spread", "home_ml", "away_ml", "details", "book", "total"]


def load_odds(data: Path) -> pd.DataFrame:
    f = Path(data) / "odds.csv"
    o = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["game_id"])
    for c in ODDS_COLS:
        if c not in o:
            o[c] = np.nan
    o["game_id"] = o.game_id.astype("int64")
    return o[ODDS_COLS].drop_duplicates("game_id", keep="last")
