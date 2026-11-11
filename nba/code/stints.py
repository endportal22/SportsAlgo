"""
Who was on the floor: rebuilds each team's five players for every stretch of every game from
ESPN's play-by-play, and the score while they were out there ("stints").

  - Period 1 starts with the box score's starters.
  - Later periods start with whoever appears in that period before being subbed in (anyone who
    records a play, or is subbed out, without first entering); gaps are filled from the previous
    period's closing five.
  - Each substitution ("X enters the game for Y") swaps one player.

A stint is kept only if both teams have exactly five players. Writes data/stints_<season>.csv.gz:
game_id, h1..h5, a1..a5 (ESPN player ids), secs, home_pts, away_pts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PBP_COLS = ["game_id", "sequence_number", "type_text", "home_score", "away_score", "period_number",
            "start_game_seconds_remaining", "team_id", "athlete_id_1", "athlete_id_2", "athlete_id_3",
            "home_team_id", "away_team_id"]


def _ids(x):
    return [int(v) for v in x if v == v and v is not None]


def game_stints(ev, roster, starters):
    """ev: one game's plays in order. roster: {player_id: 'H' or 'A'}. starters: {'H': [...], 'A': [...]}."""
    out = []
    if len(starters.get("H", [])) != 5 or len(starters.get("A", [])) != 5:
        return out
    seq = ev.sequence_number.values
    typ = ev.type_text.values
    per = ev.period_number.values
    t = ev.start_game_seconds_remaining.values
    hs, as_ = ev.home_score.values, ev.away_score.values
    a1, a2, a3 = ev.athlete_id_1.values, ev.athlete_id_2.values, ev.athlete_id_3.values
    on = {"H": list(starters["H"]), "A": list(starters["A"])}
    prev_end = {"H": list(on["H"]), "A": list(on["A"])}
    n = len(ev)
    cur_period = None
    stint_start_t = stint_h = stint_a = None
    last_h = last_a = 0

    def close(end_t, h_now, a_now):
        nonlocal stint_start_t, stint_h, stint_a
        if stint_start_t is not None and len(set(on["H"])) == 5 and len(set(on["A"])) == 5:
            secs = float(stint_start_t - end_t)
            if secs > 0 or h_now != stint_h or a_now != stint_a:
                out.append(sorted(on["H"]) + sorted(on["A"]) + [max(secs, 0.0), h_now - stint_h, a_now - stint_a])
        stint_start_t, stint_h, stint_a = end_t, h_now, a_now

    i = 0
    while i < n:
        p = per[i]
        if p != cur_period:
            # close the previous period's last stint at the period's end
            if cur_period is not None:
                close(period_end, last_h, last_a)
                prev_end = {"H": list(on["H"]), "A": list(on["A"])}
            cur_period = p
            j = i
            while j < n and per[j] == p:
                j += 1
            period_end = float(np.nanmin(t[i:j])) if j > i else 0.0
            if p > 1:
                # who starts this period: first appearance not as an incoming substitute
                first = {"H": [], "A": []}
                entered = set()
                for k in range(i, j):
                    if typ[k] == "Substitution":
                        inc, outp = a1[k], a2[k]
                        if outp == outp and int(outp) in roster and int(outp) not in entered:
                            s = roster[int(outp)]
                            if int(outp) not in first[s]:
                                first[s].append(int(outp))
                        if inc == inc:
                            entered.add(int(inc))
                    else:
                        for x in (a1[k], a2[k], a3[k]):
                            if x == x and int(x) in roster and int(x) not in entered:
                                s = roster[int(x)]
                                if int(x) not in first[s]:
                                    first[s].append(int(x))
                    if len(first["H"]) >= 5 and len(first["A"]) >= 5:
                        break
                for s in ("H", "A"):
                    f = first[s][:5]
                    for x in prev_end[s]:
                        if len(f) >= 5:
                            break
                        if x not in f and x not in entered:
                            f.append(x)
                    on[s] = f
            period_start = float(np.nanmax(t[i:j])) if j > i else 0.0
            stint_start_t, stint_h, stint_a = period_start, last_h, last_a
        if typ[i] == "Substitution":
            inc, outp = a1[i], a2[i]
            if inc == inc and outp == outp and int(inc) in roster:
                s = roster[int(inc)]
                close(float(t[i]), last_h, last_a)
                if int(outp) in on[s]:
                    on[s].remove(int(outp))
                elif len(on[s]) >= 5:
                    on[s] = on[s][:4]          # lost track of who left: this stint will be dropped if wrong
                if int(inc) not in on[s]:
                    on[s].append(int(inc))
        if hs[i] == hs[i]:
            last_h, last_a = int(hs[i]), int(as_[i])
        i += 1
    if cur_period is not None:
        close(period_end, last_h, last_a)
    return out


def build_season(pbp: pd.DataFrame, box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """pbp: ESPN play-by-play rows; box: box score rows (game_id, team, player_id, starter, min);
    games: game_id, home, away."""
    home_of = dict(zip(games.game_id, games.home))
    bx = box[box["min"].fillna(0) > 0]
    rosters, starts = {}, {}
    for gid, g in bx.groupby("game_id"):
        h = home_of.get(gid)
        if h is None:
            continue
        rosters[gid] = {int(p): ("H" if t == h else "A") for p, t in zip(g.player_id, g.team)}
        st = g[g.starter.astype(str).str.lower().isin(["true", "1", "1.0"])]
        starts[gid] = {"H": [int(p) for p, t in zip(st.player_id, st.team) if t == h],
                       "A": [int(p) for p, t in zip(st.player_id, st.team) if t != h]}
    rows = []
    pbp = pbp.sort_values(["game_id", "period_number", "sequence_number"])
    for gid, ev in pbp.groupby("game_id", sort=False):
        gid = int(gid)
        if gid not in rosters:
            continue
        for r in game_stints(ev, rosters[gid], starts[gid]):
            rows.append([gid] + r)
    cols = ["game_id"] + [f"h{k}" for k in range(1, 6)] + [f"a{k}" for k in range(1, 6)] + ["secs", "home_pts", "away_pts"]
    return pd.DataFrame(rows, columns=cols)


def load(data_dir, seasons=None):
    fs = sorted(Path(data_dir).glob("stints_*.csv.gz"))
    if seasons is not None:
        fs = [f for f in fs if int(f.stem.split("_")[1].split(".")[0]) in seasons]
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
