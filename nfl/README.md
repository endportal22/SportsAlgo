# NFL model (v4)

Predicts every NFL game: who wins, by how much, and how that changes with a different starting QB.

## How it works

- **Data** (`update_data.py`): play-by-play, schedules, snap counts, rosters and injury reports from [nflverse](https://github.com/nflverse/nflverse-data).
- **Team ratings:** Elo, plus exponentially weighted, opponent-adjusted EPA per play for passing and rushing, on offense and defense.
- **Player ratings** (`players.py`): every player's impact from a ridge plus-minus on snap shares, refit weekly since 2016. QBs also get a production rating (EPA per dropback).
- **Model** (`model.py`): a ridge regression on the home point margin, turned into a win probability. Its inputs are:
  - Elo
  - pass and run matchups
  - QB rating, plus a new-QB flag
  - roster strength from the players actually on the team
  - injuries, compared with the team's own backups. Players listed Questionable count by how often such players actually played (2018–2025), based on their last practice of the week: 45% if they didn't practice, 68% if limited, 78% if full.
  - rest
  - home field
  - division games
  - final-week adjustments

  It never sees the betting line.

## Backtest

Walk-forward over 2019–2025 (1,954 games): log loss 0.621 and 64.8% of winners picked, against the Vegas spread's 0.608 and 66.3%.

## Run it

```
pip install -r ../requirements.txt
python code/update_data.py
python code/app.py            # desktop dashboard at http://127.0.0.1:8765
python code/export_app.py     # the phone app's data files
```
