# NBA model

Predicts every NBA game: who wins, by how much, and how that changes when you mark players in or out.

## How it works

- **Data** (`update_data.py`): every game and player box score since 2015-16 from [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) (ESPN's data as daily CSV files). The last few days, upcoming games, betting lines, the injury report and current rosters come straight from ESPN.
- **Team ratings** (`ratings.py`): Elo, plus an exponentially weighted, opponent-adjusted point margin ("recent form").
- **On the floor** (`stints.py`): ESPN's play-by-play is replayed substitution by substitution to rebuild which ten players were on the floor for every stretch of every game (about 99% of game time is covered).
- **Player ratings**, refit every Monday:
  - **play-by-play plus-minus:** a ridge regression of every stretch's scoring margin on the ten players on the floor, weighted by minutes and pulled slightly toward the box-score rating (`ratings.fit_rapm`);
  - **game-level plus-minus:** each game's margin on who played and for how many minutes (used to measure who's missing);
  - **box score:** game score per 36 minutes, shrunk toward bench level for players with few minutes.
- **Lineups:** for each game, every available player's projected minutes (recent average, scaled to 240 per team) times his ratings. For upcoming games, "available" means the current roster minus players listed Out or Doubtful. In the app you can mark anyone in or out.
- **Model** (`model.py`): a ridge regression on the home margin with these inputs, turned into a win probability with a logistic curve:
  - Elo
  - recent form
  - lineup play-by-play plus-minus and lineup box score
  - who's in or out compared with the team's usual lineup
  - back-to-backs
  - rest
  - home court

  It never sees the betting line.

## Backtest

Walk-forward, 2018-19 through this season (10,377 games). Each month is predicted by a model trained only on earlier games.

| | log loss | winners |
|---|---|---|
| Elo only | 0.625 | 64.6% |
| Team ratings only | 0.621 | 65.5% |
| v1: game-level plus-minus | 0.604 | 67.4% |
| v2 (current): play-by-play plus-minus | 0.603 | 67.7% |

On the 7,460 games with a closing line (2017-18 to 2022-23, 2025-26 and this season), the model's log loss is 0.609 (v1: 0.610) and Vegas's is 0.606.

Tested with no gain (Nov 2026): head-to-head record, last-5 form, travel/time zones, 3 games in 4 nights, altitude, playoff home court, weighting recent seasons more.

The backtest knows which players actually played. Real predictions only know the injury report, so expect them to be slightly less accurate than the backtest.

## Run it

```
pip install -r ../requirements.txt
python code/update_data.py     # ~1 min the first time
python code/export_app.py      # writes the app's data files
```
