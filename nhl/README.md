# NHL Algo (v1)

Predicts NHL games (home goal margin and win probability) from team chance quality (expected goals), Elo, special teams, the starting goalies, injuries and back-to-backs, and compares every pick to the betting line. Built the same way as NFL Algo: one shared feature builder, a walk-forward backtest, and a local HUD.

**Status:** the code has been tested end-to-end against a simulated league served in the exact format of the NHL and ESPN APIs, but it hasn't been run on real NHL data yet (that happens on your Mac on the first launch). The backtest section below fills in once you press **Run backtest**.

## What's in this folder

| Folder / file | What it is |
|---|---|
| **Start NHL Algo.command** | Double-click to open the HUD in your browser. Close the Terminal window to stop it. |
| `predictions/` | Prediction CSVs and charts (from **Export CSV**) |
| `reports/` | Backtest results |
| `data/` | NHL and ESPN data (refreshed by **Update data**) |
| `models/` | The trained model (and tuned settings, if you run the tuner) |
| `code/` | The program files. You shouldn't need to open these. |
| `.venv/` | The Python environment the app runs on (created on first launch, hidden in Finder) |

## First launch

Double-click **Start NHL Algo.command**. The first time, it:
1. sets up its own Python environment and installs pandas, numpy, scipy, scikit-learn (a minute or two);
2. downloads every game since 2018-19: results, play-by-play with shot locations, time on ice, goalies, plus betting lines, rosters and injuries. That's about 11,000 games at two downloads each, so **expect 15-30 minutes**. Progress shows in the Terminal window. If it's interrupted, double-click again and it carries on where it stopped (or press **Update data** in the HUD);
3. builds the ratings (a couple of minutes after each data update; otherwise starts are quick because the result is cached).

The HUD runs on port 8766, so it can be open at the same time as the NFL HUD (8765).

## Daily routine

1. Double-click **Start NHL Algo.command**.
2. Press **Update data**. It fetches last night's games, today's lines and the latest injury list (under a minute).
3. On the Games tab, check the starting goalies. They're projections until teams confirm them on game day (usually after the morning skate). Switch a goalie in the game panel if a different one is starting. Games marked **B2B** are the second night of a back-to-back.
4. Optional: press **Export CSV** to save the day's predictions to `predictions/`.

## The HUD

- **Games:** every game on a day with the model's and Vegas's win chances, a "why" breakdown in goals for the selected game, and results with ✓/✗ for days already played. Use ← → (or the date picker) to change days.
- **Game panel:** the starting goalies (with a switch), each team's rest, and the missing regulars with their impact.
- **Power:** team rankings in goals per game versus an average team, with each team's No. 1 goalie, 5-on-5 expected goals for/against, power play and penalty kill.
- **Players:** skater ratings (impact, offense, defense, shooting) with rising and falling players and a weekly trend chart; goalie ratings (goals saved above expected) with a game-by-game chart.
- **Backtest:** past-season scores. **Run backtest** re-scores the seasons (about a minute).

For each day you view, the HUD refits the model on games played before that day. Past days therefore show honest results, and the season record (model vs Vegas) is out-of-sample.

## Command line (optional)

From Terminal:

```bash
cd ~/Desktop/"NHL Algo"
.venv/bin/python code/app.py            # the HUD (same as the launcher)
.venv/bin/python code/update_data.py    # refresh data
.venv/bin/python code/train.py          # refit the model -> models/
.venv/bin/python code/predict.py        # next game day -> predictions/
.venv/bin/python code/backtest.py       # re-score past seasons -> reports/
.venv/bin/python code/experiments.py    # test features that aren't in the model
.venv/bin/python code/experiments.py --tune --save   # tune the team-rating settings and keep the best
```

`predict.py --date 2026-10-20` gives a specific day, `--days 7` the next week, `--all` the rest of the season. `update_data.py --start 2015` downloads more history.

## How it works

### Data (`update_data.py`)

- **NHL public API** (api-web.nhle.com, no key): schedule and results, box scores (every player's time on ice; which goalie started), play-by-play (every shot attempt with its location, shot type, game state and the goalie in net), current rosters, and DraftKings lines for today's games.
- **ESPN public API** (no key): pre-game moneylines for past and upcoming games (lines go back to about 2021-22; older games have none), and the current injury list.
- Finished games are downloaded once; each update only fetches new games.

### Expected goals (`xg.py`)

xG is hockey's version of EPA: the chance an unblocked shot becomes a goal, from distance, angle, shot type, rebounds, rushes and power-play state. Rating teams and goalies on chances rather than goals settles down much faster, because goals are rare and lumpy.

### Team ratings (`features.py`)

Games are processed in date order and each team carries:
- **5-on-5 xG for and against per 60 minutes**, opponent-adjusted and exponentially weighted (half-life about 14 games). The matchup edge is the home offense against the away defense, minus the reverse, in goals per game;
- **special teams**: power-play plus penalty-kill xG per game, which covers both how often a team draws/takes penalties and what it does with them;
- **margin-of-victory Elo** (FiveThirtyEight's NHL version);
- **rest**: days since the last game, and a back-to-back flag.

Each prediction uses a snapshot taken right before puck drop. Between seasons the ratings are pulled back toward league average. Arizona's history carries over to Utah.

### Players (`players.py`)

- **Skater impact:** regularised plus-minus on time-on-ice shares. Each team-game's 5-on-5 xG per 60 is modelled as the TOI-weighted sum of its skaters' offense ratings plus the opposing skaters' defense ratings, solved with ridge regression over the last three seasons (recent games weighted more), refit every week. Values are xG per 60 on the ice compared with an average skater. Linemates who always play together split credit, so read trends, not decimals.
- **Goalies:** goals saved above expected per unblocked shot, recency-weighted and shrunk heavily toward a slightly-below-average prior (about 2,000 shots of evidence count as much as the prior), shown per game. This is the hockey version of the NFL algo's QB rating, and the starting goalie is the biggest player factor.
- **Lineups and injuries:** each team's expected lineup comes from its last 5 games. For played games, the players who actually dressed are used (lineups are known at warm-ups). For upcoming games, players no longer on the NHL roster (traded, waived, sent to the AHL) are removed, and ESPN's injury list removes Out / IR / suspended players and counts day-to-day players at 50%. Injury cost is each missing regular's impact versus the team's own depth players, in goals per game.
- **Projected goalie:** the team's No. 1 (most starts in its last 10 games) unless he's out, or the backup on the second night of a back-to-back. Switch in the HUD once starters are confirmed.

### Model (`model.py`)

Ridge regression on the home team's goal margin (empty-net goals and shootouts left out, since both are mostly noise about who was trailing late), then a logistic curve fit on past wins and losses turns the predicted margin into a win probability (overtime and shootouts included). Features: Elo, 5-on-5 xG edge, special teams, starting goalies, injuries, back-to-back, rest and home ice. The first downloaded season is a burn-in for the ratings and is never trained or scored on.

## Backtest results

Each season is predicted using only earlier seasons for training and only pre-game data for features. The HUD's Backtest tab shows the current numbers after you press **Run backtest**. They compare:

| model | what it uses |
|---|---|
| always pick home | the historical home-win rate |
| Elo only | Elo + home ice |
| team ratings only | Elo, 5-on-5 xG, special teams, rest |
| **model (current)** | team ratings + starting goalies + injuries |
| Vegas line | the vig-free moneyline (only on games that have one) |

What to expect: NHL games are much closer to coin flips than NFL games. Good public models land around 0.66-0.67 log loss and 58-61% accuracy, and the betting market is very hard to beat. Treat the edge column as "worth a second look", not as a betting signal.

## Tested and not in the model

`code/experiments.py` re-runs these against the current model: travel distance, time zones crossed, road-trip length, recent goal differential, finishing / save luck (goals minus xG), full-lineup value and playoff games. Run it once you have data to see which (if any) help; differences under about 0.001 log loss are noise. `--tune --save` also tries other decay and regression settings for the team ratings and saves the best to `models/params.json`, which every script and the HUD then use.

## Differences from NFL Algo

- Days instead of weeks: the Games tab steps through game days, and the model is refit for each day.
- Goalie switch instead of QB switch.
- Win probability comes from a fitted logistic curve rather than a normal curve, because hockey margins are small and lumpy.
- Vegas is the moneyline (vig removed) rather than a point spread.

## Ideas for next

- Shift charts (who was on the ice for each shot) for true shift-level player ratings instead of game-level TOI shares.
- A confirmed-starters feed, so the goalie switch isn't needed on game day.
- Totals (over/under), where xG rates and goaltending matter even more.
- Score-adjusted xG (trailing teams generate more shots late in games).
