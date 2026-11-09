# Sports Algo

Private prediction app for a small group of friends. Switch between leagues at the top of the app:

- **NFL:** weekly game win probabilities with starting-QB switching, power rankings, ratings for every player with weekly trends, and the season record against Vegas.
- **NHL:** daily game win probabilities with starting-goalie switching, power rankings, skater and goalie ratings, and the record against the moneyline.
- **NBA, MLB:** coming soon.

## For friends: install the app

1. Open the link you were sent: **Safari** on iPhone, **Chrome** on Android.
2. Enter the password.
3. Install it:
   - **iPhone:** tap Share, then **Add to Home Screen**.
   - **Android:** tap ⋮, then **Install app**.

It then opens in its own window like any other app, and keeps the last predictions it loaded when you're offline.

## How it updates

`.github/workflows/update.yml` runs every morning (~6am ET), every afternoon (~5pm ET, for NHL lines and goalies), and on Sunday at about 11:30am ET (after the NFL inactives). Each run:

1. downloads the latest NFL data from [nflverse](https://github.com/nflverse/nflverse-data) and NHL data from the NHL and ESPN APIs;
2. rebuilds each league's predictions (`nfl/code/export_app.py`, `nhl/code/export_app.py`);
3. encrypts them with the `APP_PASSWORD` secret;
4. publishes `app/` to GitHub Pages.

If one league fails to update, the app keeps that league's last published predictions. To update right away, go to **Actions → Update predictions → Run workflow**.

To change the password, update the `APP_PASSWORD` secret (Settings → Secrets and variables → Actions), then run the workflow. Friends will be asked for the new password.

## What's where

| Path | What it is |
|---|---|
| `app/` | The phone app (HTML/JS). `model.js` re-runs the model on the phone when you switch a QB or goalie. |
| `nfl/code/` | The NFL model: data download, features, player ratings, training, backtest, desktop HUD (`app.py`). |
| `nhl/code/` | The NHL model: data download, expected goals, features, skater and goalie ratings, desktop HUD. |
| `*/code/export_app.py` | Packages a league's output as (encrypted) JSON for the app, in a format shared by every league. |

The NHL history (2018 on) is kept on the `nhl-data-seed` branch and used to start the first hosted build; after that, each run only downloads new games.

The program code in this repository is public. The predictions and data the app shows are encrypted with the password.

## Models

Both are ridge regressions on the game margin, converted to a win probability, that never see the betting line.

- **NFL:** Elo, pass and run efficiency matchups, QB rating, roster strength from per-player ratings, injuries, rest, home field, final-week adjustment. Over 2019–2025 (walk-forward): log loss 0.621 and 64.8% of winners, against Vegas at 0.608 and 66.3%.
- **NHL:** Elo, even-strength and special-teams expected-goal ratings, starting-goalie rating, injuries relative to depth, back-to-backs, rest, home ice. Over 2020–2025 (walk-forward): log loss 0.6595, against the closing moneyline at 0.6593.

Not betting advice.

To run one locally: `pip install -r requirements.txt`, then e.g. `python nhl/code/update_data.py` and `python nhl/code/app.py` for the desktop HUD.
