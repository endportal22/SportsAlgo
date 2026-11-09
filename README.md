# Sports Algo

Game predictions for a small group of friends, as an app you install on your phone. Pick a league at the top:

| League | What you get |
|---|---|
| **NFL** | Weekly win probabilities. Switch a team's starting QB and the prediction updates. |
| **NHL** | Daily win probabilities. Switch the starting goalie. |
| **NBA** | Daily win probabilities. Mark any player in or out. |
| **MLB** | Coming for opening day. |

Each league also has power rankings, ratings for every player with their trends, and the season record against Vegas. None of the models ever see the betting line; Vegas is shown only for comparison. Not betting advice.

## Install the app (for friends)

1. Open the link you were sent: **Safari** on iPhone, **Chrome** on Android.
2. Enter the password.
3. Install it:
   - **iPhone:** tap Share, then **Add to Home Screen**.
   - **Android:** tap ⋮, then **Install app**.

It opens in its own window and keeps the last predictions it loaded when you're offline.

## How it updates

`.github/workflows/update.yml` runs:

- every morning (about 6am Eastern);
- every afternoon (about 5pm Eastern, for NHL goalies, the NBA injury report and the day's lines);
- Sunday late morning (after the NFL inactives).

Each run does this for every league (`.github/scripts/build_league.sh`):

1. downloads the newest data;
2. rebuilds the predictions;
3. encrypts them with the `APP_PASSWORD` secret;
4. publishes `app/` to GitHub Pages.

If one league's update fails, the app keeps that league's last published predictions.

- **Update now:** **Actions → Update predictions → Run workflow**.
- **Change the password:** update the `APP_PASSWORD` secret (Settings → Secrets and variables → Actions), then run the workflow. Everyone is asked for the new password.

## What's where

| Path | What it is |
|---|---|
| `app/` | The phone app: one HTML page, `model.js` (re-runs a model on the phone when you change a starter or lineup), service worker, icons. |
| `nfl/`, `nhl/`, `nba/` | One folder per league: `code/` downloads its data, rates teams and players, trains, backtests and exports the app's data. Each has its own README. |
| `shared/appdata.py` | Writes the app's (encrypted) data files, in one format for every league. |
| `.github/` | The scheduled build. |
| `data-seed` branch | Past NHL games and NBA betting lines, used once to start the hosted build. |

The code is public. The predictions and data inside the app are encrypted with the password.

## How good is it?

Walk-forward backtests: every game is predicted by a model trained only on earlier games. Lower log loss is better; a coin flip scores 0.693.

| League | Seasons | Model log loss | Vegas log loss |
|---|---|---|---|
| NFL | 2019–2025 | 0.621 | 0.608 |
| NHL | 2020–2025 | 0.660 | 0.659 |
| NBA | 2017–2026 (games with a line) | 0.610 | 0.606 |

## Run a league locally

```
pip install -r requirements.txt
python nba/code/update_data.py
python nba/code/export_app.py
```

The NFL and NHL folders also have a desktop dashboard: `python nfl/code/app.py` or `python nhl/code/app.py`.
