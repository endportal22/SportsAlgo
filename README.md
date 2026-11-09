# Sports Algo

Private prediction app for a small group of friends. It currently covers the NFL: weekly game win probabilities with starting-QB switching, power rankings, ratings for every player with weekly trends, and the season record against Vegas.

## For friends: install the app

1. Open the link you were sent: **Safari** on iPhone, **Chrome** on Android.
2. Enter the password.
3. Install it:
   - **iPhone:** tap Share, then **Add to Home Screen**.
   - **Android:** tap ⋮, then **Install app**.

It then opens in its own window like any other app, and keeps the last predictions it loaded when you're offline.

## How it updates

`.github/workflows/update.yml` runs every morning, and again on Sunday at about 11:30am ET (after the inactives for the 1pm games are announced). Each run:

1. downloads the latest data from [nflverse](https://github.com/nflverse/nflverse-data);
2. rebuilds the predictions (`code/export_app.py`);
3. encrypts them with the `APP_PASSWORD` secret;
4. publishes `app/` to GitHub Pages.

To update right away, go to **Actions → Update predictions → Run workflow**.

To change the password, update the `APP_PASSWORD` secret (Settings → Secrets and variables → Actions), then run the workflow. Friends will be asked for the new password.

## What's where

| Path | What it is |
|---|---|
| `app/` | The phone app (HTML/JS). `model.js` re-runs the model on the phone when you switch a QB. |
| `code/` | The Python model: data download, features, player ratings, training, backtest, desktop HUD (`app.py`). |
| `code/export_app.py` | Packages the model's output as (encrypted) JSON for the app. |

The program code in this repository is public. The predictions and data the app shows are encrypted with the password.

## Model

A ridge regression on point margin, converted to a win probability. The inputs are:

- team strength (Elo)
- pass and run efficiency matchups
- QB rating
- roster-based strength from per-player ratings
- injuries
- rest
- home field
- a final-week adjustment

It never sees the Vegas line. Over 2019–2025 (walk-forward): log loss 0.621 and 64.8% of winners, against Vegas at 0.608 and 66.3%. Not betting advice.

To run it locally: `pip install -r requirements.txt`, then `python code/update_data.py` and `python code/app.py` for the desktop HUD.
