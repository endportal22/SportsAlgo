# Model analysis (November 2026)

This analysis asked two questions: is there better math for the models, and is the scoring against Vegas fair? Every number below comes from walk-forward backtests. Each game was predicted by a model trained only on games before it.

The test periods:

- **NBA:** 2018-19 to 2026-27 (10,377 games)
- **NHL:** 2020-21 to 2026-27 (8,227 games)
- **NFL:** 2019 to 2026 (2,095 games; 2,088 without ties)

Vegas comparisons use only games with a closing line.

## 1. How the models are scored

A won-lost record is a weak way to compare forecasters. In the NBA, the model and Vegas picked exactly the same share of winners (67.2%), yet Vegas's percentages are measurably better. The app's Record tab now shows four kinds of evidence for every league.

| Measure | What it judges |
|---|---|
| **Log loss** | The percentages themselves; confident misses are punished hardest. Lower is better; a coin flip scores 0.693. |
| **Brier score** | The mean squared error of the percentages. Lower is better; a coin flip scores 0.25. |
| **Winners picked** | Only who was favoured. |
| **Calibration** | When a team was given 60–65%, how often it actually won. |

The Record tab also shows whether the log-loss gap to Vegas is bigger than chance (a 95% range from a bootstrap over weeks or months), and how each side did when they disagreed.

### Results

| | NBA | NHL | NFL |
|---|---|---|---|
| Games with a line | 7,460 | 7,224 | 2,088 |
| Model log loss | 0.6088 | 0.6583 | 0.6214 |
| Vegas log loss | 0.6056 | 0.6580 | 0.6093 |
| Gap (95% range) | +0.0031 (+0.0006 to +0.0057) | +0.0003 (−0.0020 to +0.0027) | +0.0121 (+0.0056 to +0.0182) |
| Verdict | Vegas slightly ahead | Even | Vegas clearly ahead |
| Model Brier / Vegas Brier | 0.2104 / 0.2092 | 0.2332 / 0.2330 | 0.2164 / 0.2109 |
| Winners picked, model / Vegas | 67.2% / 67.1% | 60.1% / 60.8% | 64.7% / 66.5% |
| Does the model know something Vegas doesn't? (encompassing test, z) | Yes (z = 3.6) | Yes (z = 3.4) | No (z = 1.1) |
| Picks that differ from Vegas: model right | 51.1% of 687 | 46.2% of 777 | 43.6% of 266 |
| 10+ point gaps: side Vegas leaned toward won | 54.1% of 715 | 43.4% of 196 | 56.8% of 387 |

The encompassing test is a logistic regression of results on both Vegas's and the model's log-odds. A significant model coefficient means the model carries information that isn't already in the line. That's true for the NBA and NHL models, but not the NFL one.

### Calibration

All three models are well calibrated:

- The calibration error is 0.007–0.011, against 0.008–0.021 for Vegas.
- A second-stage calibration trained on the model's own track record in earlier seasons didn't help: Platt scaling, isotonic regression and a slope-only correction all scored the same or worse. The fitted slopes are 0.93–1.04, where 1.00 means already calibrated.
- The NFL and NBA models rate home teams 1–2 points of probability higher than recent results support, because home advantage has shrunk since 2020. Trend features and recency weighting didn't improve log loss, so this is something to watch rather than fix.

### Turning odds into Vegas's percentages

Bookmaker odds include a margin, which has to be removed. Four methods were compared: proportional, additive, power and Shin.

- The differences are tiny (±0.001 log loss), so the comparison is fair whichever method is used.
- Two changes made the benchmark as strong as it really is:
  - **NFL:** the app now uses the moneylines instead of converting the spread with a normal curve (log loss 0.6098 vs 0.6105–0.6121).
  - **NBA:** the app now uses the main book's moneyline instead of the median across books (0.6033 vs 0.6039).

## 2. Alternative math

Each alternative replaced or extended the production model, with everything else unchanged. Each row shows the change in log loss against the production model. Negative is better. Changes smaller than about 0.001 are within chance.

| Idea | NBA | NHL | NFL |
|---|---|---|---|
| Fit the probability curve on out-of-fold margins | 0.0000 | 0.0000 | 0.0000 |
| Logistic vs normal curve for margin → win % | +0.0012 (normal worse) | −0.0003 (normal) | +0.0011 (logistic worse) |
| Confidence that grows with games played this season | −0.0002 | +0.0002 | +0.0021 |
| Train on an "expected" margin, 25% weight (NBA: margin before garbage time; NHL: expected goals) | 0.0000 | 0.0000 | −0.0008* |
| Train on the expected margin only | −0.0001 | +0.0017 | +0.0020 |
| Predict win/loss directly (logistic regression) | +0.0008 | 0.0000 | +0.0039 |
| Average of production and win/loss model | +0.0002 | −0.0001 | +0.0011 |
| Kalman-filter team ratings (dynamic state-space model), added | −0.0001 | +0.0001 | +0.0011 |
| Kalman ratings instead of Elo | −0.0001 | +0.0005 | +0.0028 |
| Poisson goals per team with a Skellam win probability (NHL) | — | +0.0010 | — |
| Average of production and Poisson (NHL) | — | +0.0001 | — |
| Gradient-boosting blend, 30% | +0.0003 | +0.0007 | +0.0014 |
| Ridge strength 2 / 30 / 100 | ≤ 0.0001 | ≤ 0.0001 | ≤ 0.0002 |
| Blowouts capped | 0.0000 | 0.0000 | 0.0000 |
| Head-to-head: raw margin, or residual vs the model's expectation (shrunk) | 0.0000 | +0.0001 | +0.0001 |

\* NFL "expected points" from summed EPA nearly equals the real margin (correlation 0.996), so this isn't a real alternative target.

**Head-to-head in detail:** a team's over- or under-performance against the model in recent meetings has no relationship with its result in the next meeting:

- NBA: correlation −0.005 (n = 10,377)
- NHL: −0.002 (n = 8,194)
- NFL: +0.007 (n = 1,533)

Head-to-head is shown in the app for reading only.

**NBA only, also no gain:** short or long form windows, season carry-over (0.2–0.55), travel and time zones, 3 games in 4 nights, altitude, and playoff home court.

**Conclusion:** the models sit at the ceiling of what public game data supports. The NBA's switch to play-by-play player ratings (v2) was the last change that measurably helped: log loss 0.6038 → 0.6028.

## 3. Where the remaining gap comes from

The gap is information, not math:

- **NBA:** the bigger the disagreement with Vegas, the more often Vegas is right. That points to late injury, rest and minutes news the model doesn't have.
- **NFL:** the model trails by a similar amount across seasons and situations; the NFL betting market is very efficient.
- **NHL:** even with Vegas. When the two differ by 10+ points, the model's side has won more often.

## 4. Changes made

- **Live picks:** the last prediction before every game is saved, and the Record tab scores those (since Nov 10, 2026).
- **More update runs:** about 6:15pm and 8:30pm Eastern, plus Sunday 2pm, catch late scratches and confirmed goalies.
- **NFL Questionable players:** they count by how often players with their last practice status actually played: 45% didn't practise, 68% limited, 78% full.
- **NBA and NHL injury-status log:** every injury-status change is saved with the time it was seen.
  - Once a status has 150 logged NBA player-games, the model uses how often players with that status actually play, instead of treating them as fully in.
- **Record tab scorecard:** a long-run scorecard with log loss, Brier score, calibration, a significance check, and the disagreement record.
- **Fairer Vegas percentages:** NFL from moneylines, NBA from the main book.

## 5. Possible next steps

- **NBA:** check the injury-status calibration once it switches on (about 1–2 weeks of logging).
- **NBA:** an official NBA injury-report history (PDFs since 2021-22) could calibrate statuses from day one.
- **MLB:** a model before opening day.
