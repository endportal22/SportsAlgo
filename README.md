# Data seed

History used once to start the hosted build of each league (later runs keep their data in the
Actions cache and only download new games):

- `nhl/`: NHL schedules, shots, skater and goalie games, lines and rosters, 2018-19 through Nov 8, 2026
- `nba/odds.csv`: NBA closing lines since 2017-18, from ESPN's game pages
- `nba/stints_<season>.csv.gz`: who was on the floor together, 2015-16 to 2025-26, rebuilt from the play-by-play (saves downloading ~3 GB of play-by-play on the first build)

NBA box scores download in about a minute, so they aren't stored here.
