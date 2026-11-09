#!/usr/bin/env bash
# Build one league's app data (download, predict, encrypt) into app/data/<league>.
# If any step fails, republish that league's last published files so it stays in the app.
#   usage: build_league.sh nfl|nhl|nba
set -uo pipefail
lg="$1"
out="app/data/$lg"
site="${SITE:?}"

# first run: start from the history kept on the data-seed branch
needs_seed=false
case "$lg" in
  nhl) [ -f nhl/data/games.csv ] || needs_seed=true ;;
  nba) [ -f nba/data/odds.csv ] || needs_seed=true ;;
esac
if $needs_seed; then
  if git fetch -q --depth 1 origin data-seed; then
    mkdir -p "$lg/data"
    git archive FETCH_HEAD "$lg" | tar -x --strip-components=1 -C "$lg/data"
    echo "$lg: seeded $(ls "$lg/data" | wc -l) files from data-seed"
  fi
fi

if python "$lg/code/update_data.py" && python "$lg/code/export_app.py" --out "$out"; then
  echo "$lg: updated"
  exit 0
fi

echo "::warning::$lg update failed; republishing its last published data"
rm -rf "$out"; mkdir -p "$out"
if curl -fsS "$site/data/$lg/files.json" -o "$out/files.json"; then
  for f in $(python -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))))" "$out/files.json"); do
    curl -fsS "$site/data/$lg/$f" -o "$out/$f" || echo "missing $f"
  done
else
  echo "::warning::no published $lg data to fall back on"
  rm -rf "$out"
fi
exit 0
