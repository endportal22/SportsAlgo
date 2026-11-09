#!/bin/bash
# Double-click to launch the NHL Algo HUD. Close this window to stop it.
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "First run: setting up a Python environment for NHL Algo (a minute or two)..."
  python3 -m venv .venv
fi
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
if ! "$PY" -c "import pandas, numpy, scipy, sklearn, joblib, certifi" 2>/dev/null; then
  echo "Installing packages..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install pandas numpy scipy scikit-learn joblib certifi matplotlib
fi
if [ ! -f data/games.csv ]; then
  echo ""
  echo "First run: downloading every NHL game since 2018-19 (results, play-by-play, time on ice,"
  echo "goalies, betting lines, injuries). This takes roughly 15-30 minutes. Progress shows below."
  echo "If it gets interrupted, double-click the launcher again and it carries on where it stopped."
  echo ""
  "$PY" code/update_data.py
fi
"$PY" code/app.py
