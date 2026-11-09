#!/bin/bash
# Double-click to launch the NFL Algo HUD. Close this window to stop it.
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
if ! "$PY" -c "import pandas, numpy, scipy, sklearn, joblib" 2>/dev/null; then
  echo "Installing missing packages..."
  "$PY" -m pip install pandas numpy scipy scikit-learn joblib
fi
if [ ! -f data/games.csv ]; then echo "First run: downloading data (about a minute)..."; "$PY" code/update_data.py; fi
"$PY" code/app.py
