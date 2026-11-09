"""Where everything lives. The code is in NFL Algo/code; everything else is one level up."""
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
