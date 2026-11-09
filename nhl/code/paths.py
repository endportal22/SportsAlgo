"""Where everything lives. The code is in NHL Algo/code; everything else is one level up."""
import os
from datetime import date
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("NHL_ALGO_ROOT", CODE.parent))   # NHL_ALGO_ROOT is for testing
DATA = Path(os.environ.get("NHL_ALGO_DATA", ROOT / "data"))
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
CACHE = DATA / "cache"


def today() -> date:
    """Today's date (NHL_TODAY=YYYY-MM-DD overrides it, for testing)."""
    t = os.environ.get("NHL_TODAY")
    return date.fromisoformat(t) if t else date.today()


def now_utc():
    """Current UTC time (noon UTC on NHL_TODAY when that's set, for testing)."""
    from datetime import datetime, timezone
    t = os.environ.get("NHL_TODAY")
    return datetime.fromisoformat(t + "T12:00:00+00:00") if t else datetime.now(timezone.utc)


def season_label(season: int) -> str:
    """2025 -> '2025-26'."""
    return f"{season}-{str(season + 1)[-2:]}"
