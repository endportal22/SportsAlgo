"""Where everything lives. The code is in nba/code; everything else is one level up."""
import os
from datetime import date, datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("NBA_ALGO_ROOT", CODE.parent))
DATA = Path(os.environ.get("NBA_ALGO_DATA", ROOT / "data"))
CACHE = DATA / "cache"


def today() -> date:
    """Today's date (NBA_TODAY=YYYY-MM-DD overrides it, for testing)."""
    t = os.environ.get("NBA_TODAY")
    return date.fromisoformat(t) if t else date.today()


def now_utc():
    from datetime import timezone
    t = os.environ.get("NBA_TODAY")
    return datetime.fromisoformat(t + "T12:00:00+00:00") if t else datetime.now(timezone.utc)


def season_label(season: int) -> str:
    """2026 -> '2026-27'."""
    return f"{season}-{str(season + 1)[-2:]}"


def current_season() -> int:
    t = today()
    return t.year if t.month >= 8 else t.year - 1
