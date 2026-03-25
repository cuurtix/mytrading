from __future__ import annotations

import pandas as pd

DEFAULT_XAUUSD_TZ = "UTC"


def xauusd_session_name(ts: pd.Timestamp, tz: str = DEFAULT_XAUUSD_TZ) -> str:
    h = ts.tz_convert(tz).hour
    # Convention explicite UTC pour XAUUSD
    if 0 <= h < 8:
        return "ASIA"
    if 8 <= h < 13:
        return "LONDON_OPEN"
    if 13 <= h < 22:
        return "NEW_YORK"
    return "LATE_SESSION"
