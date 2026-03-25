from __future__ import annotations

import pandas as pd

from src.feature_engineering import (
    detect_breakout_rejection,
    detect_equal_highs_lows,
    detect_fvg,
    detect_swings,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=8, freq="min", tz="UTC"),
            "open": [10, 11, 12, 13, 12, 11, 12, 13],
            "high": [11, 12, 15, 14, 13, 12, 13, 14],
            "low": [9, 10, 11, 12, 10, 10.5, 11, 12],
            "close": [10.5, 11.5, 14, 12.5, 11, 11.2, 12.5, 13.5],
            "volume": [100] * 8,
        }
    )


def test_swing_high_low_detection():
    df = detect_swings(_base_df(), left=1, right=1)
    assert df["swing_high"].any()
    assert df["swing_low"].any()


def test_fvg_detection_simple():
    df = _base_df().copy()
    df.loc[2, "low"] = 13
    df = detect_fvg(df)
    assert bool(df.loc[2, "fvg_bullish"])


def test_equal_high_detection():
    df = _base_df().copy()
    df.loc[6, "high"] = df.loc[1, "high"]
    out = detect_equal_highs_lows(df, tol=0.0001)
    assert bool(out.loc[6, "equal_high"])


def test_breakout_rejected_detection():
    df = _base_df().copy()
    df.loc[7, "high"] = 20
    df.loc[7, "close"] = 12
    out = detect_breakout_rejection(df, lookback=3)
    assert bool(out.loc[7, "breakout_rejected"])
