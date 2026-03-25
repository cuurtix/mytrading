from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_structure import add_impulse_retracement_features, detect_swings_hierarchical, structure_labels_from_swings
from src.sessions import xauusd_session_name


def detect_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    out = df.copy()
    out["swing_high"] = False
    out["swing_low"] = False
    for i in range(left, len(out) - right):
        window_h = out["high"].iloc[i - left : i + right + 1]
        window_l = out["low"].iloc[i - left : i + right + 1]
        out.at[i, "swing_high"] = out["high"].iloc[i] == window_h.max()
        out.at[i, "swing_low"] = out["low"].iloc[i] == window_l.min()
    return out


def detect_equal_highs_lows(df: pd.DataFrame, tol: float = 0.0008, lookback: int = 60) -> pd.DataFrame:
    out = df.copy()
    out["equal_high"] = False
    out["equal_low"] = False
    for i in range(1, len(out)):
        h = out["high"].iloc[i]
        l = out["low"].iloc[i]
        start = max(0, i - lookback)
        prev_highs = out["high"].iloc[start:i]
        prev_lows = out["low"].iloc[start:i]
        if len(prev_highs):
            out.at[i, "equal_high"] = ((prev_highs - h).abs() / max(h, 1e-8) < tol).any()
            out.at[i, "equal_low"] = ((prev_lows - l).abs() / max(l, 1e-8) < tol).any()
    return out


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fvg_bullish"] = False
    out["fvg_bearish"] = False
    out["fvg_size"] = 0.0
    for i in range(2, len(out)):
        if out["low"].iloc[i] > out["high"].iloc[i - 2]:
            out.at[i, "fvg_bullish"] = True
            out.at[i, "fvg_size"] = out["low"].iloc[i] - out["high"].iloc[i - 2]
        if out["high"].iloc[i] < out["low"].iloc[i - 2]:
            out.at[i, "fvg_bearish"] = True
            out.at[i, "fvg_size"] = out["low"].iloc[i - 2] - out["high"].iloc[i]
    return out


def detect_breakout_rejection(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["breakout_up"] = False
    out["breakout_down"] = False
    out["breakout_rejected"] = False
    ph = out["high"].rolling(lookback).max().shift(1)
    pl = out["low"].rolling(lookback).min().shift(1)

    out["breakout_up"] = out["close"] > ph
    out["breakout_down"] = out["close"] < pl
    up_rej = (out["high"] > ph) & (out["close"] < ph)
    dn_rej = (out["low"] < pl) & (out["close"] > pl)
    out["breakout_rejected"] = (up_rej | dn_rej).fillna(False)
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"]).diff().fillna(0.0)
    out["range"] = (out["high"] - out["low"]).clip(lower=1e-8)
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_ratio"] = out["body"] / out["range"]
    out["wick_upper"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["range"]
    out["wick_lower"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["range"]
    out["realized_vol"] = out["log_return"].rolling(30, min_periods=5).std().fillna(out["log_return"].std())
    out["compression_score"] = 1.0 / (1e-8 + out["range"].rolling(20, min_periods=5).mean())
    out["expansion_score"] = out["range"] / (1e-8 + out["range"].rolling(20, min_periods=5).mean())
    out["current_range_position"] = (out["close"] - out["low"].rolling(30, min_periods=5).min()) / (
        1e-8 + out["high"].rolling(30, min_periods=5).max() - out["low"].rolling(30, min_periods=5).min()
    )

    out = detect_swings(out)
    out = detect_equal_highs_lows(out)
    out = detect_fvg(out)
    out = detect_breakout_rejection(out)

    swings = detect_swings_hierarchical(out)
    out = structure_labels_from_swings(out, swings)
    out = add_impulse_retracement_features(out)

    out["distance_to_recent_swing_high"] = out["high"].rolling(30, min_periods=5).max() - out["close"]
    out["distance_to_recent_swing_low"] = out["close"] - out["low"].rolling(30, min_periods=5).min()
    out["session_name"] = out["datetime"].apply(xauusd_session_name)

    q1 = out["realized_vol"].quantile(0.33)
    q2 = out["realized_vol"].quantile(0.66)
    out["vol_regime_bucket"] = np.where(out["realized_vol"] < q1, "LOW", np.where(out["realized_vol"] < q2, "NORMAL", "HIGH"))
    return out
