from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    idx: int
    price: float
    kind: str  # high/low
    strength: float
    level: str  # minor/major


def detect_swings_hierarchical(df: pd.DataFrame, window_minor: int = 2, window_major: int = 5) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    for i in range(window_major, len(df) - window_major):
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]

        minor_high = hi == df["high"].iloc[i - window_minor : i + window_minor + 1].max()
        minor_low = lo == df["low"].iloc[i - window_minor : i + window_minor + 1].min()
        major_high = hi == df["high"].iloc[i - window_major : i + window_major + 1].max()
        major_low = lo == df["low"].iloc[i - window_major : i + window_major + 1].min()

        local_range = float((df["high"].iloc[i - window_minor : i + window_minor + 1].max() - df["low"].iloc[i - window_minor : i + window_minor + 1].min()) + 1e-8)
        if minor_high:
            swings.append(SwingPoint(i, float(hi), "high", strength=abs(float(df["close"].iloc[i] - df["open"].iloc[i])) / local_range, level="major" if major_high else "minor"))
        if minor_low:
            swings.append(SwingPoint(i, float(lo), "low", strength=abs(float(df["close"].iloc[i] - df["open"].iloc[i])) / local_range, level="major" if major_low else "minor"))
    return swings


def structure_labels_from_swings(df: pd.DataFrame, swings: List[SwingPoint]) -> pd.DataFrame:
    out = df.copy()
    out["structure_label"] = ""
    out["recent_bos_flag"] = 0
    out["recent_choch_flag"] = 0
    out["trend_context"] = "RANGE"

    highs = [s for s in swings if s.kind == "high" and s.level == "major"]
    lows = [s for s in swings if s.kind == "low" and s.level == "major"]

    for i in range(1, len(highs)):
        label = "HH" if highs[i].price > highs[i - 1].price else "LH"
        out.at[highs[i].idx, "structure_label"] = label
    for i in range(1, len(lows)):
        label = "HL" if lows[i].price > lows[i - 1].price else "LL"
        prev = out.at[lows[i].idx, "structure_label"]
        out.at[lows[i].idx, "structure_label"] = (prev + "/" + label).strip("/")

    # BOS/CHoCH basés sur rupture de derniers swings majeurs
    last_high = None
    last_low = None
    trend = "RANGE"
    for i in range(len(out)):
        high_hit = [s for s in highs if s.idx == i]
        low_hit = [s for s in lows if s.idx == i]
        if high_hit:
            last_high = high_hit[-1].price
        if low_hit:
            last_low = low_hit[-1].price

        c = out["close"].iloc[i]
        if last_high is not None and c > last_high:
            out.at[i, "recent_bos_flag"] = 1
            if trend == "DOWN":
                out.at[i, "recent_choch_flag"] = 1
            trend = "UP"
        elif last_low is not None and c < last_low:
            out.at[i, "recent_bos_flag"] = 1
            if trend == "UP":
                out.at[i, "recent_choch_flag"] = 1
            trend = "DOWN"

        out.at[i, "trend_context"] = "TREND_UP" if trend == "UP" else "TREND_DOWN" if trend == "DOWN" else "RANGE"

    return out


def add_impulse_retracement_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    impulse = out["close"].diff().rolling(lookback, min_periods=3).apply(lambda x: float(np.max(np.abs(x))), raw=False)
    retr = out["close"].rolling(lookback, min_periods=3).apply(lambda x: float(np.max(x) - np.min(x)), raw=False)
    out["recent_impulse_size"] = impulse.fillna(0.0)
    out["recent_retracement_ratio"] = (retr / (impulse.abs() + 1e-8)).fillna(0.0)
    return out
