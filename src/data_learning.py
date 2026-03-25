from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class CalibrationStats:
    volatility: float
    impulse_strength: float
    retracement_depth: float
    sweep_frequency: float
    breakout_probability: float
    wick_upper_mean: float
    wick_lower_mean: float
    avg_volume: float


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _compute_sweeps(df: pd.DataFrame, lookback: int = 30) -> float:
    highs = df["high"].rolling(lookback).max().shift(1)
    lows = df["low"].rolling(lookback).min().shift(1)
    up_sweep = (df["high"] > highs) & (df["close"] < highs)
    down_sweep = (df["low"] < lows) & (df["close"] > lows)
    sweeps = (up_sweep | down_sweep).fillna(False)
    return float(sweeps.mean())


def _compute_breakout_probability(df: pd.DataFrame, lookback: int = 20) -> float:
    prev_high = df["high"].rolling(lookback).max().shift(1)
    prev_low = df["low"].rolling(lookback).min().shift(1)
    breakout = ((df["close"] > prev_high) | (df["close"] < prev_low)).fillna(False)
    return float(breakout.mean())


def calibrate_from_dataframe(df: pd.DataFrame) -> CalibrationStats:
    returns = np.log(df["close"]).diff().dropna()
    volatility = float(returns.std(ddof=0))

    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    impulse_strength = float((body / range_).fillna(0.0).mean())

    retracement_depth = float(((df["high"] - df["close"]) / range_).fillna(0.0).mean())
    sweep_frequency = _compute_sweeps(df)
    breakout_probability = _compute_breakout_probability(df)

    wick_upper = (df["high"] - df[["open", "close"]].max(axis=1)) / range_
    wick_lower = (df[["open", "close"]].min(axis=1) - df["low"]) / range_

    return CalibrationStats(
        volatility=volatility,
        impulse_strength=impulse_strength,
        retracement_depth=retracement_depth,
        sweep_frequency=sweep_frequency,
        breakout_probability=breakout_probability,
        wick_upper_mean=float(wick_upper.fillna(0.0).mean()),
        wick_lower_mean=float(wick_lower.fillna(0.0).mean()),
        avg_volume=float(df["volume"].mean()),
    )


def calibrate_from_csv(path: Path) -> Dict[str, float]:
    df = load_ohlcv_csv(path)
    stats = calibrate_from_dataframe(df)
    return stats.__dict__
