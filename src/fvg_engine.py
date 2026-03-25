from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FVGZone:
    id: str
    direction: str
    top: float
    bottom: float
    size_abs: float
    size_relative: float
    creation_index: int
    age: int = 0
    active: bool = True
    partially_filled: bool = False
    fill_ratio: float = 0.0
    fully_filled: bool = False
    invalidated: bool = False
    originating_state: str = ""
    originating_session: str = ""


class FVGBook:
    def __init__(self):
        self.zones: List[FVGZone] = []

    def detect_new(self, df: pd.DataFrame, idx: int, state: str = "", session: str = "") -> List[FVGZone]:
        created = []
        if idx < 2:
            return created
        h2 = float(df["high"].iloc[idx - 2])
        l2 = float(df["low"].iloc[idx - 2])
        h = float(df["high"].iloc[idx])
        l = float(df["low"].iloc[idx])
        r = max(float(df["high"].iloc[idx] - df["low"].iloc[idx]), 1e-8)

        if l > h2:
            size = l - h2
            z = FVGZone(f"fvg_bull_{idx}_{len(self.zones)}", "bullish", top=l, bottom=h2, size_abs=size, size_relative=size / r, creation_index=idx, originating_state=state, originating_session=session)
            self.zones.append(z)
            created.append(z)
        if h < l2:
            size = l2 - h
            z = FVGZone(f"fvg_bear_{idx}_{len(self.zones)}", "bearish", top=l2, bottom=h, size_abs=size, size_relative=size / r, creation_index=idx, originating_state=state, originating_session=session)
            self.zones.append(z)
            created.append(z)
        return created

    def update_fill(self, idx: int, high: float, low: float) -> None:
        for z in self.zones:
            if not z.active:
                continue
            z.age = idx - z.creation_index
            overlap = max(0.0, min(high, z.top) - max(low, z.bottom))
            ratio = overlap / max(z.size_abs, 1e-8)
            if ratio > 0:
                z.partially_filled = True
                z.fill_ratio = max(z.fill_ratio, float(min(ratio, 1.0)))
            if low <= z.bottom and high >= z.top:
                z.fully_filled = True
                z.fill_ratio = 1.0
                z.active = False

    def nearest_open_distance(self, price: float) -> float:
        d = [min(abs(price - z.top), abs(price - z.bottom)) for z in self.zones if z.active]
        return float(min(d) if d else np.nan)

    def open_count_nearby(self, price: float, threshold: float) -> int:
        n = 0
        for z in self.zones:
            if z.active and min(abs(price - z.top), abs(price - z.bottom)) <= threshold:
                n += 1
        return n

    def stats(self) -> Dict[str, float]:
        if not self.zones:
            return {"fvg_frequency": 0.0, "fvg_fill_complete_rate": 0.0, "fvg_partial_rate": 0.0, "fvg_age_to_fill_mean": 0.0}
        filled = [z for z in self.zones if z.fully_filled]
        partial = [z for z in self.zones if z.partially_filled]
        return {
            "fvg_frequency": float(len(self.zones)),
            "fvg_fill_complete_rate": float(len(filled) / len(self.zones)),
            "fvg_partial_rate": float(len(partial) / len(self.zones)),
            "fvg_age_to_fill_mean": float(np.mean([z.age for z in filled]) if filled else 0.0),
        }
