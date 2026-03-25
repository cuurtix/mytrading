from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class LiquidityZone:
    id: str
    price_level: float
    upper_bound: float
    lower_bound: float
    zone_type: str
    side: str
    strength: float
    touch_count: int
    first_seen_index: int
    last_seen_index: int
    age: int = 0
    active: bool = True
    swept: bool = False
    last_swept_index: int | None = None
    metadata: Dict[str, float] = field(default_factory=dict)


class LiquidityMap:
    def __init__(self, zone_width_bps: float = 5.0):
        self.zone_width_bps = zone_width_bps
        self.zones: List[LiquidityZone] = []

    def _bounds(self, price: float) -> tuple[float, float]:
        half = price * (self.zone_width_bps / 10_000.0)
        return price - half, price + half

    def _find_matching_zone(self, price: float, side: str, zone_type: str) -> Optional[LiquidityZone]:
        for z in self.zones:
            if not z.active or z.side != side or z.zone_type != zone_type:
                continue
            if z.lower_bound <= price <= z.upper_bound:
                return z
        return None

    def add_or_touch_zone(self, idx: int, price: float, zone_type: str, side: str, strength: float, meta: Dict[str, float] | None = None) -> LiquidityZone:
        z = self._find_matching_zone(price, side, zone_type)
        if z is not None:
            z.touch_count += 1
            z.last_seen_index = idx
            z.strength = 0.7 * z.strength + 0.3 * strength
            z.metadata.update(meta or {})
            return z

        lo, hi = self._bounds(price)
        zone = LiquidityZone(
            id=f"{zone_type}_{side}_{idx}_{len(self.zones)}",
            price_level=price,
            upper_bound=hi,
            lower_bound=lo,
            zone_type=zone_type,
            side=side,
            strength=strength,
            touch_count=1,
            first_seen_index=idx,
            last_seen_index=idx,
            metadata=meta or {},
        )
        self.zones.append(zone)
        return zone

    def age_and_decay(self, idx: int, decay: float = 0.995, deactivate_age: int = 500) -> None:
        for z in self.zones:
            z.age = idx - z.first_seen_index
            z.strength *= decay
            if z.age > deactivate_age or z.strength < 0.05:
                z.active = False

    def detect_sweep(self, idx: int, high: float, low: float, close: float) -> Dict[str, float]:
        sweep_buy = 0
        sweep_sell = 0
        strength = 0.0
        for z in self.zones:
            if not z.active:
                continue
            if z.side == "buy_side" and high > z.upper_bound:
                overshoot = high - z.upper_bound
                reintegrated = close < z.upper_bound
                if overshoot > 0:
                    z.swept = True
                    z.last_swept_index = idx
                    if reintegrated:
                        sweep_buy = 1
                    strength = max(strength, overshoot / max(z.price_level, 1e-8))
            if z.side == "sell_side" and low < z.lower_bound:
                overshoot = z.lower_bound - low
                reintegrated = close > z.lower_bound
                if overshoot > 0:
                    z.swept = True
                    z.last_swept_index = idx
                    if reintegrated:
                        sweep_sell = 1
                    strength = max(strength, overshoot / max(z.price_level, 1e-8))

        return {
            "sweep_buy_side": sweep_buy,
            "sweep_sell_side": sweep_sell,
            "recent_sweep_flag": int(sweep_buy or sweep_sell),
            "recent_sweep_side": 1 if sweep_buy else -1 if sweep_sell else 0,
            "recent_sweep_strength": float(strength),
        }

    def nearest_distances(self, price: float) -> Dict[str, float]:
        buy = [abs(price - z.price_level) for z in self.zones if z.active and z.side == "buy_side"]
        sell = [abs(price - z.price_level) for z in self.zones if z.active and z.side == "sell_side"]
        strengths = [z.strength for z in self.zones if z.active]
        return {
            "distance_to_nearest_buy_liquidity": float(min(buy) if buy else np.nan),
            "distance_to_nearest_sell_liquidity": float(min(sell) if sell else np.nan),
            "nearest_liquidity_strength": float(max(strengths) if strengths else 0.0),
        }


def build_liquidity_map(df: pd.DataFrame) -> LiquidityMap:
    lm = LiquidityMap()
    rolling_high = df["high"].rolling(30, min_periods=5).max()
    rolling_low = df["low"].rolling(30, min_periods=5).min()

    for i, row in df.iterrows():
        if pd.notna(row.get("equal_high", False)) and bool(row.get("equal_high", False)):
            lm.add_or_touch_zone(i, float(row["high"]), "equal_high", "buy_side", strength=1.0)
        if pd.notna(row.get("equal_low", False)) and bool(row.get("equal_low", False)):
            lm.add_or_touch_zone(i, float(row["low"]), "equal_low", "sell_side", strength=1.0)

        if bool(row.get("swing_high", False)):
            lm.add_or_touch_zone(i, float(row["high"]), "swing_high", "buy_side", strength=1.2)
        if bool(row.get("swing_low", False)):
            lm.add_or_touch_zone(i, float(row["low"]), "swing_low", "sell_side", strength=1.2)

        if pd.notna(rolling_high.iloc[i]):
            lm.add_or_touch_zone(i, float(rolling_high.iloc[i]), "range_high", "buy_side", strength=0.7)
        if pd.notna(rolling_low.iloc[i]):
            lm.add_or_touch_zone(i, float(rolling_low.iloc[i]), "range_low", "sell_side", strength=0.7)

        lm.age_and_decay(i)

    return lm
