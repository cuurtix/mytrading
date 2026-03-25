from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))


@dataclass
class LiquidityZone:
    price: float
    weight: float
    kind: str
    decay: float = 0.999

    def step_decay(self) -> None:
        self.weight *= self.decay


@dataclass
class SessionProfile:
    vol_mult: float
    liq_mult: float
    fomo_mult: float
    sweep_mult: float


SESSION_PROFILES: Dict[str, SessionProfile] = {
    "ASIA": SessionProfile(vol_mult=0.8, liq_mult=1.2, fomo_mult=0.7, sweep_mult=0.8),
    "LONDON": SessionProfile(vol_mult=1.15, liq_mult=1.0, fomo_mult=1.1, sweep_mult=1.1),
    "NEW_YORK": SessionProfile(vol_mult=1.3, liq_mult=0.9, fomo_mult=1.25, sweep_mult=1.25),
}


def liquidity_force(price: float, zones: List[LiquidityZone], eps: float = 1e-6) -> float:
    force = 0.0
    for z in zones:
        direction = np.sign(z.price - price)
        force += direction * (z.weight / (abs(price - z.price) + eps))
    return float(force)


def sweep_probability(liquidity_density: float, volatility: float, momentum: float, k1=1.2, k2=1.0, k3=1.4) -> float:
    return sigmoid(k1 * liquidity_density + k2 * volatility + k3 * momentum)


def fvg_flags(highs: np.ndarray, lows: np.ndarray, idx: int) -> Dict[str, bool]:
    if idx < 2:
        return {"bullish": False, "bearish": False}
    bullish = lows[idx] > highs[idx - 2]
    bearish = highs[idx] < lows[idx - 2]
    return {"bullish": bool(bullish), "bearish": bool(bearish)}


def imbalance_intensity(open_p: float, close_p: float, high_p: float, low_p: float) -> float:
    body = abs(close_p - open_p)
    rng = max(high_p - low_p, 1e-8)
    return float(body / rng)


def rebalance_force(distance_to_fvg: float, lam: float = 0.04) -> float:
    return float(lam * distance_to_fvg)


def zone_reversal_probability(zone_strength: float, liquidity: float, volatility: float, a=1.3, b=0.8, c=-0.6) -> float:
    return sigmoid(a * zone_strength + b * liquidity + c * volatility)


def trend_sign(ema_fast: float, ema_slow: float) -> int:
    return int(np.sign(ema_fast - ema_slow))


def fomo_intensity(momentum: float, distance_from_range: float, breakout_signal: int, liquidity_proximity: float, a1=2.0, a2=1.0, a3=1.4, a4=1.5) -> float:
    return sigmoid(
        a1 * momentum + a2 * distance_from_range + a3 * breakout_signal + a4 * liquidity_proximity
    )


def market_impact_square_root(volatility: float, order_size: float, market_volume: float) -> float:
    denom = max(market_volume, 1e-8)
    return float(volatility * np.sqrt(max(order_size, 0.0) / denom))


def update_of_herding(prev_of: float, rho: float, noise_scale: float, rng: np.random.Generator) -> float:
    return float(rho * prev_of + rng.normal(0.0, noise_scale))
