from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.market_models import (
    LiquidityZone,
    SESSION_PROFILES,
    fomo_intensity,
    imbalance_intensity,
    liquidity_force,
    market_impact_square_root,
    rebalance_force,
    sigmoid,
    sweep_probability,
    update_of_herding,
)


@dataclass
class SimParams:
    alpha: float = 0.004
    beta: float = 0.15
    gamma: float = 0.8
    delta: float = 0.5
    of_noise: float = 0.8
    rho: float = 0.35
    seed: int = 7


class SyntheticXAUUSDMarket:
    def __init__(self, calibration: Dict[str, float], timezone: str = "UTC", params: SimParams | None = None):
        self.calibration = calibration
        self.timezone = timezone
        self.params = params or SimParams()
        self.rng = np.random.default_rng(self.params.seed)
        self.zones: List[LiquidityZone] = []

    def _session_name(self, ts: pd.Timestamp) -> str:
        hour = ts.tz_convert(self.timezone).hour
        if 0 <= hour < 8:
            return "ASIA"
        if 8 <= hour < 16:
            return "LONDON"
        return "NEW_YORK"

    def _regime(self, vol: float) -> str:
        base = max(self.calibration["volatility"], 1e-8)
        x = vol / base
        if x < 0.8:
            return "LOW"
        if x < 1.2:
            return "NORMAL"
        if x < 1.8:
            return "HIGH"
        return "PANIC"

    def _update_zones(self, price: float, high: float, low: float) -> None:
        self.zones.append(LiquidityZone(price=high, weight=1.0, kind="equal_high"))
        self.zones.append(LiquidityZone(price=low, weight=1.0, kind="equal_low"))
        self.zones.append(LiquidityZone(price=price * 1.0015, weight=0.8, kind="stop_cluster"))
        self.zones.append(LiquidityZone(price=price * 0.9985, weight=0.8, kind="stop_cluster"))
        for z in self.zones:
            z.step_decay()
        self.zones = [z for z in self.zones if z.weight > 0.05][-80:]

    def run(self, historical_df: pd.DataFrame, n_steps: int = 500) -> pd.DataFrame:
        if historical_df.empty:
            raise ValueError("Historique vide")

        df = historical_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        base_price = float(df["close"].iloc[-1])

        returns_hist = np.log(df["close"]).diff().dropna()
        vol0 = float(returns_hist.std(ddof=0)) if len(returns_hist) else self.calibration["volatility"]

        ema_fast = base_price
        ema_slow = base_price
        prev_of = 0.0

        rows = []
        ts = df["datetime"].iloc[-1]

        last_high = float(df["high"].tail(30).max())
        last_low = float(df["low"].tail(30).min())

        for _ in range(n_steps):
            ts = ts + pd.Timedelta(minutes=1)
            session = self._session_name(ts)
            prof = SESSION_PROFILES[session]

            market_vol = max(vol0 * prof.vol_mult, 1e-5)
            of_base = update_of_herding(prev_of, self.params.rho, self.params.of_noise, self.rng)

            self._update_zones(base_price, last_high, last_low)
            liq_force = liquidity_force(base_price, self.zones) / 1000.0
            liq_density = float(np.mean([z.weight for z in self.zones])) if self.zones else 0.1

            momentum = abs(of_base)
            breakout = int(base_price > last_high or base_price < last_low)
            distance_range = abs(base_price - (last_high + last_low) / 2.0) / max(last_high - last_low, 1e-6)
            nearest_liq_dist = min((abs(base_price - z.price) for z in self.zones), default=1.0)
            liq_prox = 1.0 / max(nearest_liq_dist, 1e-6)

            fomo = fomo_intensity(momentum, distance_range, breakout, liq_prox) * prof.fomo_mult
            herd_factor = np.sign(of_base) * min(abs(prev_of), 3.0)
            of_fomo = fomo * herd_factor
            of_total = of_base + of_fomo

            order_size = abs(of_total) * self.calibration["avg_volume"] * 0.015
            impact = market_impact_square_root(market_vol, order_size, self.calibration["avg_volume"] * prof.liq_mult)

            p_sweep = sweep_probability(liq_density, market_vol, momentum) * prof.sweep_mult
            sweep_event = self.rng.random() < min(max(p_sweep, 0.0), 1.0)

            eps = float(self.rng.normal(0.0, market_vol))
            delta_p = (
                self.params.alpha * of_total
                + self.params.beta * liq_force
                + self.params.gamma * impact
                + self.params.delta * fomo
                + eps
            )

            if sweep_event:
                direction = 1 if base_price < (last_high + last_low) / 2 else -1
                overshoot = direction * abs(self.rng.normal(0.0, market_vol * 6))
                continuation = self.rng.random() < 0.55
                delta_p += overshoot if continuation else -overshoot

            next_price = max(0.01, base_price + delta_p)
            candle_range = abs(self.rng.normal(0.0, market_vol * 10)) + market_vol * 2
            high = max(next_price, base_price) + candle_range * self.rng.uniform(0.2, 0.9)
            low = min(next_price, base_price) - candle_range * self.rng.uniform(0.2, 0.9)

            imb = imbalance_intensity(base_price, next_price, high, low)
            fvg_bias = rebalance_force((next_price - base_price))

            ema_fast = 0.2 * next_price + 0.8 * ema_fast
            ema_slow = 0.05 * next_price + 0.95 * ema_slow
            trend = np.sign(ema_fast - ema_slow)
            regime = self._regime(market_vol)
            state = (
                "EXPANSION" if abs(delta_p) > 2 * market_vol else
                "CONSOLIDATION" if abs(delta_p) < 0.5 * market_vol else
                "TREND_UP" if trend > 0 else
                "TREND_DOWN" if trend < 0 else
                "RANGE"
            )

            volume = max(1.0, self.calibration["avg_volume"] * (1 + abs(of_total) * 0.1) * prof.liq_mult)

            rows.append(
                {
                    "datetime": ts,
                    "open": base_price,
                    "high": high,
                    "low": low,
                    "close": next_price,
                    "volume": volume,
                    "ofi": of_total,
                    "fomo": fomo,
                    "impact": impact,
                    "liquidity_force": liq_force,
                    "sweep": int(sweep_event),
                    "imbalance": imb,
                    "fvg_rebalance_force": fvg_bias,
                    "regime": regime,
                    "state": state,
                    "session": session,
                }
            )

            prev_of = of_total
            base_price = next_price
            last_high = max(last_high * 0.999, high)
            last_low = min(last_low * 1.001, low)

        return pd.DataFrame(rows)
