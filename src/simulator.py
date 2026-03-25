from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior_learning import LearnedBehavior
from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.state_engine import sample_next_state


@dataclass
class SimulationConfig:
    seed: int = 11
    n_steps: int = 500


class SyntheticMarketSimulator:
    def __init__(self, learned: LearnedBehavior, timeframe_seconds: int):
        self.learned = learned
        self.timeframe_seconds = timeframe_seconds

    def _direction_and_amplitude(self, state: str, ctx: pd.Series, rng: np.random.Generator) -> tuple[int, float]:
        rstats = self.learned.conditional_returns.get(state, {"mu": 0.0, "sigma": self.learned.volatility_stats["log_return_sigma"], "cont_3": 0.5})
        sampled = rng.normal(rstats["mu"], rstats["sigma"])

        direction = 1 if sampled >= 0 else -1
        amplitude = abs(sampled)

        # contexte sweep / breakout / FVG
        if ctx.get("recent_sweep_flag", 0) == 1:
            if ctx.get("recent_sweep_side", 0) == 1:
                continuation = self.learned.sweep_stats.get("continuation_after_sweep_buy", 0.5)
            else:
                continuation = self.learned.sweep_stats.get("continuation_after_sweep_sell", 0.5)
            if rng.random() > continuation:
                direction *= -1
            amplitude *= 1.0 + min(1.0, float(ctx.get("recent_sweep_strength", 0.0)) * 10)

        if state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN"}:
            fomo_amp = min(1.8, 1.0 + rstats.get("cont_3", 0.5))
            amplitude *= fomo_amp
        if state in {"BREAKOUT_REJECTED", "POST_SWEEP_REVERSAL"}:
            direction *= -1
            amplitude *= 0.8

        if ctx.get("distance_to_nearest_open_fvg", 9999.0) < ctx.get("range", 1.0):
            amplitude *= 0.9

        return direction, max(1e-7, amplitude)

    def run(self, history_df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
        if history_df.empty:
            raise ValueError("historique vide")

        rng = np.random.default_rng(cfg.seed)
        last = history_df.iloc[-1]
        current_price = float(last["close"])
        ts = pd.to_datetime(last["datetime"], utc=True)
        current_state = str(self.learned.feature_df["state"].iloc[-1]) if "state" in self.learned.feature_df.columns else "RANGE"

        lmap = LiquidityMap()
        fvg_book = FVGBook()

        out = []
        for i in range(cfg.n_steps):
            if out:
                ref = pd.Series(out[-1])
            else:
                ref = self.learned.feature_df.iloc[-1].copy()

            next_state = sample_next_state(current_state, ref, self.learned.transition_model.transition_probs, rng)
            direction, amplitude = self._direction_and_amplitude(next_state, ref, rng)

            vol_bucket = str(ref.get("vol_regime_bucket", "NORMAL"))
            session = str(ref.get("session_name", "LONDON"))
            range_stats = self.learned.transition_model.state_range_stats.get(
                (next_state, vol_bucket, session),
                {"mu": self.learned.volatility_stats["range_mean"], "sigma": self.learned.volatility_stats["range_mean"] * 0.3},
            )
            wstats = self.learned.conditional_wicks.get(next_state, {"upper_mu": 0.25, "lower_mu": 0.25, "upper_sigma": 0.07, "lower_sigma": 0.07})

            log_ret = direction * amplitude
            next_close = max(0.01, current_price * np.exp(log_ret))
            candle_range = max(1e-8, rng.normal(range_stats["mu"], range_stats["sigma"]))
            wick_up = max(0.0, rng.normal(wstats["upper_mu"], wstats["upper_sigma"]))
            wick_dn = max(0.0, rng.normal(wstats["lower_mu"], wstats["lower_sigma"]))

            high = max(current_price, next_close) + candle_range * wick_up
            low = min(current_price, next_close) - candle_range * wick_dn

            # update contextual objects
            lmap.add_or_touch_zone(i, high, "synthetic_high", "buy_side", strength=0.5)
            lmap.add_or_touch_zone(i, low, "synthetic_low", "sell_side", strength=0.5)
            sweep = lmap.detect_sweep(i, high, low, next_close)
            liq_dist = lmap.nearest_distances(next_close)
            lmap.age_and_decay(i)

            temp_df = pd.DataFrame([{"high": high, "low": low}])
            fvg_book.detect_new(pd.concat([history_df[["high", "low"]].tail(2), temp_df], ignore_index=True), 2, state=next_state, session=session)
            fvg_book.update_fill(i, high, low)
            dist_fvg = fvg_book.nearest_open_distance(next_close)

            ts = ts + pd.Timedelta(seconds=self.timeframe_seconds)
            row = {
                "datetime": ts,
                "open": current_price,
                "high": high,
                "low": low,
                "close": next_close,
                "volume": float(history_df["volume"].mean() * (1 + amplitude * 10)),
                "state": next_state,
                "direction": direction,
                "amplitude": amplitude,
                "range": candle_range,
                "recent_sweep_flag": sweep["recent_sweep_flag"],
                "recent_sweep_side": sweep["recent_sweep_side"],
                "recent_sweep_strength": sweep["recent_sweep_strength"],
                "distance_to_nearest_buy_liquidity": liq_dist["distance_to_nearest_buy_liquidity"],
                "distance_to_nearest_sell_liquidity": liq_dist["distance_to_nearest_sell_liquidity"],
                "nearest_liquidity_strength": liq_dist["nearest_liquidity_strength"],
                "distance_to_nearest_open_fvg": dist_fvg if not np.isnan(dist_fvg) else candle_range * 3,
                "vol_regime_bucket": vol_bucket,
                "session_name": session,
            }
            out.append(row)
            current_state = next_state
            current_price = next_close

        sim = pd.DataFrame(out)
        sim["breakout_up"] = sim["close"] > sim["high"].rolling(20, min_periods=5).max().shift(1)
        sim["breakout_down"] = sim["close"] < sim["low"].rolling(20, min_periods=5).min().shift(1)
        sim["breakout_rejected"] = ((sim["high"] > sim["high"].rolling(20, min_periods=5).max().shift(1)) & (sim["close"] < sim["high"].rolling(20, min_periods=5).max().shift(1))).fillna(False)
        return sim
