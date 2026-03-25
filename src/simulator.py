from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from src.behavior_learning import LearnedBehavior
from src.feature_engineering import add_market_features
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

    def _direction_and_amplitude(self, key3: tuple[str, str, str], state: str, ctx: pd.Series, rng: np.random.Generator) -> tuple[int, float]:
        current_context = {
            "realized_vol": float(ctx.get("realized_vol", 0.0)),
            "distance_to_nearest_buy_liquidity": float(ctx.get("distance_to_nearest_buy_liquidity", 3.0)),
            "distance_to_nearest_sell_liquidity": float(ctx.get("distance_to_nearest_sell_liquidity", 3.0)),
            "compression_score": float(ctx.get("compression_score", 1.0)),
            "expansion_score": float(ctx.get("expansion_score", 1.0)),
            "recent_sweep_strength": float(ctx.get("recent_sweep_strength", 0.0)),
        }
        similar_indices = self.learned.find_similar_contexts(current_context, state, n_neighbors=15)
        cont3 = 0.5
        if similar_indices:
            chosen_idx = int(rng.choice(similar_indices))
            historical_return = float(self.learned.historical_contexts.iloc[chosen_idx]["log_return"])
            direction = 1 if historical_return >= 0 else -1
            amplitude = abs(historical_return) * abs(float(rng.normal(1.0, 0.1)))
        else:
            rstats = self.learned.conditional_returns.get(key3, {"mu": 0.0, "sigma": self.learned.volatility_stats["log_return_sigma"], "cont_3": 0.5})
            sampled = rng.normal(rstats["mu"], rstats["sigma"])
            direction = 1 if sampled >= 0 else -1
            amplitude = abs(sampled)
            cont3 = float(rstats.get("cont_3", 0.5))

        if ctx.get("recent_sweep_flag", 0) == 1:
            cont = self.learned.sweep_stats.get("continuation_after_sweep_buy", 0.5) if ctx.get("recent_sweep_side", 0) == 1 else self.learned.sweep_stats.get("continuation_after_sweep_sell", 0.5)
            if rng.random() > cont:
                direction *= -1
            amplitude *= 1.0 + min(1.0, float(ctx.get("recent_sweep_strength", 0.0)) * 8)

        if state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN"}:
            amplitude *= min(1.8, 1.0 + cont3)
        if state in {"BREAKOUT_REJECTED", "POST_SWEEP_REVERSAL"}:
            direction *= -1
            amplitude *= 0.8

        if ctx.get("distance_to_nearest_open_fvg", 9.0) < 1.0:
            amplitude *= 0.9
        return direction, max(1e-7, amplitude)

    def run(self, history_df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
        if history_df.empty:
            raise ValueError("historique vide")

        rng = np.random.default_rng(cfg.seed)
        hist = history_df[["datetime", "open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
        current_state = str(self.learned.feature_df["state"].iloc[-1]) if "state" in self.learned.feature_df.columns else "RANGE"

        # Build persistent maps from recent learned history (structure-based)
        logger = logging.getLogger(__name__)
        base_feat = self.learned.feature_df.tail(300).copy() if set(["swing_high", "swing_low"]).issubset(self.learned.feature_df.columns) else add_market_features(hist.tail(300).copy())
        base_feat = base_feat.reset_index(drop=True)
        logger.info("[SIM] bootstrap base_feat_len_before=%s", len(base_feat))
        if len(base_feat) < 5:
            raise ValueError("dataset insuffisant pour initialiser la simulation (moins de 5 lignes)")

        lmap = LiquidityMap()
        fvg = FVGBook()
        roll_h = base_feat["high"].rolling(30, min_periods=5).max()
        roll_l = base_feat["low"].rolling(30, min_periods=5).min()
        logger.info("[SIM] bootstrap roll_h_len=%s roll_l_len=%s", len(roll_h), len(roll_l))
        boot = base_feat.copy()
        boot["rolling_high"] = roll_h.reset_index(drop=True)
        boot["rolling_low"] = roll_l.reset_index(drop=True)
        before_dropna = len(boot)
        boot = boot.dropna(subset=["rolling_high", "rolling_low"]).reset_index(drop=True)
        dropped = before_dropna - len(boot)
        logger.info("[SIM] bootstrap aligned_len=%s dropped_for_rolling_nan=%s", len(boot), dropped)
        if boot.empty:
            raise ValueError("dataset insuffisant pour simulation: rolling windows vides après alignement")

        for i, r in boot.iterrows():
            lmap.ingest_feature_row(i, r, float(r["rolling_high"]), float(r["rolling_low"]))
            lmap.detect_sweep(i, float(r["high"]), float(r["low"]), float(r["close"]))
            lmap.age_and_decay(i)
            fvg.detect_new(boot, i, state=r.get("trend_context", ""), session=r.get("session_name", ""))
            fvg.update_fill(i, float(r["high"]), float(r["low"]))

        out = []
        for step in range(cfg.n_steps):
            feat_recent = add_market_features(hist.tail(300).copy())
            ref = feat_recent.iloc[-1]
            next_state = sample_next_state(current_state, ref, self.learned.transition_model.transition_probs, rng)

            vol_bucket = str(ref.get("vol_regime_bucket", "NORMAL"))
            session = str(ref.get("session_name", "LONDON_OPEN"))
            key3 = (next_state, vol_bucket, session)
            direction, amplitude = self._direction_and_amplitude(key3, next_state, ref, rng)

            range_stats = self.learned.conditional_ranges.get(key3, {"mu": self.learned.volatility_stats["range_mean"], "sigma": self.learned.volatility_stats["range_mean"] * 0.3})
            wstats = self.learned.conditional_wicks.get(key3, {"upper_mu": 0.25, "lower_mu": 0.25, "upper_sigma": 0.07, "lower_sigma": 0.07})

            current_price = float(hist["close"].iloc[-1])
            next_close = max(0.01, current_price * np.exp(direction * amplitude))
            candle_range = max(1e-8, rng.normal(range_stats["mu"], range_stats["sigma"]))
            high = max(current_price, next_close) + candle_range * max(0.0, rng.normal(wstats["upper_mu"], wstats["upper_sigma"]))
            low = min(current_price, next_close) - candle_range * max(0.0, rng.normal(wstats["lower_mu"], wstats["lower_sigma"]))

            local_scale = float(feat_recent["range"].tail(30).mean() or 1.0)
            liq_dist = lmap.nearest_distances(next_close, local_scale=local_scale)
            sweep = lmap.detect_sweep(step + len(boot), high, low, next_close)
            dist_fvg = fvg.nearest_open_distance(next_close)

            new_dt = pd.to_datetime(hist["datetime"].iloc[-1], utc=True) + pd.Timedelta(seconds=self.timeframe_seconds)
            new_row = {
                "datetime": new_dt,
                "open": current_price,
                "high": high,
                "low": low,
                "close": next_close,
                "volume": float(hist["volume"].mean() * (1 + amplitude * 10)),
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
                "distance_to_nearest_open_fvg": (dist_fvg / max(local_scale, 1e-8)) if not np.isnan(dist_fvg) else 3.0,
                "vol_regime_bucket": vol_bucket,
                "session_name": session,
            }
            out.append(new_row)

            hist = pd.concat([hist, pd.DataFrame([{k: new_row[k] for k in ["datetime", "open", "high", "low", "close", "volume"]}])], ignore_index=True)
            feat_new = add_market_features(hist.tail(60).copy()).iloc[-1]
            lmap.ingest_feature_row(step + len(boot), feat_new, rolling_high=float(hist["high"].tail(30).max()), rolling_low=float(hist["low"].tail(30).min()))
            lmap.age_and_decay(step + len(boot))
            tmp = pd.DataFrame([{"high": high, "low": low}])
            fvg.detect_new(pd.concat([hist[["high", "low"]].tail(2), tmp], ignore_index=True), 2, state=next_state, session=session)
            fvg.update_fill(step + len(boot), high, low)

            current_state = next_state

        sim = pd.DataFrame(out)
        sim["breakout_up"] = sim["close"] > sim["high"].rolling(20, min_periods=5).max().shift(1)
        sim["breakout_down"] = sim["close"] < sim["low"].rolling(20, min_periods=5).min().shift(1)
        sim["breakout_rejected"] = ((sim["high"] > sim["high"].rolling(20, min_periods=5).max().shift(1)) & (sim["close"] < sim["high"].rolling(20, min_periods=5).max().shift(1))).fillna(False)
        return sim
