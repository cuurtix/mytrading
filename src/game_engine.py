from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_learning import CalibrationBundle
from src.feature_engineering import add_market_features
from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.state_engine import sample_next_state


@dataclass
class GamePosition:
    id: int
    side: str
    entry: float
    size: float
    leverage: int
    fee_paid: float


@dataclass
class GameAccount:
    balance: float = 10000.0
    realized_pnl: float = 0.0
    positions: List[GamePosition] = field(default_factory=list)


class TradingGameEngine:
    def __init__(self, bundle: CalibrationBundle, seed: int = 11):
        self.bundle = bundle
        self.rng = np.random.default_rng(seed)
        self.history = bundle.merged_df[["datetime", "open", "high", "low", "close", "volume"]].copy().tail(500).reset_index(drop=True)
        self.account = GameAccount()
        self.next_pos_id = 1
        self.pending_player_impact = 0.0

        self.lmap = LiquidityMap()
        self.fvg = FVGBook()
        feat = bundle.learned.feature_df.tail(300).copy()
        roll_h = feat["high"].rolling(30, min_periods=5).max()
        roll_l = feat["low"].rolling(30, min_periods=5).min()
        for i, r in feat.iterrows():
            self.lmap.ingest_feature_row(i, r, roll_h.iloc[i], roll_l.iloc[i])
            self.lmap.detect_sweep(i, float(r["high"]), float(r["low"]), float(r["close"]))
            self.lmap.age_and_decay(i)
            self.fvg.detect_new(feat, i, state=str(r.get("state", "RANGE")), session=str(r.get("session_name", "ASIA")))
            self.fvg.update_fill(i, float(r["high"]), float(r["low"]))

    def _mark_to_market(self, price: float) -> Dict[str, float]:
        upnl = 0.0
        exposure = 0.0
        margin = 0.0
        for p in self.account.positions:
            pnl = (price - p.entry) * p.size if p.side == "long" else (p.entry - price) * p.size
            upnl += pnl
            exposure += p.entry * p.size
            margin += (p.entry * p.size) / max(p.leverage, 1)
        equity = self.account.balance + upnl
        return {
            "balance": self.account.balance,
            "equity": equity,
            "unrealized_pnl": upnl,
            "realized_pnl": self.account.realized_pnl,
            "margin_used": margin,
            "free_margin": equity - margin,
            "exposure": exposure,
            "open_positions": len(self.account.positions),
        }

    def _apply_player_impact(self, side: str, size: float, local_volume: float, local_liquidity_strength: float) -> Dict[str, float]:
        rel = size / max(local_volume * max(local_liquidity_strength, 0.1), 1e-8)
        impact = np.sign(1 if side == "buy" else -1) * min(0.02, 0.002 * np.sqrt(abs(rel)))
        spread_widen = min(0.8, abs(rel) * 0.2)
        self.pending_player_impact += impact
        return {"impact": float(impact), "spread_widen": float(spread_widen), "slippage": float(abs(impact) * 0.5)}

    def place_order(self, side: str, size: float, leverage: int = 50) -> Dict[str, float]:
        last = self.history.iloc[-1]
        local_volume = float(self.history["volume"].tail(30).mean())
        liq = self.lmap.nearest_distances(float(last["close"]), local_scale=float(self.history["close"].pct_change().std() or 1.0))
        exec_impact = self._apply_player_impact(side, size, local_volume, liq["nearest_liquidity_strength"])

        mid = float(last["close"])
        base_spread = 0.15
        spread = base_spread * (1 + exec_impact["spread_widen"])
        fill = mid + spread / 2 + exec_impact["slippage"] if side == "buy" else mid - spread / 2 - exec_impact["slippage"]
        fee = abs(fill * size) * 0.0002

        pos = GamePosition(
            id=self.next_pos_id,
            side="long" if side == "buy" else "short",
            entry=float(fill),
            size=float(size),
            leverage=int(leverage),
            fee_paid=float(fee),
        )
        self.next_pos_id += 1
        self.account.positions.append(pos)
        return {"fill": float(fill), "fee": float(fee), **exec_impact}

    def close_fraction(self, fraction: float) -> Dict[str, float]:
        fraction = max(0.0, min(1.0, fraction))
        price = float(self.history["close"].iloc[-1])
        realized = 0.0
        remaining = []
        for p in self.account.positions:
            close_size = p.size * fraction
            keep_size = p.size - close_size
            pnl = ((price - p.entry) if p.side == "long" else (p.entry - price)) * close_size
            realized += pnl - p.fee_paid * fraction
            if keep_size > 1e-9:
                p.size = keep_size
                p.fee_paid *= (1 - fraction)
                remaining.append(p)
        self.account.positions = remaining
        self.account.balance += realized
        self.account.realized_pnl += realized
        return {"realized": float(realized)}

    def close_all(self) -> Dict[str, float]:
        return self.close_fraction(1.0)

    def deposit(self, amount: float) -> None:
        self.account.balance += max(0.0, amount)

    def withdraw(self, amount: float) -> bool:
        m = self._mark_to_market(float(self.history["close"].iloc[-1]))
        amount = max(0.0, amount)
        if m["free_margin"] - amount < 0:
            return False
        self.account.balance -= amount
        return True

    def step_market(self) -> Dict[str, object]:
        feat = add_market_features(self.history.tail(300).copy())
        ref = feat.iloc[-1]
        current_state = str(self.bundle.learned.feature_df["state"].iloc[-1]) if "state" in self.bundle.learned.feature_df.columns else "RANGE"
        next_state = sample_next_state(current_state, ref, self.bundle.learned.transition_model.transition_probs, self.rng)

        vol_bucket = str(ref.get("vol_regime_bucket", "NORMAL"))
        session = str(ref.get("session_name", "LONDON_OPEN"))
        key = (next_state, vol_bucket, session)
        rstats = self.bundle.learned.conditional_returns.get(key, {"mu": 0.0, "sigma": self.bundle.learned.volatility_stats["log_return_sigma"], "cont_3": 0.5})
        range_stats = self.bundle.learned.conditional_ranges.get(key, {"mu": self.bundle.learned.volatility_stats["range_mean"], "sigma": self.bundle.learned.volatility_stats["range_mean"] * 0.3})
        wstats = self.bundle.learned.conditional_wicks.get(key, {"upper_mu": 0.25, "lower_mu": 0.25, "upper_sigma": 0.08, "lower_sigma": 0.08})

        sampled = self.rng.normal(rstats["mu"], rstats["sigma"])
        direction = 1 if sampled >= 0 else -1
        amplitude = abs(sampled)

        # bounded FOMO + player impact + stop hunt cascade
        if next_state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN"}:
            amplitude *= min(1.8, 1.0 + rstats.get("cont_3", 0.5))
        amplitude += abs(self.pending_player_impact)
        direction = np.sign(direction + np.sign(self.pending_player_impact) * min(1.0, abs(self.pending_player_impact) * 50)) or direction
        self.pending_player_impact *= 0.5

        current_price = float(self.history["close"].iloc[-1])
        next_close = max(0.01, current_price * np.exp(direction * amplitude))
        candle_range = max(1e-8, self.rng.normal(range_stats["mu"], range_stats["sigma"]))
        high = max(current_price, next_close) + candle_range * max(0.0, self.rng.normal(wstats["upper_mu"], wstats["upper_sigma"]))
        low = min(current_price, next_close) - candle_range * max(0.0, self.rng.normal(wstats["lower_mu"], wstats["lower_sigma"]))

        local_scale = float(feat["range"].tail(30).mean() or 1.0)
        sweep = self.lmap.detect_sweep(len(self.history), high, low, next_close)
        # stop hunt / cascade effect
        if sweep["recent_sweep_flag"]:
            cascade = min(0.003, sweep["recent_sweep_strength"] * 3)
            next_close = max(0.01, next_close * (1 + (1 if sweep["recent_sweep_side"] == -1 else -1) * cascade))

        new_dt = pd.to_datetime(self.history["datetime"].iloc[-1], utc=True) + pd.Timedelta(seconds=self.bundle.timeframe_seconds)
        row = {
            "datetime": new_dt,
            "open": current_price,
            "high": max(high, next_close),
            "low": min(low, next_close),
            "close": next_close,
            "volume": float(self.history["volume"].tail(50).mean() * (1 + amplitude * 8)),
        }
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True).tail(2000).reset_index(drop=True)

        feat_new = add_market_features(self.history.tail(80).copy()).iloc[-1]
        self.lmap.ingest_feature_row(len(self.history), feat_new, rolling_high=float(self.history["high"].tail(30).max()), rolling_low=float(self.history["low"].tail(30).min()))
        self.lmap.age_and_decay(len(self.history))

        tmp = pd.DataFrame([{"high": row["high"], "low": row["low"]}])
        self.fvg.detect_new(pd.concat([self.history[["high", "low"]].tail(2), tmp], ignore_index=True), 2, state=next_state, session=session)
        self.fvg.update_fill(len(self.history), row["high"], row["low"])

        metrics = self._mark_to_market(float(next_close))
        return {
            "candle": row,
            "metrics": metrics,
            "state": next_state,
            "sweep": sweep,
            "fvg_open": self.fvg.open_count_nearby(float(next_close), threshold=local_scale * 1.5),
            "positions": [p.__dict__ for p in self.account.positions],
        }

    def reset(self) -> None:
        self.__init__(self.bundle)
