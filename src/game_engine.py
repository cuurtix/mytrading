from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import logging

import numpy as np
import pandas as pd

from src.data_learning import CalibrationBundle
from src.feature_engineering import add_market_features
from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.logic_monitor import VALID_SESSIONS, validate_ohlc, validate_snapshot
from src.model_config import ModelConfig
from src.sessions import xauusd_session_name
from src.state_engine import sample_next_state


@dataclass
class GamePosition:
    id: int
    side: str
    entry: float
    size: float
    leverage: int


@dataclass
class GameAccount:
    balance: float = 10000.0
    realized_pnl: float = 0.0
    positions: List[GamePosition] = field(default_factory=list)


class TradingGameEngine:
    maintenance_margin_ratio: float = 0.5

    def __init__(self, bundle: CalibrationBundle, seed: int = 11, config: ModelConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.bundle = bundle
        self.rng = np.random.default_rng(seed)
        self.config = config or ModelConfig()
        self.history = bundle.merged_df[["datetime", "open", "high", "low", "close", "volume"]].copy().tail(500).reset_index(drop=True)
        self.account = GameAccount()
        self.next_pos_id = 1

        self.current_state = str(bundle.learned.feature_df["state"].iloc[-1]) if "state" in bundle.learned.feature_df.columns else "RANGE"
        self.pending_player_impact = 0.0  # pression prix résiduelle
        self.recent_order_participation = 0.0
        self.recent_order_impact = 0.0
        self.recent_order_notional = 0.0
        self.recent_order_side = "none"
        self.last_order_rejected_had_impact = False
        self.recent_events: List[str] = []

        self.lmap = LiquidityMap()
        self.fvg = FVGBook()
        feat = bundle.learned.feature_df.tail(300).copy().reset_index(drop=True)
        roll_h = feat["high"].rolling(30, min_periods=5).max().reset_index(drop=True)
        roll_l = feat["low"].rolling(30, min_periods=5).min().reset_index(drop=True)
        feat_boot = feat.copy()
        feat_boot["rolling_high"] = roll_h
        feat_boot["rolling_low"] = roll_l
        init_len = len(feat_boot)
        feat_boot = feat_boot.dropna(subset=["rolling_high", "rolling_low"]).reset_index(drop=True)
        dropped = init_len - len(feat_boot)
        self.logger.info("[GAME] bootstrap feature rows initial=%s aligned=%s dropped=%s", init_len, len(feat_boot), dropped)

        for i, r in feat_boot.iterrows():
            self.lmap.ingest_feature_row(i, r, float(r["rolling_high"]), float(r["rolling_low"]))
            self.lmap.detect_sweep(i, float(r["high"]), float(r["low"]), float(r["close"]))
            self.lmap.age_and_decay(i)
            self.fvg.detect_new(feat_boot, i, state=str(r.get("state", "RANGE")), session=str(r.get("session_name", "ASIA")))
            self.fvg.update_fill(i, float(r["high"]), float(r["low"]))

    def _push_event(self, msg: str) -> None:
        self.recent_events = ([msg] + self.recent_events)[:40]

    def _mark_to_market(self, price: float) -> Dict[str, float]:
        upnl, exposure, margin = 0.0, 0.0, 0.0
        for p in self.account.positions:
            pnl = (price - p.entry) * p.size if p.side == "long" else (p.entry - price) * p.size
            upnl += pnl
            exposure += p.entry * p.size
            margin += (p.entry * p.size) / max(p.leverage, 1)
        equity = self.account.balance + upnl
        return {
            "balance": float(self.account.balance),
            "equity": float(equity),
            "unrealized_pnl": float(upnl),
            "realized_pnl": float(self.account.realized_pnl),
            "margin_used": float(margin),
            "free_margin": float(equity - margin),
            "exposure": float(exposure),
            "open_positions": len(self.account.positions),
        }

    def _maintenance_margin(self, price: float) -> float:
        return self._mark_to_market(price)["margin_used"] * self.maintenance_margin_ratio

    def _enforce_liquidation_if_needed(self) -> bool:
        price = float(self.history["close"].iloc[-1])
        m = self._mark_to_market(price)
        if m["equity"] <= self._maintenance_margin(price) and self.account.positions:
            realized = self.close_fraction(1.0, reason="liquidation")
            self._push_event(f"LIQUIDATION forcée: realized={realized['realized']:.2f}")
            return True
        return False

    def snapshot(self) -> Dict[str, object]:
        last_price = float(self.history["close"].iloc[-1])
        snap = {
            "metrics": self._mark_to_market(last_price),
            "positions": [p.__dict__ for p in self.account.positions],
            "last_price": last_price,
            "state": self.current_state,
            "timestamp": str(self.history["datetime"].iloc[-1]),
            "recent_events": self.recent_events,
            "recent_order_participation": self.recent_order_participation,
            "recent_order_impact": self.recent_order_impact,
        }
        errs = validate_snapshot(snap)
        if errs:
            self._push_event("MONITOR: " + ",".join(errs))
        return snap

    # -------- impact model --------
    def _local_executable_volume_notional(self, session: str, local_notional: float, liquidity_strength: float) -> float:
        session_mult = {"ASIA": 0.8, "LONDON_OPEN": 1.15, "NEW_YORK": 1.25, "LATE_SESSION": 0.7}.get(session, 1.0)
        liq_capacity = max(self.config.min_liquidity_capacity, liquidity_strength * self.config.liquidity_strength_to_notional)
        local_exec = session_mult * (self.config.local_volume_weight * local_notional + self.config.liquidity_capacity_weight * liq_capacity)
        return max(self.config.min_local_executable_notional, local_exec)

    def _global_market_reference_notional(self) -> float:
        return float(self.config.global_gold_reference_daily_notional)  # garde-fou macro configurable, pas le driver principal gameplay

    def _compute_order_impact(self, side: str, order_notional: float, local_executable_notional: float, sigma_local: float) -> Dict[str, float]:
        effective_participation = order_notional / max(local_executable_notional, 1e-8)
        macro_guardrail = order_notional / max(self._global_market_reference_notional(), 1e-8)

        activation_threshold = self.config.impact_activation_threshold
        if effective_participation < activation_threshold:
            impact = 0.0
        else:
            sign = 1.0 if side == "buy" else -1.0
            Y = self.config.impact_y
            raw = sign * Y * max(sigma_local, 1e-6) * np.sqrt(effective_participation)
            raw *= (1.0 - min(0.6, np.sqrt(macro_guardrail)))
            impact = float(np.clip(raw, -self.config.impact_clip_abs, self.config.impact_clip_abs))

        # Corrélations participation -> spread/slippage
        spread_widen = float(min(1.0, np.sqrt(effective_participation) * (1 + sigma_local * 10) * self.config.spread_participation_multiplier))
        slippage = float(abs(impact) * (1.0 + spread_widen + sigma_local * 8))
        return {
            "impact": impact,
            "spread_widen": spread_widen,
            "slippage": slippage,
            "effective_participation": float(effective_participation),
            "macro_guardrail": float(macro_guardrail),
        }

    # -------- actions --------
    def place_order(self, side: str, size: float, leverage: int = 50) -> Dict[str, object]:
        last = self.history.iloc[-1]
        mid = float(last["close"])
        notional = abs(mid * size)
        fee = notional * 0.0002

        margin_required = notional / max(leverage, 1)
        free_margin = self._mark_to_market(mid)["free_margin"]
        if free_margin < (margin_required + fee):
            self.last_order_rejected_had_impact = False
            self._push_event(f"ORDER {side.upper()} refusé: marge insuffisante")
            return {"ok": False, "reason": "Marge insuffisante", "required_margin": margin_required, "free_margin": free_margin, "snapshot": self.snapshot()}

        local_notional = float((self.history["close"].tail(30) * self.history["volume"].tail(30)).mean())
        sigma_local = float(self.history["close"].pct_change().tail(60).std() or 1e-4)
        liq = self.lmap.nearest_distances(mid, local_scale=max(sigma_local, 1e-6))
        session = xauusd_session_name(pd.to_datetime(last["datetime"], utc=True))

        local_exec_notional = self._local_executable_volume_notional(session, local_notional, liq["nearest_liquidity_strength"])
        impact_info = self._compute_order_impact(side, notional, local_exec_notional, sigma_local)

        # impact appliqué seulement si accepté
        self.pending_player_impact += impact_info["impact"]
        self.recent_order_participation = impact_info["effective_participation"]
        self.recent_order_impact = impact_info["impact"]
        self.recent_order_notional = notional
        self.recent_order_side = side

        spread = 0.15 * (1 + impact_info["spread_widen"])
        fill = mid + spread / 2 + impact_info["slippage"] if side == "buy" else mid - spread / 2 - impact_info["slippage"]

        self.account.balance -= fee
        self.account.realized_pnl -= fee

        pos = GamePosition(id=self.next_pos_id, side="long" if side == "buy" else "short", entry=float(fill), size=float(size), leverage=int(leverage))
        self.next_pos_id += 1
        self.account.positions.append(pos)
        self._push_event(f"ORDER {side.upper()} accepted size={size} fill={fill:.2f} fee={fee:.2f}")
        return {"ok": True, "execution": {"fill": float(fill), "fee": float(fee), **impact_info}, "snapshot": self.snapshot()}

    def close_fraction(self, fraction: float, reason: str = "user") -> Dict[str, object]:
        fraction = max(0.0, min(1.0, fraction))
        price = float(self.history["close"].iloc[-1])
        realized = 0.0
        remaining = []
        for p in self.account.positions:
            close_size = p.size * fraction
            keep_size = p.size - close_size
            pnl = ((price - p.entry) if p.side == "long" else (p.entry - price)) * close_size
            realized += pnl
            if keep_size > 1e-9:
                p.size = keep_size
                remaining.append(p)
        self.account.positions = remaining
        self.account.balance += realized
        self.account.realized_pnl += realized
        self._push_event(f"CLOSE fraction={fraction:.2f} reason={reason} realized={realized:.2f}")
        return {"ok": True, "realized": float(realized), "fraction_applied_each_position": fraction, "snapshot": self.snapshot()}

    def close_all(self) -> Dict[str, object]:
        return self.close_fraction(1.0)

    def deposit(self, amount: float) -> Dict[str, object]:
        self.account.balance += max(0.0, amount)
        self._push_event(f"DEPOSIT {amount:.2f}")
        return {"ok": True, "snapshot": self.snapshot()}

    def withdraw(self, amount: float) -> Dict[str, object]:
        m = self._mark_to_market(float(self.history["close"].iloc[-1]))
        amount = max(0.0, amount)
        if m["free_margin"] - amount < 0:
            return {"ok": False, "reason": "Retrait impossible (marge)", "snapshot": self.snapshot()}
        self.account.balance -= amount
        self._push_event(f"WITHDRAW {amount:.2f}")
        return {"ok": True, "snapshot": self.snapshot()}

    def _direction_and_amplitude_contextual(self, key3: tuple[str, str, str], state: str, ctx: pd.Series) -> tuple[int, float]:
        current_context = {
            "realized_vol": float(ctx.get("realized_vol", 0.0)),
            "distance_to_nearest_buy_liquidity": float(ctx.get("distance_to_nearest_buy_liquidity", 3.0)),
            "distance_to_nearest_sell_liquidity": float(ctx.get("distance_to_nearest_sell_liquidity", 3.0)),
            "compression_score": float(ctx.get("compression_score", 1.0)),
            "expansion_score": float(ctx.get("expansion_score", 1.0)),
            "recent_sweep_strength": float(ctx.get("recent_sweep_strength", 0.0)),
        }
        similar = self.bundle.learned.find_similar_contexts(current_context, state, n_neighbors=15)
        if similar:
            idx = int(self.rng.choice(similar))
            hist_ret = float(self.bundle.learned.historical_contexts.iloc[idx]["log_return"])
            direction = 1 if hist_ret >= 0 else -1
            amplitude = abs(hist_ret) * abs(float(self.rng.normal(1.0, 0.1)))
            cont3 = 0.5
        else:
            rstats = self.bundle.learned.conditional_returns.get(key3, {"mu": 0.0, "sigma": self.bundle.learned.volatility_stats["log_return_sigma"], "cont_3": 0.5})
            sampled = float(self.rng.normal(rstats["mu"], rstats["sigma"]))
            direction = 1 if sampled >= 0 else -1
            amplitude = abs(sampled)
            cont3 = float(rstats.get("cont_3", 0.5))

        if state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN", "HIGH_VOLATILITY_PANIC"}:
            amplitude *= min(1.9, 1.0 + cont3)
        if state == "LOW_VOLATILITY_COMPRESSION":
            amplitude *= 0.6
        return int(direction), max(1e-7, float(amplitude))

    # -------- market loop --------
    def step_market(self) -> Dict[str, object]:
        feat = add_market_features(self.history.tail(300).copy())
        ref = feat.iloc[-1]
        next_state = sample_next_state(self.current_state, ref, self.bundle.learned.transition_model.transition_probs, self.rng)

        vol_bucket = str(ref.get("vol_regime_bucket", "NORMAL"))
        session = str(ref.get("session_name", "LONDON_OPEN"))
        if session not in VALID_SESSIONS:
            session = "ASIA"

        key = (next_state, vol_bucket, session)
        range_stats = self.bundle.learned.conditional_ranges.get(key, {"mu": self.bundle.learned.volatility_stats["range_mean"], "sigma": self.bundle.learned.volatility_stats["range_mean"] * 0.3})
        wstats = self.bundle.learned.conditional_wicks.get(key, {"upper_mu": 0.25, "lower_mu": 0.25, "upper_sigma": 0.08, "lower_sigma": 0.08})

        direction, amplitude = self._direction_and_amplitude_contextual(key, next_state, ref)

        amplitude += abs(self.pending_player_impact)
        direction = np.sign(direction + np.sign(self.pending_player_impact) * min(1.0, abs(self.pending_player_impact) * 50)) or direction
        self.pending_player_impact *= self.config.pending_impact_decay

        current_price = float(self.history["close"].iloc[-1])
        next_close = max(0.01, current_price * np.exp(direction * amplitude))
        candle_range = max(1e-8, self.rng.normal(range_stats["mu"], range_stats["sigma"]))
        high = max(current_price, next_close) + candle_range * max(0.0, self.rng.normal(wstats["upper_mu"], wstats["upper_sigma"]))
        low = min(current_price, next_close) - candle_range * max(0.0, self.rng.normal(wstats["lower_mu"], wstats["lower_sigma"]))

        local_scale = float(feat["range"].tail(30).mean() or 1.0)
        liq_dist = self.lmap.nearest_distances(next_close, local_scale=max(local_scale, 1e-6))
        fvg_dist = self.fvg.nearest_open_distance(next_close)
        fvg_dist_rel = fvg_dist / max(local_scale, 1e-8) if not np.isnan(fvg_dist) else np.nan
        near_fvg = bool(not np.isnan(fvg_dist_rel) and fvg_dist_rel <= self.config.fvg_rebalance_distance_threshold)
        if near_fvg and next_state == "REBALANCING_TO_FVG":
            next_close = max(0.01, current_price + (next_close - current_price) * self.config.fvg_rebalance_damping)
        sweep = self.lmap.detect_sweep(len(self.history), high, low, next_close)

        # sweep/cascade conditionnelle corrélée participation/liquidité/état/vol
        proximity_score = max(0.0, 1.0 - min(liq_dist["distance_to_nearest_buy_liquidity"], liq_dist["distance_to_nearest_sell_liquidity"]))
        breakout_score = 1.0 if next_state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN"} else 0.0
        vol_score = 1.0 if next_state == "HIGH_VOLATILITY_PANIC" else 0.4
        sweep_trigger_score = 0.35 * proximity_score + 0.35 * min(1.0, self.recent_order_participation * 2000) + 0.2 * breakout_score + 0.1 * vol_score
        if sweep["recent_sweep_flag"] and sweep_trigger_score > self.config.sweep_trigger_threshold:
            cascade_score = 0.5 * sweep["recent_sweep_strength"] + 0.25 * breakout_score + 0.2 * min(1.0, self.recent_order_participation * 2000) + 0.05 * vol_score
            if cascade_score > self.config.cascade_trigger_threshold:
                cascade = min(self.config.cascade_cap, cascade_score * self.config.cascade_multiplier)
                next_close = max(0.01, next_close * (1 + (1 if sweep["recent_sweep_side"] == -1 else -1) * cascade))
                self._push_event(f"STOP_CASCADE score={cascade_score:.3f}")

        new_dt = pd.to_datetime(self.history["datetime"].iloc[-1], utc=True) + pd.Timedelta(seconds=self.bundle.timeframe_seconds)
        row = {"datetime": new_dt, "open": current_price, "high": max(high, next_close), "low": min(low, next_close), "close": next_close, "volume": float(self.history["volume"].tail(50).mean() * (1 + amplitude * 8))}
        errs = validate_ohlc(row)
        if errs:
            self._push_event("MONITOR_OHLC: " + ",".join(errs))
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True).tail(2000).reset_index(drop=True)

        feat_new = add_market_features(self.history.tail(80).copy()).iloc[-1]
        self.lmap.ingest_feature_row(len(self.history), feat_new, rolling_high=float(self.history["high"].tail(30).max()), rolling_low=float(self.history["low"].tail(30).min()))
        self.lmap.age_and_decay(len(self.history))
        tmp = pd.DataFrame([{"high": row["high"], "low": row["low"]}])
        self.fvg.detect_new(pd.concat([self.history[["high", "low"]].tail(2), tmp], ignore_index=True), 2, state=next_state, session=session)
        self.fvg.update_fill(len(self.history), row["high"], row["low"])

        self.current_state = next_state
        self._enforce_liquidation_if_needed()
        return {"ok": True, "candle": row, "state": next_state, "sweep": sweep, "snapshot": self.snapshot()}

    def reset(self) -> Dict[str, object]:
        learned_bundle = self.bundle
        self.__init__(learned_bundle)
        self._push_event("RESET TOTAL")
        return {"ok": True, "snapshot": self.snapshot()}
