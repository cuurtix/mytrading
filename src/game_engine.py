from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_learning import CalibrationBundle
from src.feature_engineering import add_market_features
from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.logic_monitor import validate_ohlc, validate_snapshot
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

    def __init__(self, bundle: CalibrationBundle, seed: int = 11):
        self.bundle = bundle
        self.rng = np.random.default_rng(seed)
        self.history = bundle.merged_df[["datetime", "open", "high", "low", "close", "volume"]].copy().tail(500).reset_index(drop=True)
        self.account = GameAccount()
        self.next_pos_id = 1
        self.pending_player_impact = 0.0
        self.current_state = str(bundle.learned.feature_df["state"].iloc[-1]) if "state" in bundle.learned.feature_df.columns else "RANGE"
        self.recent_events: List[str] = []

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

    def _push_event(self, msg: str) -> None:
        self.recent_events = ([msg] + self.recent_events)[:30]

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
        }
        errs = validate_snapshot(snap)
        if errs:
            self._push_event("MONITOR: " + ",".join(errs))
        return snap

    def _reference_market_volume_notional(self, session: str, local_notional: float, liquidity_strength: float) -> float:
        session_notional = 361e9 / 24.0
        session_mult = {
            "ASIA": 0.8,
            "LONDON_OPEN": 1.2,
            "NEW_YORK": 1.3,
            "LATE_SESSION": 0.7,
        }.get(session, 1.0)
        session_ref = session_notional * session_mult
        local_ref = max(local_notional, 1e6)
        liquidity_ref = max(1e6, liquidity_strength * 5e7)
        return 0.2 * session_ref + 0.5 * local_ref + 0.3 * liquidity_ref

    def _compute_order_impact(self, side: str, order_notional: float, ref_notional: float, local_vol: float) -> Dict[str, float]:
        participation = order_notional / max(ref_notional, 1e-8)
        activation_threshold = 5e-6
        if participation < activation_threshold:
            impact = 0.0
        else:
            sign = 1.0 if side == "buy" else -1.0
            Y = 0.55
            raw = sign * Y * max(local_vol, 1e-6) * np.sqrt(participation)
            impact = float(np.clip(raw, -0.02, 0.02))
        spread_widen = min(0.8, max(0.0, np.sqrt(participation) * 0.6))
        slippage = abs(impact) * (1.0 + local_vol)
        return {"impact": impact, "spread_widen": spread_widen, "slippage": slippage, "participation": participation}

    def place_order(self, side: str, size: float, leverage: int = 50) -> Dict[str, object]:
        last = self.history.iloc[-1]
        mid = float(last["close"])
        notional = abs(mid * size)
        fee = notional * 0.0002

        # validation marge avant application impact
        margin_required = notional / max(leverage, 1)
        free_margin = self._mark_to_market(mid)["free_margin"]
        if free_margin < (margin_required + fee):
            return {"ok": False, "reason": "Marge insuffisante", "required_margin": margin_required, "free_margin": free_margin, "snapshot": self.snapshot()}

        local_volume_notional = float((self.history["close"].tail(30) * self.history["volume"].tail(30)).mean())
        local_scale = float(self.history["close"].pct_change().std() or 1.0)
        liq = self.lmap.nearest_distances(mid, local_scale=local_scale)
        session = xauusd_session_name(pd.to_datetime(last["datetime"], utc=True))
        ref_notional = self._reference_market_volume_notional(session, local_volume_notional, liq["nearest_liquidity_strength"])
        exec_impact = self._compute_order_impact(side, notional, ref_notional, local_scale)

        # impact appliqué uniquement si ordre accepté
        self.pending_player_impact += exec_impact["impact"]

        spread = 0.15 * (1 + exec_impact["spread_widen"])
        fill = mid + spread / 2 + exec_impact["slippage"] if side == "buy" else mid - spread / 2 - exec_impact["slippage"]

        # frais débités immédiatement
        self.account.balance -= fee
        self.account.realized_pnl -= fee

        pos = GamePosition(id=self.next_pos_id, side="long" if side == "buy" else "short", entry=float(fill), size=float(size), leverage=int(leverage))
        self.next_pos_id += 1
        self.account.positions.append(pos)
        self._push_event(f"ORDER {side.upper()} accepted size={size} fill={fill:.2f} fee={fee:.2f}")
        return {"ok": True, "execution": {"fill": float(fill), "fee": float(fee), **exec_impact}, "snapshot": self.snapshot()}

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

    def step_market(self) -> Dict[str, object]:
        feat = add_market_features(self.history.tail(300).copy())
        ref = feat.iloc[-1]
        next_state = sample_next_state(self.current_state, ref, self.bundle.learned.transition_model.transition_probs, self.rng)

        vol_bucket = str(ref.get("vol_regime_bucket", "NORMAL"))
        session = str(ref.get("session_name", "LONDON_OPEN"))
        key = (next_state, vol_bucket, session)
        rstats = self.bundle.learned.conditional_returns.get(key, {"mu": 0.0, "sigma": self.bundle.learned.volatility_stats["log_return_sigma"], "cont_3": 0.5})
        range_stats = self.bundle.learned.conditional_ranges.get(key, {"mu": self.bundle.learned.volatility_stats["range_mean"], "sigma": self.bundle.learned.volatility_stats["range_mean"] * 0.3})
        wstats = self.bundle.learned.conditional_wicks.get(key, {"upper_mu": 0.25, "lower_mu": 0.25, "upper_sigma": 0.08, "lower_sigma": 0.08})

        sampled = self.rng.normal(rstats["mu"], rstats["sigma"])
        direction = 1 if sampled >= 0 else -1
        amplitude = abs(sampled)
        if next_state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN", "HIGH_VOLATILITY_PANIC"}:
            amplitude *= min(1.9, 1.0 + rstats.get("cont_3", 0.5))
        if next_state == "LOW_VOLATILITY_COMPRESSION":
            amplitude *= 0.6

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
        participation_effective = min(1.0, abs(self.pending_player_impact) * 100)
        breakout_score = 1.0 if next_state in {"BREAKOUT_ACCEPTED", "EXPANSION_UP", "EXPANSION_DOWN"} else 0.0
        vol_score = 1.0 if next_state == "HIGH_VOLATILITY_PANIC" else 0.4
        cascade_score = 0.6 * sweep["recent_sweep_strength"] + 0.5 * breakout_score + 0.4 * participation_effective + 0.3 * vol_score
        if sweep["recent_sweep_flag"] and cascade_score > 0.65:
            cascade = min(0.004, cascade_score * 0.002)
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
