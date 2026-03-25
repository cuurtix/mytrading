from __future__ import annotations

from typing import Dict, List


VALID_SESSIONS = {"ASIA", "LONDON_OPEN", "NEW_YORK", "LATE_SESSION"}


def validate_ohlc(candle: Dict[str, float]) -> List[str]:
    errs = []
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    if h < max(o, c):
        errs.append("high < max(open, close)")
    if l > min(o, c):
        errs.append("low > min(open, close)")
    if h < l:
        errs.append("high < low")
    return errs


def validate_metrics(metrics: Dict[str, float]) -> List[str]:
    errs = []
    if abs((metrics["balance"] + metrics["unrealized_pnl"]) - metrics["equity"]) > 1e-6:
        errs.append("equity mismatch")
    if abs((metrics["equity"] - metrics["margin_used"]) - metrics["free_margin"]) > 1e-6:
        errs.append("free_margin mismatch")
    return errs


def validate_snapshot(snapshot: Dict[str, object]) -> List[str]:
    req = ["metrics", "positions", "last_price", "state", "timestamp", "recent_events"]
    errs = [f"missing_{k}" for k in req if k not in snapshot]
    metrics = snapshot.get("metrics")
    if isinstance(metrics, dict):
        errs.extend(validate_metrics(metrics))
    state = snapshot.get("state")
    if state is None:
        errs.append("missing_state")
    return errs
