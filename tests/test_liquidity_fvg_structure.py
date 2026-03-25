from __future__ import annotations

import pandas as pd

from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.market_structure import detect_swings_hierarchical, structure_labels_from_swings
from src.state_engine import build_transition_key


def test_liquidity_zone_persistence_and_sweep():
    lm = LiquidityMap(zone_width_bps=10)
    z1 = lm.add_or_touch_zone(0, 100.0, "equal_high", "buy_side", 1.0)
    z2 = lm.add_or_touch_zone(1, 100.02, "equal_high", "buy_side", 0.9)
    assert z1.id == z2.id
    assert z2.touch_count >= 2

    sweep = lm.detect_sweep(2, high=100.5, low=99.8, close=99.9)
    assert sweep["sweep_buy_side"] == 1
    assert z2.swept is True

    lm.age_and_decay(1000, deactivate_age=10)
    assert z2.active is False


def test_fvg_lifecycle_fill_ratio():
    df = pd.DataFrame(
        {
            "high": [100, 101, 102, 103],
            "low": [99, 100, 101.5, 102],
        }
    )
    book = FVGBook()
    created = book.detect_new(df, 2, state="TREND_UP", session="LONDON")
    assert len(created) == 1
    z = created[0]

    book.update_fill(3, high=102.0, low=101.0)
    assert z.partially_filled is True
    assert z.fill_ratio > 0

    book.update_fill(4, high=z.top + 1, low=z.bottom - 1)
    assert z.fully_filled is True
    assert z.fill_ratio == 1.0


def test_structure_hh_hl_lh_ll_bos_choch():
    df = pd.DataFrame(
        {
            "high": [10, 12, 11, 13, 12, 14, 13, 15],
            "low": [9, 10, 9.5, 11, 10.5, 12, 11, 13],
            "open": [9.5] * 8,
            "close": [10, 11, 10.2, 12.5, 11.2, 13.2, 11.1, 14.2],
        }
    )
    swings = detect_swings_hierarchical(df, window_minor=1, window_major=1)
    out = structure_labels_from_swings(df, swings)
    assert "recent_bos_flag" in out.columns
    assert "recent_choch_flag" in out.columns
    assert "trend_context" in out.columns


def test_sweep_vs_breakout_context_key():
    row = pd.Series(
        {
            "vol_regime_bucket": "HIGH",
            "distance_to_nearest_buy_liquidity": 0.2,
            "breakout_rejected": True,
            "recent_sweep_flag": 1,
            "session_name": "NEW_YORK",
        }
    )
    key = build_transition_key("BREAKOUT_REJECTED", row)
    assert key[2] == "NEAR"
    assert key[3] == "REJECTED"
    assert key[4] == "SWEEP"
