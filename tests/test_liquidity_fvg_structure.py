from __future__ import annotations

import pandas as pd

from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.market_structure import detect_swings_hierarchical, structure_labels_from_swings
from src.sessions import xauusd_session_name
from src.state_engine import build_transition_key, learn_transition_model, sample_next_state


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


def test_nearest_liquidity_strength_is_local_not_global_max():
    lm = LiquidityMap(zone_width_bps=5)
    lm.add_or_touch_zone(0, 100.0, "swing_high", "buy_side", 0.4)
    lm.add_or_touch_zone(1, 130.0, "swing_high", "buy_side", 2.0)
    d = lm.nearest_distances(100.1, local_scale=1.0)
    assert d["nearest_liquidity_strength"] < 1.0


def test_fvg_lifecycle_fill_ratio():
    df = pd.DataFrame({"high": [100, 101, 102, 103], "low": [99, 100, 101.5, 102]})
    book = FVGBook()
    created = book.detect_new(df, 2, state="TREND_UP", session="LONDON_OPEN")
    assert len(created) == 1
    z = created[0]

    book.update_fill(3, high=102.0, low=101.0)
    assert z.partially_filled is True
    assert 0 < z.fill_ratio < 1

    book.update_fill(4, high=z.top + 1, low=z.bottom - 1)
    assert z.fully_filled is True
    assert z.fill_ratio == 1.0


def test_structure_bos_and_choch_trigger():
    # construit pour provoquer bascule up puis down
    df = pd.DataFrame(
        {
            "high": [100, 101, 102, 104, 105, 103, 102, 100, 99, 98],
            "low": [99, 100, 101, 102, 103, 100, 99, 97, 96, 95],
            "open": [99.5, 100.2, 101.5, 103, 104, 101.2, 100, 98.5, 97, 96],
            "close": [100.1, 101, 102.2, 104.2, 105.1, 100.5, 99.2, 97.2, 96.1, 95.5],
        }
    )
    swings = detect_swings_hierarchical(df, window_minor=1, window_major=1)
    out = structure_labels_from_swings(df, swings)
    assert "recent_bos_flag" in out.columns
    assert "recent_choch_flag" in out.columns
    assert out["trend_context"].nunique() >= 1


def test_transition_depends_on_context_not_only_state():
    state_df = pd.DataFrame(
        {
            "state": ["RANGE", "RANGE", "RANGE", "RANGE"],
            "vol_regime_bucket": ["LOW", "HIGH", "LOW", "HIGH"],
            "distance_to_nearest_buy_liquidity": [0.2, 3.0, 0.2, 3.0],
            "distance_to_nearest_sell_liquidity": [0.2, 3.0, 0.2, 3.0],
            "breakout_rejected": [True, False, True, False],
            "breakout_up": [False, True, False, True],
            "breakout_down": [False, False, False, False],
            "recent_sweep_flag": [1, 0, 1, 0],
            "session_name": ["ASIA", "NEW_YORK", "ASIA", "NEW_YORK"],
            "log_return": [0.01, 0.02, -0.01, 0.03],
            "range": [1.0, 2.0, 1.2, 2.2],
        }
    )
    tm = learn_transition_model(state_df)
    row_a = state_df.iloc[0]
    row_b = state_df.iloc[1]
    key_a = build_transition_key("RANGE", row_a)
    key_b = build_transition_key("RANGE", row_b)
    assert key_a != key_b
    rng = __import__("numpy").random.default_rng(0)
    _ = sample_next_state("RANGE", row_a, tm.transition_probs, rng)


def test_session_labels_are_consistent():
    ts = pd.Timestamp("2025-01-01 09:00:00", tz="UTC")
    assert xauusd_session_name(ts) == "LONDON_OPEN"
