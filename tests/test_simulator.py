from __future__ import annotations

import pandas as pd
import pytest

from src.behavior_learning import learn_behavior
from src.simulator import SimulationConfig, SyntheticMarketSimulator


def _mk_df(n: int) -> pd.DataFrame:
    dt = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": 2000 + base * 0.1,
            "high": 2001 + base * 0.1,
            "low": 1999 + base * 0.1,
            "close": 2000.5 + base * 0.1,
            "volume": 1000 + base,
        }
    )


def test_simulator_alignment_handles_non_consecutive_index():
    hist = _mk_df(80)
    hist.index = list(range(1000, 1080))
    learned = learn_behavior(hist.reset_index(drop=True))
    sim = SyntheticMarketSimulator(learned, timeframe_seconds=60)
    out = sim.run(hist, SimulationConfig(seed=1, n_steps=10))
    assert len(out) == 10
    assert {"open", "high", "low", "close"}.issubset(out.columns)


def test_simulator_raises_on_too_short_bootstrap():
    hist = _mk_df(4)
    learned = learn_behavior(hist)
    sim = SyntheticMarketSimulator(learned, timeframe_seconds=60)
    with pytest.raises(ValueError, match="insuffisant"):
        sim.run(hist, SimulationConfig(seed=1, n_steps=3))
