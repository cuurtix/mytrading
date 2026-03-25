from __future__ import annotations

import pandas as pd

from src.behavior_learning import learn_behavior
from src.data_ingestion import IngestionReport, NormalizedDataset
from src.data_learning import learn_from_report


def _mk_df(freq: str, n: int = 20):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": 100 + base,
            "high": 101 + base,
            "low": 99 + base,
            "close": 100.5 + base,
            "volume": 1000 + base,
        }
    )


def test_calibration_uses_merged_timeframe_not_first_dataset():
    d1 = NormalizedDataset("a", "csv", None, 60, "1m", {}, _mk_df("1min"))
    d2 = NormalizedDataset("b", "csv", None, 300, "5m", {}, _mk_df("5min", 40))
    d3 = NormalizedDataset("c", "csv", None, 300, "5m", {}, _mk_df("5min", 30))
    report = IngestionReport(datasets=[d1, d2, d3], logs=[])

    bundle = learn_from_report(report)
    assert bundle.timeframe_seconds == 300


def test_liquidity_features_are_incremental_no_future_leak():
    df = _mk_df("1min", 80)
    learned = learn_behavior(df)
    first = learned.feature_df.iloc[0]
    assert float(first["distance_to_nearest_buy_liquidity"]) >= 2.5
    assert int(first["recent_sweep_flag"]) in (0, 1)
