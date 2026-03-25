from __future__ import annotations
import pytest

from pathlib import Path
import zipfile

import pandas as pd

from src.data_ingestion import detect_timeframe_seconds, read_zip_datasets, sanitize_ohlc


@pytest.mark.skipif(__import__("importlib").util.find_spec("openpyxl") is None, reason="openpyxl non installé")
def test_ingestion_zip_with_xlsx(tmp_path: Path):
    df = pd.DataFrame(
        {
            "timestamp": [1700000000, 1700000060, 1700000120],
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "basevolume": [10, 11, 12],
            "usdtvolume": [100, 101, 102],
        }
    )
    xlsx = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(xlsx, arcname="nested/sample.xlsx")

    report = read_zip_datasets(zpath)
    assert len(report.datasets) == 1
    ds = report.datasets[0]
    assert ds.timeframe_label == "1m"
    assert ds.mapped_columns["volume"] == "basevolume"
    assert "usdtvolume" in ds.dataframe.columns


def test_timestamp_parsing_seconds_and_text():
    s_sec = pd.Series([1700000000, 1700000060])
    tf_sec, label = detect_timeframe_seconds(pd.to_datetime(s_sec, unit="s", utc=True))
    assert tf_sec == 60
    assert label == "1m"

    s_text = pd.Series(pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"], utc=True))
    tf_sec2, label2 = detect_timeframe_seconds(s_text)
    assert tf_sec2 == 3600
    assert label2 == "1h"


def test_timeframe_mapping():
    t1 = pd.to_datetime([0, 60, 120], unit="s", utc=True)
    t5 = pd.to_datetime([0, 300, 600], unit="s", utc=True)
    t1h = pd.to_datetime([0, 3600, 7200], unit="s", utc=True)
    assert detect_timeframe_seconds(pd.Series(t1))[1] == "1m"
    assert detect_timeframe_seconds(pd.Series(t5))[1] == "5m"
    assert detect_timeframe_seconds(pd.Series(t1h))[1] == "1h"


def test_ohlc_sanitation():
    raw = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"], utc=True),
            "open": [10, 10, 10],
            "high": [9, 11, 12],
            "low": [8, 9, 9],
            "close": [10, 10.5, 11],
            "volume": [1, 2, 3],
        }
    )
    clean, removed = sanitize_ohlc(raw)
    assert len(clean) == 2
    assert removed == 1
    assert clean["datetime"].is_monotonic_increasing
    assert clean["datetime"].nunique() == len(clean)
