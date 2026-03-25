from __future__ import annotations

import importlib.util
import pytest

if importlib.util.find_spec("flask") is None:
    pytest.skip("flask non installé dans l'environnement de test", allow_module_level=True)

from src.data_ingestion import IngestionReport, NormalizedDataset
from src.data_learning import learn_from_report
from src.game_engine import TradingGameEngine
import pandas as pd
import game_server


def _engine():
    dt = pd.date_range('2025-01-01', periods=120, freq='1min', tz='UTC')
    base = pd.Series(range(120), dtype=float)
    df = pd.DataFrame({'datetime': dt, 'open': 2000 + base*0.1, 'high': 2001 + base*0.1, 'low': 1999 + base*0.1, 'close': 2000.5 + base*0.1, 'volume': 1000 + base})
    ds = NormalizedDataset('x', 'csv', None, 60, '1m', {}, df)
    bundle = learn_from_report(IngestionReport(datasets=[ds], logs=[]))
    return TradingGameEngine(bundle)


def test_api_snapshot_after_actions():
    game_server.engine = _engine()
    client = game_server.app.test_client()

    for endpoint, payload in [
        ('/api/step', {}),
        ('/api/order', {'side': 'buy', 'size': 1, 'leverage': 50}),
        ('/api/close', {'fraction': 0.2}),
        ('/api/deposit', {'amount': 100}),
        ('/api/withdraw', {'amount': 10}),
        ('/api/reset', {}),
    ]:
        resp = client.post(endpoint, json=payload)
        data = resp.get_json()
        assert 'snapshot' in data
        for k in ['metrics', 'positions', 'last_price', 'state', 'timestamp', 'recent_events']:
            assert k in data['snapshot']
