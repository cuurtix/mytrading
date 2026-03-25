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
from unittest.mock import patch


def _engine():
    dt = pd.date_range('2025-01-01', periods=120, freq='1min', tz='UTC')
    base = pd.Series(range(120), dtype=float)
    df = pd.DataFrame({'datetime': dt, 'open': 2000 + base*0.1, 'high': 2001 + base*0.1, 'low': 1999 + base*0.1, 'close': 2000.5 + base*0.1, 'volume': 1000 + base})
    ds = NormalizedDataset('x', 'csv', None, 60, '1m', {}, df)
    bundle = learn_from_report(IngestionReport(datasets=[ds], logs=[]))
    return TradingGameEngine(bundle)


def test_api_snapshot_after_actions():
    game_server.engine = _engine()
    game_server.server_state.update({"status": "ready", "phase": "ready", "timeframe": "1m"})
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


def test_api_init_returns_initial_candles_and_snapshot_keys():
    game_server.engine = _engine()
    game_server.server_state.update({"status": "ready", "phase": "ready", "timeframe": "1m"})
    client = game_server.app.test_client()
    resp = client.post('/api/init', json={})
    data = resp.get_json()
    assert data["ok"] is True
    assert "initial_candles" in data
    assert len(data["initial_candles"]) > 0
    assert "snapshot" in data
    for k in ['metrics', 'positions', 'last_price', 'state', 'timestamp', 'recent_events']:
        assert k in data
        assert k in data["snapshot"]


def test_api_init_fallback_when_no_dataset():
    game_server.engine = None
    client = game_server.app.test_client()
    with patch("game_server.scan_data_sources", side_effect=ValueError("no data")):
        game_server._initialize_runtime()
        data = client.post('/api/init', json={}).get_json()
    assert data["ok"] is True
    assert data["timeframe"] == "1m"
    assert len(data["initial_candles"]) > 0


def test_health_and_init_status_endpoints():
    game_server.engine = _engine()
    game_server.server_state.update({"status": "ready", "phase": "ready", "timeframe": "1m"})
    client = game_server.app.test_client()
    h = client.get('/api/health').get_json()
    s = client.get('/api/init_status').get_json()
    assert h["ok"] is True
    assert s["ok"] is True
    assert s["status"] == "ready"
