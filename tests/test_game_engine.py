from __future__ import annotations

import pandas as pd

from src.data_ingestion import IngestionReport, NormalizedDataset
from src.data_learning import learn_from_report
from src.game_engine import TradingGameEngine


def _mk_df(n=200, freq='1min'):
    dt = pd.date_range('2025-01-01', periods=n, freq=freq, tz='UTC')
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame({
        'datetime': dt,
        'open': 2000 + base*0.1,
        'high': 2001 + base*0.1,
        'low': 1999 + base*0.1,
        'close': 2000.5 + base*0.1,
        'volume': 1000 + base,
    })


def _engine():
    ds = NormalizedDataset('x', 'csv', None, 60, '1m', {}, _mk_df())
    bundle = learn_from_report(IngestionReport(datasets=[ds], logs=[]))
    return TradingGameEngine(bundle)


def test_buy_sell_close_partial_and_all():
    e = _engine()
    e.place_order('buy', size=2)
    e.place_order('sell', size=1)
    assert len(e.account.positions) == 2

    e.close_fraction(0.5)
    assert len(e.account.positions) == 2
    assert e.account.positions[0].size > 0

    e.close_all()
    assert len(e.account.positions) == 0


def test_deposit_withdraw_and_step():
    e = _engine()
    e.deposit(500)
    ok = e.withdraw(100)
    assert ok is True
    out = e.step_market()
    assert 'metrics' in out and 'candle' in out


def test_player_order_impact_is_recorded():
    e = _engine()
    ex = e.place_order('buy', size=5000)
    assert ex['impact'] != 0
