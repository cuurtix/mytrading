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
    r1 = e.place_order('buy', size=2)
    r2 = e.place_order('sell', size=1)
    assert r1['ok'] and r2['ok']
    assert len(e.account.positions) == 2

    c = e.close_fraction(0.5)
    assert c['ok']
    assert len(e.account.positions) == 2
    assert e.account.positions[0].size > 0

    e.close_all()
    assert len(e.account.positions) == 0


def test_deposit_withdraw_and_step_and_state_progresses():
    e = _engine()
    e.deposit(500)
    out_w = e.withdraw(100)
    assert out_w['ok'] is True
    s0 = e.current_state
    out = e.step_market()
    assert out['ok'] and 'snapshot' in out
    assert e.current_state == out['state']
    assert isinstance(s0, str)


def test_player_order_impact_and_margin_validation():
    e = _engine()
    ex = e.place_order('buy', size=20)
    assert ex['execution']['impact'] != 0

    too_big = e.place_order('buy', size=10_000_000)
    assert too_big['ok'] is False
    assert 'Marge insuffisante' in too_big['reason']
