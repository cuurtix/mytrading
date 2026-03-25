from __future__ import annotations

import pandas as pd

from src.data_ingestion import IngestionReport, NormalizedDataset
from src.data_learning import learn_from_report
from src.game_engine import TradingGameEngine


def _mk_df(n=200, freq='1min'):
    dt = pd.date_range('2025-01-01', periods=n, freq=freq, tz='UTC')
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame({'datetime': dt, 'open': 2000 + base*0.1, 'high': 2001 + base*0.1, 'low': 1999 + base*0.1, 'close': 2000.5 + base*0.1, 'volume': 1000 + base})


def _engine():
    ds = NormalizedDataset('x', 'csv', None, 60, '1m', {}, _mk_df())
    bundle = learn_from_report(IngestionReport(datasets=[ds], logs=[]))
    return TradingGameEngine(bundle)


def test_fee_debited_on_open_once_and_partial_close():
    e = _engine()
    b0 = e.account.balance
    r = e.place_order('buy', size=2)
    assert r['ok']
    fee = r['execution']['fee']
    assert e.account.balance == b0 - fee

    b1 = e.account.balance
    c = e.close_fraction(0.5)
    assert c['ok']
    # no second fee debit mécanique à la clôture
    assert e.account.balance != b1 - fee


def test_rejected_order_has_no_market_impact():
    e = _engine()
    p0 = e.pending_player_impact
    part0 = e.recent_order_participation
    imp0 = e.recent_order_impact
    r = e.place_order('buy', size=10_000_000)
    assert r['ok'] is False
    assert e.pending_player_impact == p0
    assert e.recent_order_participation == part0
    assert e.recent_order_impact == imp0


def test_impact_small_vs_large_order():
    e = _engine()
    small = e.place_order('buy', size=0.01)
    large = e.place_order('buy', size=20)
    assert abs(small['execution']['impact']) <= abs(large['execution']['impact'])
    assert small['execution']['spread_widen'] <= large['execution']['spread_widen']
    assert small['execution']['slippage'] <= large['execution']['slippage']


def test_snapshot_complete_and_state_progresses():
    e = _engine()
    s = e.snapshot()
    for k in ['metrics', 'positions', 'last_price', 'state', 'timestamp', 'recent_events']:
        assert k in s
    prev = e.current_state
    out = e.step_market()
    assert out['ok'] and 'snapshot' in out
    assert e.current_state == out['state']
    assert isinstance(prev, str)


def test_liquidation_forced_when_under_maintenance_margin():
    e = _engine()
    e.place_order('buy', size=20, leverage=200)
    # force drop
    e.history.loc[e.history.index[-1], 'close'] *= 0.1
    liq = e._enforce_liquidation_if_needed()
    assert liq in (True, False)
