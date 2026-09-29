from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aurum.execution import PaperBroker, ReplayFeed
from aurum.execution.mt5 import MAGIC, MT5Broker
from aurum.execution.oanda import OandaBroker
from aurum.instrument import XAUUSD


def bars_from(prices, start="2024-01-08 00:00", freq="1h"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 1.0}, index=idx)


# --- paper broker ---------------------------------------------------------------------

def test_paper_round_trip_costs_and_persistence(tmp_path):
    bars = bars_from([2000.0] * 3 + [2010.0] * 3)
    feed = ReplayFeed(bars, 3)
    b = PaperBroker(feed, tmp_path / "acct.json")
    b.sync(feed(10))
    r = b.open(1, 0.5, stop=1990.0, target=None)
    assert r.ok and r.price == 2000.0
    feed.advance(3)
    b.sync(feed(10))
    assert b.account().equity > 100_000
    c = b.close()
    assert c.ok and c.price == 2010.0
    ins = XAUUSD
    per_side = 0.5 * 100 * (ins.spread_usd / 2 + ins.slippage_usd) + 0.5 * ins.commission_per_lot / 2
    expected = 100_000 + 10 * 0.5 * 100 - 2 * per_side
    assert b.account().balance == pytest.approx(expected)
    again = PaperBroker(feed, tmp_path / "acct.json")  # reload from disk
    assert again.account().balance == pytest.approx(expected)
    assert len(again.trades()) == 1 and again.position() is None


def test_paper_stop_gap_fills_at_open(tmp_path):
    bars = bars_from([2000.0] * 5)
    bars.iloc[4, :4] = [1980.0, 1981.0, 1979.0, 1980.0]
    feed = ReplayFeed(bars, 3)
    b = PaperBroker(feed, tmp_path / "a.json")
    b.sync(feed(10))
    b.open(1, 0.1, stop=1995.0, target=None)
    feed.advance(2)
    ev = b.sync(feed(10))
    assert any(e["event"] == "stop" for e in ev)
    assert b.trades()[-1]["exit_price"] == 1980.0 and b.position() is None


def test_paper_wednesday_triple_swap(tmp_path):
    bars = bars_from([2000.0] * 10, start="2024-01-10 18:00")  # Wed, crosses 17:00 NY (22:00 UTC)
    feed = ReplayFeed(bars, 1)
    b = PaperBroker(feed, tmp_path / "a.json")
    b.sync(feed(10))
    b.open(1, 1.0, stop=None, target=None)
    feed.advance(9)
    ev = b.sync(feed(20))
    swaps = [e for e in ev if e["event"] == "swap"]
    assert len(swaps) == 1 and swaps[0]["usd"] == pytest.approx(3 * XAUUSD.swap_long_per_lot)


def test_paper_refuses_double_open(tmp_path):
    feed = ReplayFeed(bars_from([2000.0] * 3), 3)
    b = PaperBroker(feed, tmp_path / "a.json")
    b.sync(feed(10))
    assert b.open(1, 0.1, None, None).ok
    assert not b.open(-1, 0.1, None, None).ok


# --- OANDA -------------------------------------------------------------------------------

class FakeOanda:
    def __init__(self):
        self.calls = []
        self.trades = []

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if "/candles" in path:
            t = pd.date_range("2026-01-05", periods=4, freq="1h", tz="UTC")
            return {"candles": [{"complete": i < 3, "volume": 10, "time": ts.isoformat().replace("+00:00", "Z"),
                                 "mid": {"o": "2400.0", "h": "2401.5", "l": "2399.0", "c": "2400.8"}}
                                for i, ts in enumerate(t)]}
        if path.endswith("/summary"):
            return {"account": {"NAV": "100250.5", "balance": "100000", "marginUsed": "1200", "currency": "USD"}}
        if path.endswith("/openTrades"):
            return {"trades": self.trades}
        if path.endswith("/orders"):
            units = body["order"]["units"]
            self.trades = [{"id": "77", "instrument": "XAU_USD", "currentUnits": units, "price": "2400.9",
                            "openTime": "2026-01-05T04:00:00Z", "stopLossOrder": {"price": body["order"]["stopLossOnFill"]["price"]}}]
            return {"orderFillTransaction": {"id": "76", "price": "2400.9", "tradeOpened": {"tradeID": "77"}}}
        if "/close" in path:
            self.trades = []
            return {"longOrderFillTransaction": {"id": "80", "price": "2410.1"}}
        raise AssertionError(path)


def test_oanda_adapter_round_trip():
    f = FakeOanda()
    b = OandaBroker(token="t", account_id="101-1", transport=f)
    c = b.candles(10)
    assert len(c) == 3 and c.index.tz is not None  # incomplete bar dropped
    assert b.account().equity == 100250.5
    r = b.open(1, 0.25, stop=2390.0, target=None)
    order = f.calls[-1][2]["order"]
    assert order["units"] == "25" and order["stopLossOnFill"]["price"] == "2390.00"
    assert r.ok and r.broker_id == "77"
    p = b.position()
    assert p.side == 1 and p.lots == 0.25 and p.stop == 2390.0
    assert b.close().ok and f.calls[-1][2] == {"longUnits": "ALL"}
    assert b.position() is None


def test_oanda_rejects_bad_env():
    with pytest.raises(ValueError):
        OandaBroker(token="t", account_id="a", env="prod")


# --- MT5 ------------------------------------------------------------------------------

def fake_mt5(trade_mode=0):
    state = {"positions": [], "sent": []}
    m = SimpleNamespace(
        TIMEFRAME_H1=16385, POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1, ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
        TRADE_ACTION_DEAL=1, ORDER_TIME_GTC=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2,
        TRADE_RETCODE_DONE=10009, state=state,
    )
    m.initialize = lambda **kw: True
    m.symbol_select = lambda s, on: True
    m.last_error = lambda: (0, "ok")
    m.account_info = lambda: SimpleNamespace(equity=50_000.0, balance=49_900.0, margin=500.0, currency="USD", trade_mode=trade_mode)
    m.symbol_info = lambda s: SimpleNamespace(filling_mode=2)
    m.symbol_info_tick = lambda s: SimpleNamespace(bid=2400.0, ask=2400.3)
    t0 = int(pd.Timestamp("2026-01-05 02:00", tz="UTC").timestamp())
    m.copy_rates_from_pos = lambda s, tf, start, n: np.array(
        [(t0 + 3600 * i, 2400.0, 2401.0, 2399.0, 2400.5, 100) for i in range(5)],
        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")])

    def order_send(req):
        state["sent"].append(req)
        if "position" in req:
            state["positions"] = []
        else:
            state["positions"] = [SimpleNamespace(ticket=555, type=req["type"], volume=req["volume"], price_open=req["price"],
                                                  sl=req["sl"], tp=req["tp"], time=t0, magic=MAGIC)]
        return SimpleNamespace(retcode=10009, price=req["price"], order=555, comment="done")

    m.order_send = order_send
    m.positions_get = lambda symbol=None: state["positions"]
    return m


def test_mt5_adapter_with_fake_terminal():
    m = fake_mt5()
    b = MT5Broker(server_utc_offset=2, module=m)
    assert not b.live
    c = b.candles(5)
    assert c.index[0] == pd.Timestamp("2026-01-05 00:00", tz="UTC")  # server time shifted to UTC
    r = b.open(-1, 0.3, stop=2410.0, target=2380.0)
    sent = m.state["sent"][-1]
    assert r.ok and sent["type"] == m.ORDER_TYPE_SELL and sent["price"] == 2400.0 and sent["type_filling"] == m.ORDER_FILLING_IOC
    p = b.position()
    assert p.side == -1 and p.lots == 0.3 and p.stop == 2410.0
    assert b.close().ok and m.state["sent"][-1]["type"] == m.ORDER_TYPE_BUY
    assert b.position() is None


def test_mt5_real_account_is_live():
    assert MT5Broker(module=fake_mt5(trade_mode=2)).live
