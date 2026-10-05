import json

import numpy as np
import pandas as pd
import pytest

from aurum.backtest import BacktestConfig, run_backtest
from aurum.data import MarketData, synthetic_market
from aurum.indicators import ENGINES, engine, flips, side_win_rates
from aurum.strategies import STRATEGIES, run_strategy
from aurum.swing import SwingParams, backtest_swing, swing_stops, to_daily, trade_plan, swing_config


@pytest.fixture(scope="module")
def md():
    return synthetic_market(start="2020-01-01", end="2023-12-31", seed=12)


def test_engines_give_direction_and_line(md):
    d = to_daily(md.bars)
    for name in ENGINES:
        e = engine(d, name)
        assert set(np.unique(e["dir"])) <= {-1.0, 1.0}
        assert e["line"].iloc[50:].notna().all(), name
        f = flips(e["dir"])
        assert set(np.unique(f)) <= {-1.0, 0.0, 1.0}


def test_engines_are_causal(md):
    d = to_daily(md.bars)
    for name in ENGINES:
        full = engine(d, name)
        short = engine(d.iloc[:-60], name)
        pd.testing.assert_frame_equal(full.iloc[:-60], short, check_exact=False, rtol=1e-9)


def test_daily_bars_roll_at_new_york_close():
    idx = pd.date_range("2024-01-08 20:00", periods=4, freq="1h", tz="UTC")  # 15:00-18:00 New York
    bars = pd.DataFrame({"open": [1, 2, 3, 4.0], "high": [1, 2, 3, 4.0], "low": [1, 2, 3, 4.0], "close": [1, 2, 3, 4.0]}, index=idx)
    d = to_daily(bars)
    assert list(d["close"]) == [2.0, 4.0]  # 17:00 New York starts the next trading day
    assert len(to_daily(bars, now=pd.Timestamp("2024-01-08 23:30", tz="UTC"))) == 1  # forming day dropped


def test_trailing_stop_ratchets_and_exits():
    idx = pd.date_range("2024-01-01", periods=8, freq="1D", tz="UTC")
    p = np.array([100, 100, 102, 105, 108, 104, 100, 99.0])
    bars = pd.DataFrame({"open": p, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 1.0}, index=idx)
    events = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"), "event": pd.Series(dtype=str)})
    m = MarketData(bars, pd.DataFrame(index=idx), events, "t")
    sig = pd.Series(1.0, index=idx)
    line = pd.Series([95, 96, 98, 101, 104, 104, 104, 104.0], index=idx)
    res = run_backtest(m, sig, BacktestConfig(stop_atr=None, atr_period=2, daily_loss_limit=1, max_drawdown_kill=1),
                       stop_line=line)
    t = res.trades.iloc[0]
    assert t.reason == "stop" and t.exit_price == 104.0  # trailed up to 104 and stopped there


def test_pct_r_target_is_seven_r(md):
    res, _ = backtest_swing(md, SwingParams("halftrend", {"amplitude": 5}, None), "pct_r")
    t = res.trades
    assert len(t) > 3
    r = (t["target"] - t["entry_price"]).abs() / (t["entry_price"] - t["initial_stop"]).abs()
    np.testing.assert_allclose(r, 7.0, rtol=1e-6)
    np.testing.assert_allclose((t["entry_price"] - t["initial_stop"]).abs() / t["entry_price"], 0.04, rtol=1e-6)


def test_structure_stop_sits_beyond_swing_extreme(md):
    d = to_daily(md.bars)
    st = swing_stops(d)
    assert (st["long"].dropna() < d["low"].rolling(10).min().dropna()).all()
    assert (st["short"].dropna() > d["high"].rolling(10).max().dropna()).all()
    cfg, _ = swing_config("swing_r")
    plan = trade_plan(d, -1, 100_000, cfg, stops=st)
    assert plan["stop"] > plan["entry"] > plan["target"]
    assert plan["risk_reward"] == 7.0


def test_swing_strategies_run_through_registry(md):
    for name, strat in STRATEGIES.items():
        if strat.swing is None:
            continue
        res, sig, m, line = run_strategy(md, strat)
        assert len(m.bars) < len(md.bars) and np.isfinite(res.equity).all(), name
        wr = side_win_rates(res.trades)
        assert set(wr) == {"long", "short"}


def test_swing_chart_payload_is_json_safe():
    from aurum import service

    r = service.swing_chart("swing_halftrend_structure", source="synthetic", start="2020-01-01", end="2023-12-31", seed=12)
    json.dumps(r, allow_nan=False)
    assert r["candles"] and r["trail"] and r["markers"]
    assert set(r["table"]) >= {"timeframes", "aligned_up", "aligned_down", "long", "short"}
    assert r["plan"]["stop"] and r["plan"]["entry"]


def test_pine_files_are_consistent():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "tradingview"
    for f in ("aurum_swing.pine", "aurum_swing_indicator.pine"):
        src = (root / f).read_text()
        assert src.count("//@version=6") == 1
        assert src.count("(") == src.count(")") and src.count("[") == src.count("]")
        assert "\t" not in src
        assert "Swing + 0.5 ATR, 7R target" in src and "fHalfTrend" in src
        assert "nz(highPrice, high)" in src  # na on the first bars would freeze HalfTrend forever
    ind = (root / "aurum_swing_indicator.pine").read_text()
    assert "indicator(" in ind and "strategy." not in ind


def test_breakout_pine_is_well_formed():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "tradingview" / "aurum_gold_breakout.pine").read_text()
    assert src.startswith("//@version=6") and src.count("//@version") == 1
    assert src.count("indicator(") == 1 and "strategy." not in src
    assert src.count("(") == src.count(")") and src.count("[") == src.count("]")
    assert "\t" not in src
