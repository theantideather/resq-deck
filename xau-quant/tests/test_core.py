import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aurum import calendar as cal
from aurum.agents import View, claude_strategist, risk_manager, run_desk
from aurum.backtest import BacktestConfig, run_backtest
from aurum.data import MarketData, align_to_bars, synthetic_market
from aurum.features import build_features
from aurum.instrument import XAUUSD
from aurum.ml import triple_barrier_labels, walk_forward
from aurum.regime import detect_regimes
from aurum.strategies import STRATEGIES, hysteresis
from aurum.validation import bootstrap_trades, deflated_sharpe, probabilistic_sharpe


@pytest.fixture(scope="module")
def md():
    return synthetic_market(start="2021-01-01", end="2022-12-31", seed=3)


@pytest.fixture(scope="module")
def feats(md):
    return build_features(md, fair_window=1000)


def flat_market(prices, start="2024-01-08 00:00", freq="1h"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    p = np.asarray(prices, float)
    bars = pd.DataFrame({"open": p, "high": p + 0.5, "low": p - 0.5, "close": p, "volume": 1.0}, index=idx)
    empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"), "event": pd.Series(dtype=str)})
    return MarketData(bars, pd.DataFrame(index=idx), empty, "test")


# --- instrument and calendar -------------------------------------------------

def test_lot_rounding_and_point_value():
    assert XAUUSD.round_lots(0.0149) == 0.01
    assert XAUUSD.round_lots(0.004) == 0.0
    assert XAUUSD.round_lots(1000) == XAUUSD.max_lot
    assert XAUUSD.usd_per_point(1.0) == 100.0


def test_london_session_follows_daylight_saving():
    summer = pd.DatetimeIndex(["2024-07-01 07:00"], tz="UTC")  # 08:00 BST
    winter = pd.DatetimeIndex(["2024-01-08 07:00"], tz="UTC")  # 07:00 GMT
    assert cal.session_flags(summer)["london"].iloc[0]
    assert not cal.session_flags(winter)["london"].iloc[0]


def test_nfp_is_first_friday():
    d = cal.nfp_dates(2024, 2024)
    assert all(x.weekday() == 4 and x.day <= 7 for x in d)
    assert d[0].isoformat() == "2024-01-05"


def test_event_risk_window():
    idx = pd.date_range("2024-01-05 11:00", periods=6, freq="1h", tz="UTC")
    ev = cal.event_calendar(2024, 2024)
    flag = cal.event_risk(idx, ev, "1h", "0h")  # NFP 13:30 UTC in January
    assert flag.tolist() == [False, False, True, False, False, False]  # only the 13:00 bar opens in [12:30, 13:30]


# --- data and lookahead ---------------------------------------------------------

def test_daily_series_not_visible_before_publication():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    idx = pd.date_range("2024-01-02", "2024-01-05", freq="1h", tz="UTC")
    a = align_to_bars(s, idx, "46h")
    assert a[: pd.Timestamp("2024-01-03 21:00", tz="UTC")].isna().all()
    assert a[pd.Timestamp("2024-01-03 22:00", tz="UTC")] == 1.0
    assert a[pd.Timestamp("2024-01-04 22:00", tz="UTC")] == 2.0


def test_synthetic_market_has_no_weekend_bars(md):
    ny = md.bars.index.tz_convert(cal.NEW_YORK)
    assert not (ny.weekday == 5).any()
    assert (md.bars["high"] >= md.bars[["open", "close"]].max(axis=1)).all()


def test_features_do_not_look_ahead(md):
    """Features at bar t must not change when bars after t are removed."""
    cut = len(md.bars) - 700
    short = md.slice(end=md.bars.index[cut - 1])
    full_f = build_features(md, fair_window=1000)
    short_f = build_features(short, fair_window=1000)
    common = short_f.index[-300:]
    cols = [c for c in short_f.columns if c != "cot_mm_z"]
    pd.testing.assert_frame_equal(full_f.loc[common, cols], short_f.loc[common, cols], check_exact=False, rtol=1e-9)


def test_regime_filter_is_causal(md):
    c = md.bars["close"]
    n = len(c)
    full = detect_regimes(c, train_end=4000)
    short = detect_regimes(c.iloc[: n - 500], train_end=4000)
    common = short.dropna().index[-200:]
    np.testing.assert_allclose(full.loc[common, "p_calm"], short.loc[common, "p_calm"], atol=1e-9)
    probs = full.filter(like="p_").dropna()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


# --- backtest mechanics ----------------------------------------------------------

def test_flat_signal_does_nothing(md):
    res = run_backtest(md, pd.Series(0.0, index=md.bars.index))
    assert res.stats["trades"] == 0
    assert (res.equity == res.config.initial_equity).all()


def test_single_trade_pnl_matches_hand_calculation():
    prices = [2000.0] * 3 + [2010.0] * 3  # Monday, no rollover crossed
    m = flat_market(prices)
    sig = pd.Series([1, 1, 1, 1, 0, 0], index=m.bars.index, dtype=float)
    ins = XAUUSD.with_overrides(slippage_usd=0.0)
    cfg = BacktestConfig(stop_atr=None, atr_period=2, instrument=ins, initial_equity=100_000)
    res = run_backtest(m, sig, cfg)
    t = res.trades.iloc[0]
    assert t.entry_price == 2000.0 and t.exit_price == 2010.0
    expected_gross = 10.0 * t.lots * 100
    expected_costs = t.lots * 100 * ins.spread_usd + t.lots * ins.commission_per_lot
    assert t.gross_pnl == pytest.approx(expected_gross)
    assert t.costs == pytest.approx(expected_costs)
    assert res.equity.iloc[-1] == pytest.approx(100_000 + expected_gross - expected_costs)


def test_wednesday_rollover_charges_triple_swap():
    # Wednesday 2024-01-10, hourly across 17:00 New York (22:00 UTC).
    idx_start = "2024-01-10 18:00"
    m = flat_market([2000.0] * 10, start=idx_start)
    sig = pd.Series(1.0, index=m.bars.index)
    cfg = BacktestConfig(stop_atr=None, atr_period=2)
    res = run_backtest(m, sig, cfg)
    t = res.trades.iloc[0]
    assert t.swap == pytest.approx(3 * XAUUSD.swap_long_per_lot * t.lots)


def test_stop_gap_fills_at_open():
    prices = [2000.0] * 20
    m = flat_market(prices)
    b = m.bars.copy()
    b.iloc[15, :4] = [1980.0, 1981.0, 1979.0, 1980.0]  # gap far below any 2 ATR stop
    m = MarketData(b, m.macro, m.events, "test")
    sig = pd.Series(1.0, index=b.index)
    res = run_backtest(m, sig, BacktestConfig(stop_atr=2.0, atr_period=2))
    stop = res.trades[res.trades.reason == "stop"].iloc[0]
    assert stop.exit_price == 1980.0


def test_all_strategies_run(md, feats):
    for name, strat in STRATEGIES.items():
        sig = strat.signal(md, feats)
        assert sig.between(-1, 1).all(), name
        res = run_backtest(md, sig, strat.config)
        assert np.isfinite(res.equity).all(), name


def test_hysteresis():
    z = pd.Series([0, 2, 1, 0.5, 0.1, -2, -0.1])
    assert hysteresis(z, 1.5, 0.25).tolist() == [0, -1, -1, -1, 0, 1, 0]


# --- machine learning -------------------------------------------------------------

def test_triple_barrier_labels():
    p = [100.0] * 20 + [110.0] * 10
    m = flat_market(p)
    lab = triple_barrier_labels(m.bars, horizon=5, k_atr=1.0, atr_period=3)
    assert lab.iloc[16] == 1  # jump to 110 inside the horizon
    assert lab.iloc[5] == 0   # nothing moves
    assert lab.iloc[-5:].isna().all()


def test_walk_forward_finds_planted_edge_and_not_noise(md):
    labels = triple_barrier_labels(md.bars, horizon=12, k_atr=1.0)
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({"noise": rng.standard_normal(len(md.bars))}, index=md.bars.index)
    planted = noise.assign(oracle=labels.fillna(0) + rng.standard_normal(len(md.bars)) * 0.8)
    good = walk_forward(md, planted, horizon=12, k_atr=1.0, min_train=3000, test_bars=2000)
    bad = walk_forward(md, noise, horizon=12, k_atr=1.0, min_train=3000, test_bars=2000)
    assert good.hit_rate() > 0.65
    assert 0.44 < bad.hit_rate() < 0.56


# --- validation ------------------------------------------------------------------

def test_sharpe_statistics():
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.002, 0.01, 750))
    assert probabilistic_sharpe(strong) > 0.95
    assert deflated_sharpe(strong, n_trials=200) < probabilistic_sharpe(strong)
    trades = pd.DataFrame({"net_pnl": rng.normal(10, 100, 200)})
    b = bootstrap_trades(trades, 10_000)
    assert b["max_dd_p95"] <= b["max_dd_median"] <= 0


# --- agents ------------------------------------------------------------------------

def test_risk_manager_never_enlarges_or_flips():
    rng = np.random.default_rng(2)
    for _ in range(500):
        q = float(rng.uniform(-1, 1))
        views = [View(a, float(rng.uniform(-1, 1)), float(rng.uniform(0, 1)), veto=bool(rng.random() < 0.1))
                 for a in ("macro", "technical", "strategist")]
        d = risk_manager(q, views)
        assert abs(d.final_signal) <= abs(q) + 1e-12
        assert d.final_signal == 0 or np.sign(d.final_signal) == np.sign(q)


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.calls = []
        text = json.dumps(payload)

        def create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_claude_strategist_parses_structured_view():
    pytest.importorskip("anthropic")
    payload = {"bias": "bearish", "confidence": 0.7, "thesis": "real yields rising",
               "key_drivers": ["TIPS +20bp"], "risks": ["PBoC buying"], "veto_reason": ""}
    client = FakeClient(payload)
    v = claude_strategist({"price": 2400}, ["Fed hawkish"], client=client)
    assert v.bias == -0.5 and v.confidence == 0.7 and not v.veto
    call = client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "untrusted" in call["messages"][0]["content"]


def test_claude_strategist_refusal_returns_none():
    pytest.importorskip("anthropic")
    client = FakeClient({}, stop_reason="refusal")
    assert claude_strategist({"price": 2400}, client=client) is None


def test_desk_runs_without_llm(md, feats):
    snap, d = run_desk(md, feats, 0.5, use_llm=False)
    assert snap["price"] > 0
    assert {v.analyst for v in d.views} == {"macro", "positioning", "technical", "event"}
