import contextlib
import io

import pandas as pd
import pytest

from aurum.backtest import run_backtest
from aurum.data import synthetic_market
from aurum.execution import PaperBroker, ReplayFeed
from aurum.execution.base import Account
from aurum.features import build_features
from aurum.notify import Notifier
from aurum.runner import LIVE_ACK, LiveTradingBlocked, Runner, RunnerConfig, read_journal
from aurum.strategies import STRATEGIES


class Recorder(Notifier):
    def __init__(self):
        super().__init__("", "", "")
        self.msgs = []

    def send(self, text):
        self.msgs.append(text)
        return True


@pytest.fixture(scope="module")
def md():
    return synthetic_market(start="2023-01-01", end="2023-12-31", seed=11)


def make(md, tmp_path, start=2500, **cfg_kw):
    feed = ReplayFeed(md.bars, start)
    broker = PaperBroker(feed, tmp_path / "paper.json")
    clock = {"t": md.bars.index[feed.pos - 1] + pd.Timedelta(minutes=61)}
    cfg = RunnerConfig(**{"strategy": "london_breakout", "bars": 1500, "with_macro": False,
                          "journal_path": tmp_path / "j.jsonl", "state_path": tmp_path / "s.json", **cfg_kw})
    rec = Recorder()
    runner = Runner(broker, cfg, notifier=rec, now=lambda: clock["t"])

    def step(n=1):
        out = None
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(n):
                out = runner.cycle()
                feed.advance()
                clock["t"] = md.bars.index[feed.pos - 1] + pd.Timedelta(minutes=61)
        return out

    return runner, broker, feed, clock, rec, step


def test_live_runner_matches_backtest_trade_for_trade(md, tmp_path):
    """Same bars, same strategy: the live path must make the same trades as the backtester."""
    start, n = 2500, 400
    runner, broker, *_, step = make(md, tmp_path, start=start, use_desk=False)
    step(n)
    live = [(pd.Timestamp(t["opened_utc"]), t["side"], t["reason"]) for t in broker.trades()]
    sub = md.slice(md.bars.index[start - 1500], md.bars.index[start + n - 1])
    strat = STRATEGIES["london_breakout"]
    res = run_backtest(sub, strat.signal(sub, build_features(sub, fair_window=500)), strat.config)
    bt = res.trades[res.trades.entry_time >= md.bars.index[start]]
    bt = bt[bt.reason != "end"]
    expected = [(t - pd.Timedelta(hours=1), int(s), r) for t, s, r in zip(bt.entry_time, bt.side, bt.reason)]
    assert len(live) >= 8
    assert live == expected[: len(live)]


def test_journal_and_notifications(md, tmp_path):
    runner, broker, *_, rec, step = make(md, tmp_path)
    step(150)
    j = read_journal(tmp_path / "j.jsonl")
    assert len(j) == 150 and {"quant_signal", "desk_signal", "equity"} <= set(j[-1])
    assert any(m.startswith("aurum [paper] OPEN") for m in rec.msgs)


def test_stale_data_is_skipped(md, tmp_path):
    runner, broker, feed, clock, rec, step = make(md, tmp_path)
    clock["t"] += pd.Timedelta(hours=10)
    out = runner.cycle()
    assert out["action"] == "skip" and "old" in out["notes"][0]


def test_daily_loss_limit_and_kill_switch(md, tmp_path, monkeypatch):
    runner, broker, feed, clock, rec, step = make(md, tmp_path)
    step(1)
    equity = {"v": 100_000.0}
    monkeypatch.setattr(broker, "account", lambda: Account(equity["v"], equity["v"]))
    equity["v"] = 96_000.0
    out = step(1)
    assert "daily loss limit active" in out["notes"] and broker.position() is None
    equity["v"] = 79_000.0
    out = step(1)
    assert runner.state["killed"] and "kill switch active" in out["notes"]
    assert any("kill switch" in m for m in rec.msgs)


class FakeLive(PaperBroker):
    live = True


def test_live_money_is_blocked_without_ack(md, tmp_path, monkeypatch):
    feed = ReplayFeed(md.bars, 2000)
    broker = FakeLive(feed, tmp_path / "p.json")
    cfg = RunnerConfig(journal_path=tmp_path / "j", state_path=tmp_path / "s")
    with pytest.raises(LiveTradingBlocked):
        Runner(broker, cfg, notifier=Recorder())
    cfg.allow_live = True
    with pytest.raises(LiveTradingBlocked):
        Runner(broker, cfg, notifier=Recorder())
    monkeypatch.setenv("AURUM_LIVE_ACK", LIVE_ACK)
    Runner(broker, cfg, notifier=Recorder())  # both present: allowed


def test_notifier_formats():
    sent = []
    n = Notifier("tok", "42", "https://hooks.example/x", sender=lambda url, p: sent.append((url, p)))
    assert n.send("hello")
    assert sent[0][0].endswith("/bottok/sendMessage") and sent[0][1]["chat_id"] == "42"
    assert sent[1][1] == {"content": "hello", "text": "hello"}
    assert not Notifier("", "", "").send("x")
