"""JSON friendly entry points shared by the web UI and the MCP server.

Everything here returns plain dicts and lists (no pandas), so it can go
straight into an HTTP response or an MCP tool result.
"""

from __future__ import annotations

import math
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as cal
from .backtest import BacktestConfig, run_backtest
from .data import MarketData, load_market
from .features import build_features
from .regime import detect_regimes
from .strategies import STRATEGIES, ensemble, run_strategy

def _find_pine() -> Path:
    """The Pine file lives in the repo's tradingview/ folder, not inside the package."""
    candidates = [os.environ.get("AURUM_PINE_PATH", ""),
                  Path(__file__).resolve().parent.parent / "tradingview" / "aurum_gold.pine",
                  Path.cwd() / "tradingview" / "aurum_gold.pine"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return Path(candidates[1])


PINE_PATH = _find_pine()
_lock = threading.Lock()


def _clean(x):
    """Make a value JSON safe: NaN and inf become None, numpy scalars become Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


@lru_cache(maxsize=8)
def _market(source: str, start: str, end: str, seed: int, csv_path: str | None) -> tuple[MarketData, pd.DataFrame]:
    kw = {"start": start, "end": end, "seed": seed} if source == "synthetic" else {}
    md = load_market(source, csv_path=csv_path, **kw)
    return md, build_features(md)


def market(source: str = "synthetic", start: str = "2019-01-01", end: str = "2024-12-31",
           seed: int = 7, csv_path: str | None = None) -> tuple[MarketData, pd.DataFrame]:
    with _lock:
        return _market(source, start, end, seed, csv_path)


def market_from_bars(rows: list[dict]) -> tuple[MarketData, pd.DataFrame]:
    """Build a market from OHLCV rows, e.g. the output of TradingView MCP's data_get_ohlcv.

    Each row needs a time (unix seconds or ms, or ISO string) and open, high,
    low, close; volume is optional. No macro data comes with it, so the macro
    features and macro reversion are empty; price, session and event features work.
    """
    df = pd.DataFrame(rows)
    df.columns = [str(c).lower() for c in df.columns]
    tcol = next(c for c in ("time", "timestamp", "datetime", "date", "t") if c in df.columns)
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.max() > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    for a, b in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        if b not in df.columns and a in df.columns:
            df[b] = df[a]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    bars = df[["open", "high", "low", "close", "volume"]].astype(float)
    bars.index = pd.DatetimeIndex(idx)
    bars = bars[~bars.index.duplicated()].sort_index()
    events = cal.event_calendar(bars.index[0].year, bars.index[-1].year + 1)
    md = MarketData(bars, pd.DataFrame(index=bars.index), events, source="tradingview")
    return md, build_features(md, fair_window=min(2000, max(200, len(bars) // 3)))


def _downsample_candles(bars: pd.DataFrame, max_points: int = 2500) -> tuple[list[dict], str]:
    rules = ["1h", "4h", "1D", "1W"]
    for rule in rules:
        if rule == "1h":
            b = bars
        else:
            b = bars.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if len(b) <= max_points:
            break
    out = [{"time": int(ts.timestamp()), "open": round(r.open, 2), "high": round(r.high, 2),
            "low": round(r.low, 2), "close": round(r.close, 2)} for ts, r in zip(b.index, b.itertuples())]
    return out, rule


def strategies() -> list[dict]:
    return [{"name": s.name, "description": s.description, "timeframe": s.timeframe,
             "stop_atr": s.config.stop_atr, "target_atr": s.config.target_atr, "stop_pct": s.config.stop_pct,
             "target_r": s.config.target_r, "trailing": s.stop_line is not None,
             "risk_per_trade": s.config.risk_per_trade} for s in STRATEGIES.values()]


def compare(**mkw) -> dict:
    md, feats = market(**mkw)
    rows = []
    for name, strat in STRATEGIES.items():
        st = run_strategy(md, strat, feats)[0].stats
        rows.append({"strategy": name, **{k: st[k] for k in (
            "total_return", "cagr", "sharpe", "max_drawdown", "trades", "win_rate", "profit_factor", "swap", "costs")}})
    return _clean({"source": md.source, "start": md.bars.index[0], "end": md.bars.index[-1],
                   "bars": len(md.bars), "results": rows})


def backtest(strategy: str = "ensemble", n_trials: int = 1, validate: bool = True,
             risk_per_trade: float | None = None, **mkw) -> dict:
    from .validation import scorecard

    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")
    md, feats = market(**mkw)
    strat = STRATEGIES[strategy]
    cfg = strat.config
    if risk_per_trade:
        cfg = BacktestConfig(**{**cfg.__dict__, "risk_per_trade": risk_per_trade})
    res, sig, m, line = run_strategy(md, strat, feats, cfg)
    eq = res.equity.resample("1D").last().dropna()
    dd = eq / eq.cummax() - 1
    candles, tf = _downsample_candles(m.bars)
    t = res.trades.copy()
    trades = [{
        "entry_time": int(r.entry_time.timestamp()), "exit_time": int(r.exit_time.timestamp()),
        "side": int(r.side), "lots": r.lots, "entry": r.entry_price, "exit": r.exit_price,
        "reason": r.reason, "pnl": r.net_pnl, "r": r.r_multiple,
    } for r in t.itertuples()]
    out = {
        "strategy": strategy, "description": strat.description, "source": md.source,
        "stats": res.stats,
        "equity": [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in eq.items()],
        "drawdown": [{"time": int(ts.timestamp()), "value": round(v * 100, 3)} for ts, v in dd.items()],
        "candles": candles, "candle_timeframe": tf,
        "trades": trades,
    }
    if validate:
        card = scorecard(m, sig, res, n_trials=n_trials, stop_line=line)
        out["validation"] = {
            "psr": card.psr, "dsr": card.dsr, "n_trials": n_trials, "bootstrap": card.bootstrap,
            "cost_stress": [{"cost_x": float(k), **v} for k, v in card.cost_table.to_dict("index").items()]
            if card.cost_table is not None else [],
            "verdict": card.verdict,
        }
    return _clean(out)


def brief(use_llm: bool = True, headlines: list[str] | None = None,
          md_feats: tuple[MarketData, pd.DataFrame] | None = None, **mkw) -> dict:
    from .agents import run_desk

    md, feats = md_feats or market(**mkw)
    regimes = detect_regimes(md.bars["close"], train_end=min(len(md.bars) // 2, 20000))
    comps = {n: STRATEGIES[n].signal(md, feats) for n in ("london_breakout", "trend", "macro_reversion")}
    ens = ensemble(md, feats, regimes, comps)
    snap, decision = run_desk(md, feats, float(ens.iloc[-1]), regimes, {**comps, "ensemble": ens},
                              headlines, use_llm=use_llm)
    return _clean({"snapshot": snap, "decision": decision.to_dict()})


def analyze_bars(rows: list[dict], use_llm: bool = False, headlines: list[str] | None = None) -> dict:
    """Run the desk on bars you pass in (for example from TradingView)."""
    md, feats = market_from_bars(rows)
    if len(md.bars) < 300:
        raise ValueError(f"need at least 300 bars, got {len(md.bars)}")
    return brief(use_llm=use_llm, headlines=headlines, md_feats=(md, feats))


def pine_source() -> str:
    if not PINE_PATH.is_file():
        raise FileNotFoundError("tradingview/aurum_gold.pine not found; run from the xau-quant folder "
                                "or set AURUM_PINE_PATH")
    return PINE_PATH.read_text()


# TradingView webhook alerts, appended as JSON lines so the web server and
# the MCP server (separate processes) see the same log.
ALERT_LOG = Path(os.environ.get("AURUM_ALERT_LOG",
                               Path(os.environ.get("AURUM_HOME", Path.home() / ".aurum")) / "alerts.jsonl"))


def record_alert(payload: dict) -> dict:
    import json

    payload = {**payload, "received_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, ALERT_LOG.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return payload


def recent_alerts(limit: int = 50) -> list[dict]:
    import json

    if not ALERT_LOG.exists():
        return []
    lines = ALERT_LOG.read_text().splitlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Paper trading, research, news, review
# ---------------------------------------------------------------------------

def _home() -> Path:
    return Path(os.environ.get("AURUM_HOME", Path.home() / ".aurum"))


class _SyntheticReplay:
    """Offline feed for demos: walks the synthetic market one bar per cycle, position kept on disk."""

    def __init__(self):
        from .data import synthetic_market

        self.md = synthetic_market(start="2023-01-01", end="2024-12-31", seed=21)
        self.path = _home() / "replay_pos.json"
        self.pos = 3000
        if self.path.exists():
            import json
            self.pos = json.loads(self.path.read_text()).get("pos", 3000)

    def advance(self) -> None:
        import json
        self.pos = min(self.pos + 1, len(self.md.bars))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"pos": self.pos}))

    def __call__(self, count: int) -> pd.DataFrame:
        return self.md.bars.iloc[max(0, self.pos - count): self.pos]

    def now(self) -> pd.Timestamp:
        return self.md.bars.index[self.pos - 1] + pd.Timedelta(minutes=61)


_replay: _SyntheticReplay | None = None


def paper_feed(kind: str = "yahoo", csv_path: str | None = None):
    """Bars for the paper broker: yahoo (GC=F hourly), csv, oanda (real prices, no orders) or replay (offline demo)."""
    global _replay
    if kind == "replay":
        if _replay is None:
            _replay = _SyntheticReplay()
        return _replay
    if kind == "yahoo":
        from .data import YAHOO, load_yahoo
        return lambda count: load_yahoo(YAHOO["gold"], "1h", "729d").tail(count)
    if kind == "csv":
        from .data import load_csv
        if not csv_path:
            raise ValueError("csv feed needs csv_path")
        return lambda count: load_csv(csv_path).tail(count)
    if kind == "oanda":
        from .execution.oanda import OandaBroker
        ob = OandaBroker()
        return lambda count: ob.candles(count)
    raise ValueError("feed must be yahoo, csv, oanda or replay")


def paper_broker(feed: str = "yahoo", csv_path: str | None = None):
    from .execution import PaperBroker
    return PaperBroker(paper_feed(feed, csv_path), _home() / "paper_account.json")


def paper_status(journal_limit: int = 30) -> dict:
    import json
    from .runner import read_journal

    path = _home() / "paper_account.json"
    state = json.loads(path.read_text()) if path.exists() else None
    out = {"exists": state is not None, "journal": read_journal(_home() / "journal.jsonl", journal_limit)}
    if state:
        pos = state.get("position")
        price = state.get("last_price")
        open_pnl = pos["side"] * (price - pos["entry_price"]) * pos["lots"] * 100 if pos and price else 0.0
        out.update({"balance": state["balance"], "initial": state.get("initial"), "equity": state["balance"] + open_pnl,
                    "open_pnl": open_pnl, "position": pos, "last_price": price, "last_bar": state.get("last_bar"),
                    "trades": state["trades"][-50:], "n_trades": len(state["trades"])})
    return _clean(out)


def paper_cycle(strategy: str = "ensemble", feed: str = "replay", csv_path: str | None = None,
                use_desk: bool = True, use_llm: bool = False) -> dict:
    """One runner cycle against the paper account."""
    from .runner import Runner, RunnerConfig

    broker = paper_broker(feed, csv_path)
    kw = {}
    if feed == "replay":
        kw["now"] = broker.feed.now
    cfg = RunnerConfig(strategy=strategy, bars=3000, with_macro=feed != "replay", use_desk=use_desk,
                       use_llm=use_llm, journal_path=_home() / "journal.jsonl", state_path=_home() / "runner_state.json")
    headlines = None
    if use_llm:
        from .news import fetch_headlines, headline_texts
        headlines = lambda: headline_texts(fetch_headlines())
    rec = Runner(broker, cfg, headlines=headlines, **kw).cycle()
    if feed == "replay":
        broker.feed.advance()
    return _clean(rec)


def paper_reset(initial_equity: float = 100_000.0) -> dict:
    for name in ("paper_account.json", "runner_state.json", "replay_pos.json", "journal.jsonl"):
        p = _home() / name
        if p.exists():
            p.unlink()
    global _replay
    _replay = None
    return {"ok": True, "initial_equity": initial_equity}


def optimize(strategy: str = "london_breakout", train_days: int = 365, test_days: int = 91,
             grid: dict | None = None, **mkw) -> dict:
    from .research import GRIDS, pbo_cscv, reality_check, walk_forward_optimize
    from .validation import deflated_sharpe, probabilistic_sharpe

    if strategy not in GRIDS and not grid:
        raise ValueError(f"no parameter grid for {strategy}; choose from {list(GRIDS)}")
    md, feats = market(**mkw)
    strat = STRATEGIES[strategy]
    res = walk_forward_optimize(md, feats, strat.signal, grid or GRIDS[strategy], strat.config,
                                train_days=train_days, test_days=test_days)
    R = res.returns_matrix
    pbo = pbo_cscv(R.to_numpy()) if R.shape[1] >= 2 else {}
    rc = reality_check(R.to_numpy(), n_boot=500) if R.shape[1] >= 2 else {}
    oos = res.oos_returns
    return _clean({
        "strategy": strategy, "n_trials": res.n_trials, "grid": grid or GRIDS[strategy],
        "oos_sharpe": res.sharpe(), "oos_total_return": float(res.oos_equity.iloc[-1] - 1) if len(oos) else None,
        "oos_psr": probabilistic_sharpe(oos) if len(oos) > 10 else None,
        "oos_dsr": deflated_sharpe(oos, res.n_trials) if len(oos) > 10 else None,
        "pbo": pbo, "reality_check": rc,
        "windows": res.chosen.astype(str).to_dict("records"),
        "oos_equity": [{"time": int(pd.Timestamp(t).timestamp()), "value": round(float(v), 5)}
                       for t, v in res.oos_equity.items()],
        "in_sample_sharpes": {str(i): float(R[c].mean() / R[c].std() * np.sqrt(252)) if R[c].std() > 0 else 0.0
                              for i, c in enumerate(R.columns)},
    })


def swing_chart(strategy: str = "swing_halftrend_structure", **mkw) -> dict:
    """Everything needed to draw the friend's chart for a swing strategy: candles, trailing
    line, EMAs, flip markers, trades with their stop and target, the timeframe table,
    win rates, the PO3 candle and the plan for the current signal."""
    from .indicators import mtf_directions, po3_candle, side_win_rates
    from .swing import swing_components, swing_stops, trade_plan

    strat = STRATEGIES.get(strategy)
    if strat is None or strat.swing is None:
        raise ValueError(f"{strategy!r} is not a swing strategy; choose from "
                         f"{[n for n, x in STRATEGIES.items() if x.swing is not None]}")
    md, feats = market(**mkw)
    res, sig, m, line = run_strategy(md, strat, feats)
    d = m.bars
    comp = swing_components(d, strat.swing)
    t = lambda ts: int(pd.Timestamp(ts).timestamp())
    tail = d.index[-min(len(d), 1500)]
    view = comp.loc[tail:]
    candles = [{"time": t(i), "open": round(r.open, 2), "high": round(r.high, 2), "low": round(r.low, 2),
                "close": round(r.close, 2)} for i, r in zip(d.loc[tail:].index, d.loc[tail:].itertuples())]
    trail = [{"time": t(i), "value": round(v, 2), "color": "up" if dr > 0 else "down"}
             for i, v, dr in zip(view.index, view["line"], view["dir"]) if np.isfinite(v)]
    emas = {k: [{"time": t(i), "value": round(v, 2)} for i, v in view[k].dropna().items()]
            for k in ("ema_regime", "ema_fast", "ema_mid") if k in view}
    flips = view["dir"].diff().fillna(0)
    markers = [{"time": t(i), "side": int(np.sign(v))} for i, v in flips[flips != 0].items()]
    trades = [{"entry_time": t(r.entry_time), "exit_time": t(r.exit_time), "side": int(r.side),
               "entry": r.entry_price, "exit": r.exit_price, "stop": r.initial_stop, "target": r.target,
               "reason": r.reason, "pnl": r.net_pnl, "r": r.r_multiple}
              for r in res.trades.itertuples() if r.entry_time >= tail]
    intraday = len(md.bars) > len(d)
    if intraday:
        rules = {"60": "1h", "240": "4h", "1D": "1D", "1W": "1W"}
        mt = mtf_directions(md.bars.tail(24 * 5 * 60), rules, strat.swing.engine, **strat.swing.engine_params)
    else:
        rules = {"1D": "1D", "3D": "3D", "1W": "1W"}
        mt = mtf_directions(d, rules, strat.swing.engine, **strat.swing.engine_params)
    last = mt.iloc[-1]
    tfs = {k: (None if pd.isna(last[k]) else int(last[k])) for k in rules}
    # The daily row is the strategy's own daily engine (broker trading day, full history).
    tfs["1D"] = int(comp["dir"].iloc[-1])
    table = {"timeframes": tfs, "aligned_up": sum(1 for v in tfs.values() if v == 1),
             "aligned_down": sum(1 for v in tfs.values() if v == -1), **side_win_rates(res.trades)}
    side = int(comp["dir"].iloc[-1])
    stops = swing_stops(d) if strat.entry_stops else None
    plan = trade_plan(d, side, res.config.initial_equity, strat.config, line if strat.stop_line else None, stops)
    open_trade = trades[-1] if trades and trades[-1]["reason"] == "end" else None
    return _clean({
        "strategy": strategy, "description": strat.description, "timeframe": strat.timeframe,
        "engine": strat.swing.engine, "engine_params": strat.swing.engine_params,
        "candles": candles, "trail": trail, "emas": emas, "markers": markers, "trades": trades[-60:],
        "table": table, "po3": {**po3_candle(md.bars if intraday else d, "1D"), "time": None},
        "plan": plan, "open_trade": open_trade, "last_flip": markers[-1] if markers else None,
        "stats": {k: res.stats[k] for k in ("total_return", "sharpe", "max_drawdown", "trades", "win_rate",
                                             "profit_factor", "avg_r", "swap")},
    })


def news(limit: int = 30) -> list[dict]:
    from .news import fetch_headlines
    return _clean([{**i, "published": i["published"].isoformat() if i["published"] is not None else None}
                   for i in fetch_headlines(limit=limit)])


def review_paper(use_llm: bool = True) -> dict:
    import json
    from .review import review

    path = _home() / "paper_account.json"
    trades = json.loads(path.read_text())["trades"] if path.exists() else []
    return _clean(review(trades, use_llm=use_llm, lessons_path=_home() / "lessons.json"))


def execute_alert(payload: dict) -> dict | None:
    """With AURUM_WEBHOOK_EXECUTE=paper, a TradingView alert trades the paper account.

    buy / sell open (reversing if needed) at the alert price with the alert's
    qty_oz; close flattens. Nothing else is ever executed from a webhook.
    """
    if os.environ.get("AURUM_WEBHOOK_EXECUTE", "").lower() != "paper":
        return None
    action = str(payload.get("action", "")).lower()
    price = payload.get("price")
    if action not in ("buy", "sell", "close") or not isinstance(price, (int, float)):
        return {"executed": False, "reason": "needs action buy/sell/close and a numeric price"}
    from .execution import PaperBroker

    b = PaperBroker(lambda n: pd.DataFrame(), _home() / "paper_account.json")  # prices come from the alert
    b.set_price(float(price), pd.Timestamp.now(tz="UTC").isoformat())
    results = []
    pos = b.position()
    want = {"buy": 1, "sell": -1, "close": 0}[action]
    if pos and pos.side != want:
        results.append(b.close(f"tv {action}").to_dict())
    if want and (pos is None or pos.side != want):
        lots = b.instrument.round_lots(float(payload.get("qty_oz", 0)) / b.instrument.contract_size_oz)
        stop = payload.get("stop")
        results.append(b.open(want, lots, float(stop) if isinstance(stop, (int, float)) else None, None,
                              f"tv {payload.get('mode', '')}").to_dict())
    return {"executed": True, "orders": results}
