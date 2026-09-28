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
from .strategies import STRATEGIES, ensemble

PINE_PATH = Path(__file__).resolve().parent.parent / "tradingview" / "aurum_gold.pine"
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
    return [{"name": s.name, "description": s.description,
             "stop_atr": s.config.stop_atr, "target_atr": s.config.target_atr,
             "risk_per_trade": s.config.risk_per_trade} for s in STRATEGIES.values()]


def compare(**mkw) -> dict:
    md, feats = market(**mkw)
    rows = []
    for name, strat in STRATEGIES.items():
        st = run_backtest(md, strat.signal(md, feats), strat.config).stats
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
    sig = strat.signal(md, feats)
    res = run_backtest(md, sig, cfg)
    eq = res.equity.resample("1D").last().dropna()
    dd = eq / eq.cummax() - 1
    candles, tf = _downsample_candles(md.bars)
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
        card = scorecard(md, sig, res, n_trials=n_trials)
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
    return PINE_PATH.read_text()


# TradingView webhook alerts, appended as JSON lines so the web server and
# the MCP server (separate processes) see the same log.
ALERT_LOG = Path(os.environ.get("AURUM_ALERT_LOG", Path.home() / ".aurum" / "alerts.jsonl"))


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
