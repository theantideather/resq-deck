"""Trend flip indicators and dashboard helpers, ported from their TradingView originals.

Every engine returns a DataFrame with:
    dir    +1 uptrend, -1 downtrend (known at the bar's close)
    line   the trailing level: the stop a swing trader trails behind price

    supertrend   ta.supertrend: ATR (RMA) bands around hl2 that only ratchet
                 in the trend's favour. Default 10, 3.
    halftrend    everget's HalfTrend: amplitude-bar highs/lows with an
                 ATR(100) channel. Default amplitude 2.
    chandelier   everget's Chandelier Exit: highest close minus k*ATR for
                 longs, lowest close plus k*ATR for shorts. Default 22, 3.
    ut_bot       UT Bot Alerts (QuantNomad): ATR trailing stop on close.
                 Default key 1, ATR 10.

These are the indicators whose look fits the friend's BTC chart (stepped
trailing line, ribbon fill to price, triangle flip markers). Settings
calibrated against his marker dates are in docs/SWING.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def true_range(d: pd.DataFrame) -> pd.Series:
    pc = d["close"].shift()
    return pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)


def atr_rma(d: pd.DataFrame, n: int) -> pd.Series:
    return rma(true_range(d), n)


def supertrend(d: pd.DataFrame, period: int = 10, factor: float = 3.0) -> pd.DataFrame:
    a = atr_rma(d, period).to_numpy()
    hl2 = ((d["high"] + d["low"]) / 2).to_numpy()
    c = d["close"].to_numpy()
    n = len(c)
    up, dn = hl2 - factor * a, hl2 + factor * a
    fu, fd = up.copy(), dn.copy()
    direction = np.ones(n)
    for i in range(1, n):
        if np.isnan(a[i]):
            continue
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        if direction[i - 1] < 0 and c[i] > fd[i - 1]:
            direction[i] = 1
        elif direction[i - 1] > 0 and c[i] < fu[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction > 0, fu, fd)
    return pd.DataFrame({"dir": direction, "line": line}, index=d.index)


def halftrend(d: pd.DataFrame, amplitude: int = 2) -> pd.DataFrame:
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    n = len(c)
    hp = d["high"].rolling(amplitude).max().to_numpy()
    lp = d["low"].rolling(amplitude).min().to_numpy()
    hma = d["high"].rolling(amplitude).mean().to_numpy()
    lma = d["low"].rolling(amplitude).mean().to_numpy()
    trend, nxt = 0, 0
    max_low = l[0]
    min_high = h[0]
    up = down = np.nan
    direction = np.ones(n)
    line = np.full(n, np.nan)
    prev_trend = 0
    for i in range(1, n):
        if np.isnan(hp[i]):
            direction[i] = 1
            continue
        if nxt == 1:
            max_low = max(lp[i], max_low)
            if hma[i] < max_low and c[i] < l[i - 1]:
                trend, nxt, min_high = 1, 0, hp[i]
        else:
            min_high = min(hp[i], min_high)
            if lma[i] > min_high and c[i] > h[i - 1]:
                trend, nxt, max_low = 0, 1, lp[i]
        if trend == 0:
            up = (down if not np.isnan(down) else max_low) if prev_trend != 0 else (
                max_low if np.isnan(up) else max(max_low, up))
        else:
            down = (up if not np.isnan(up) else min_high) if prev_trend != 1 else (
                min_high if np.isnan(down) else min(min_high, down))
        direction[i] = 1 if trend == 0 else -1
        line[i] = up if trend == 0 else down
        prev_trend = trend
    return pd.DataFrame({"dir": direction, "line": line}, index=d.index)


def chandelier(d: pd.DataFrame, length: int = 22, mult: float = 3.0, use_close: bool = True) -> pd.DataFrame:
    a = (atr_rma(d, length) * mult).to_numpy()
    hi = (d["close"] if use_close else d["high"]).rolling(length).max().to_numpy()
    lo = (d["close"] if use_close else d["low"]).rolling(length).min().to_numpy()
    ls, ss = (hi - a).copy(), (lo + a).copy()
    c = d["close"].to_numpy()
    n = len(c)
    direction = np.ones(n)
    for i in range(1, n):
        if np.isnan(ls[i]) or np.isnan(ss[i]):
            continue
        lp = ls[i - 1] if not np.isnan(ls[i - 1]) else ls[i]
        sp = ss[i - 1] if not np.isnan(ss[i - 1]) else ss[i]
        ls[i] = max(ls[i], lp) if c[i - 1] > lp else ls[i]
        ss[i] = min(ss[i], sp) if c[i - 1] < sp else ss[i]
        direction[i] = 1 if c[i] > sp else (-1 if c[i] < lp else direction[i - 1])
    line = np.where(direction > 0, ls, ss)
    return pd.DataFrame({"dir": direction, "line": line}, index=d.index)


def ut_bot(d: pd.DataFrame, key: float = 1.0, period: int = 10) -> pd.DataFrame:
    x = (atr_rma(d, period) * key).to_numpy()
    src = d["close"].to_numpy()
    n = len(src)
    ts = np.zeros(n)
    pos = np.ones(n)
    for i in range(1, n):
        p = ts[i - 1]
        if np.isnan(x[i]):
            ts[i] = src[i]
            continue
        if src[i] > p and src[i - 1] > p:
            ts[i] = max(p, src[i] - x[i])
        elif src[i] < p and src[i - 1] < p:
            ts[i] = min(p, src[i] + x[i])
        elif src[i] > p:
            ts[i] = src[i] - x[i]
        else:
            ts[i] = src[i] + x[i]
        if src[i - 1] < p and src[i] > p:
            pos[i] = 1
        elif src[i - 1] > p and src[i] < p:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]
    return pd.DataFrame({"dir": pos, "line": ts}, index=d.index)


ENGINES = {
    "supertrend": (supertrend, {"period": 10, "factor": 3.0}),
    "halftrend": (halftrend, {"amplitude": 2}),
    "chandelier": (chandelier, {"length": 22, "mult": 3.0}),
    "ut_bot": (ut_bot, {"key": 1.0, "period": 10}),
}


def engine(d: pd.DataFrame, name: str = "supertrend", **params) -> pd.DataFrame:
    fn, defaults = ENGINES[name]
    return fn(d, **{**defaults, **params})


def flips(direction: pd.Series) -> pd.Series:
    """+1 on the bar an uptrend starts, -1 on the bar a downtrend starts, else 0."""
    ch = direction.diff().fillna(0)
    return pd.Series(np.sign(ch), index=direction.index)


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


def resample_ohlc(d: pd.DataFrame, rule: str, offset: str | None = None) -> pd.DataFrame:
    agg = {**OHLC, **({"volume": "sum"} if "volume" in d else {})}
    kw = {"offset": offset} if offset else {}
    return d.resample(rule, label="left", closed="left", **kw).agg(agg).dropna(subset=["open"])


def mtf_directions(d: pd.DataFrame, rules: dict[str, str], name: str = "supertrend", **params) -> pd.DataFrame:
    """Engine direction on several timeframes, aligned to d's index without lookahead.

    A higher timeframe bar's direction only becomes visible once that bar has
    closed, i.e. on base bars that start at or after its close.
    """
    out = pd.DataFrame(index=d.index)
    step = d.index.to_series().diff().median()
    for label, rule in rules.items():
        h = resample_ohlc(d, rule)
        if len(h) < 30:
            out[label] = np.nan
            continue
        dirn = engine(h, name, **params)["dir"]
        close_time = h.index + pd.Timedelta(rule) if not rule.endswith(("W", "ME")) else h.index.shift(1, freq=rule)
        s = pd.Series(dirn.to_numpy(), index=close_time)
        out[label] = s.reindex(s.index.union(d.index + step)).ffill().reindex(d.index + step).to_numpy()
    out["aligned_up"] = (out[list(rules)] > 0).sum(axis=1)
    out["aligned_down"] = (out[list(rules)] < 0).sum(axis=1)
    return out


def po3_candle(d: pd.DataFrame, rule: str = "1D") -> dict:
    """The current higher timeframe candle (ICT 'power of three' view): open, high, low, close, range."""
    h = resample_ohlc(d, rule)
    last = h.iloc[-1]
    return {"time": h.index[-1], "open": float(last["open"]), "high": float(last["high"]),
            "low": float(last["low"]), "close": float(last["close"]), "range": float(last["high"] - last["low"])}


def side_win_rates(trades: pd.DataFrame) -> dict:
    """Long and short win rates with counts, like the 'Long WR 75% (4)' rows of the chart table."""
    out = {}
    for side, name in ((1, "long"), (-1, "short")):
        t = trades[trades["side"] == side] if len(trades) else trades
        out[name] = {"win_rate": float((t["net_pnl"] > 0).mean()) if len(t) else None, "trades": int(len(t))}
    return out
