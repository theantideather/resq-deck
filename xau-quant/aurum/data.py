"""Market data for gold: loaders for real sources and a synthetic market.

`MarketData` holds two frames on the same UTC index:

    bars    open, high, low, close, volume for XAUUSD (or GC futures)
    macro   the cross asset context gold trades off: DXY, US 10y real yield,
            10y breakeven, VIX, silver, and CFTC managed money positioning

Daily and weekly macro series are joined to intraday bars only after the
moment they were actually published (`available_after`). Joining a daily
close to the same day's hourly bars is the most common lookahead bug in
retail gold backtests, and it makes a macro model look far better than it is.

Real sources (need network and, for Yahoo, `pip install yfinance`):

    Yahoo    GC=F gold futures, DX-Y.NYB dollar index, SI=F silver, ^VIX, ^GSPC
    FRED     DFII10 10y TIPS real yield, T10YIE 10y breakeven
    CFTC     disaggregated Commitments of Traders, gold (code 088691)
    CSV      MT5 history exports or any OHLCV file
"""

from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import calendar as cal

MACRO_COLUMNS = ["dxy", "real_yield", "breakeven", "vix", "silver", "cot_mm_net"]


@dataclass
class MarketData:
    bars: pd.DataFrame
    macro: pd.DataFrame
    events: pd.DataFrame
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.bars.index = cal.ensure_utc(pd.DatetimeIndex(self.bars.index))
        self.macro = self.macro.reindex(self.bars.index).ffill()

    def slice(self, start=None, end=None) -> "MarketData":
        b = self.bars.loc[start:end]
        return MarketData(b, self.macro.loc[b.index], self.events, self.source)

    @property
    def freq_hours(self) -> float:
        d = pd.Series(self.bars.index).diff().median()
        return d / pd.Timedelta(hours=1)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_to_bars(
    series: pd.Series,
    index: pd.DatetimeIndex,
    available_after: str | pd.Timedelta,
) -> pd.Series:
    """Forward fill a lower frequency series onto bars, respecting publication time.

    `series` is indexed by observation date (midnight). A value stamped date d
    is treated as known from d + available_after. Examples:

        Yahoo daily close of DXY         "22h"   (NY close, 17:00 ET)
        FRED DFII10                      "46h"   (posted the next afternoon)
        CFTC COT, as of Tuesday          "3D21h" (released Friday 15:30 ET)
    """
    s = series.dropna().copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.normalize() + pd.Timedelta(available_after)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    index = cal.ensure_utc(index)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


# ---------------------------------------------------------------------------
# Synthetic market
# ---------------------------------------------------------------------------

def synthetic_market(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    freq: str = "1h",
    seed: int = 7,
    start_price: float = 1280.0,
) -> MarketData:
    """A gold market with the structure real XAUUSD has, for offline work and tests.

    What is built in, and so what strategies can find:
      * hourly volatility that follows the session clock (quiet Asia, loud
        London/New York overlap) with GARCH style clustering
      * a negative beta to the dollar and to real yield changes
      * a slow pull back towards a macro fair value set by DXY and real yields
      * persistent bull, range and bear regimes with their own drift and vol
      * jumps at NFP and FOMC times

    It is not a forecast of anything. Use it to check that code works and that
    the research pipeline can recover structure that is known to be there,
    then run the same pipeline on real data before believing any number.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    # Gold CFDs close from Friday 17:00 New York to Sunday 18:00 New York.
    ny = idx.tz_convert(cal.NEW_YORK)
    closed = (ny.weekday == 5) | ((ny.weekday == 4) & (ny.hour >= 17)) | ((ny.weekday == 6) & (ny.hour < 18))
    idx = idx[~closed]
    n = len(idx)
    hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    bars_per_year = 24 * 5.2 * 52 / hours

    # Session volatility profile.
    sess = cal.session_flags(idx)
    profile = np.full(n, 0.7)
    profile[sess["asia"].to_numpy()] = 0.8
    profile[sess["london"].to_numpy()] = 1.2
    profile[sess["new_york"].to_numpy()] = 1.4
    profile[sess["overlap"].to_numpy()] = 1.7
    profile /= profile.mean()

    # Regimes: 0 bull, 1 range, 2 bear. Mean stay about three months.
    drift_ann = np.array([0.28, 0.0, -0.18])
    vol_mult = np.array([1.0, 0.75, 1.35])
    stay = 1 - 1 / (bars_per_year / 4)
    regime = np.empty(n, dtype=int)
    r = 1
    for i in range(n):
        if rng.random() > stay:
            r = rng.choice([x for x in range(3) if x != r])
        regime[i] = r

    # Macro drivers, hourly.
    dxy_vol = 0.065 / np.sqrt(bars_per_year)
    dxy_ret = rng.standard_normal(n) * dxy_vol * profile
    dxy = 96.0 * np.exp(np.cumsum(dxy_ret))
    ry = np.empty(n)
    ry[0] = 0.8
    ry_vol = 0.9 / np.sqrt(bars_per_year)  # percentage points per year
    for i in range(1, n):
        ry[i] = ry[i - 1] + 0.3 / bars_per_year * (1.0 - ry[i - 1]) + ry_vol * profile[i] * rng.standard_normal()
    d_ry_bp = np.diff(ry, prepend=ry[0]) * 100
    vix = np.empty(n)
    lv = np.log(16.0)
    for i in range(n):
        lv += 2.0 / bars_per_year * (np.log(17.0) - lv) + 0.9 / np.sqrt(bars_per_year) * rng.standard_normal()
        vix[i] = np.exp(lv)

    # GARCH(1,1) shocks.
    base_vol = 0.12 / np.sqrt(bars_per_year)
    omega, alpha, beta = 0.05, 0.08, 0.87
    h = 1.0
    eps = np.empty(n)
    for i in range(n):
        z = rng.standard_normal()
        eps[i] = np.sqrt(h) * z
        h = omega + alpha * eps[i] ** 2 + beta * h

    events = cal.event_calendar(idx[0].year, idx[-1].year)
    ev_flag = cal.event_risk(idx, events, before="0h", after=freq).to_numpy()

    # Fair value: a DXY and real yield anchor that also drifts with the regime,
    # so the pull back towards it is a real but small edge.
    anchor = np.log(start_price) + 1.1 * np.log(dxy[0]) + 0.12 * ry[0]
    anchor = anchor + np.cumsum(drift_ann[regime] / bars_per_year)
    fair = anchor - 1.1 * np.log(dxy) - 0.12 * ry
    jumps = np.where(ev_flag, rng.standard_normal(n) * base_vol * 6, 0.0)
    shock = (
        drift_ann[regime] / bars_per_year
        - dxy_ret
        - 0.0012 * d_ry_bp
        + base_vol * vol_mult[regime] * profile * eps
        + jumps
    )
    logp = np.empty(n)
    logp[0] = np.log(start_price)
    for i in range(1, n):
        logp[i] = logp[i - 1] + shock[i] - 0.004 * (logp[i - 1] - fair[i - 1])
    close = np.exp(logp)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    # Weekend gaps: the first bar after a close opens away from Friday's close.
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(idx.asi8) > pd.Timedelta(freq).value * 2
    open_[gap] *= np.exp(rng.standard_normal(gap.sum()) * base_vol * 3)
    wick = np.abs(rng.standard_normal((2, n))) * base_vol * profile * 0.6 * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = (1000 * profile * (1 + 3 * ev_flag) * rng.lognormal(0, 0.3, n)).round()

    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    silver = close / 80.0 * np.exp(np.cumsum(rng.standard_normal(n) * base_vol * 0.8))
    # Managed money net length follows 13 week trend, weekly, released with a lag.
    weekly = pd.Series(close, index=idx).resample("W-TUE").last()
    mm = (weekly.pct_change(13).fillna(0) * 900_000).clip(-120_000, 280_000) + 80_000
    macro = pd.DataFrame(
        {
            "dxy": dxy,
            "real_yield": ry,
            "breakeven": 2.2 + 0.1 * rng.standard_normal(n).cumsum() / np.sqrt(bars_per_year),
            "vix": vix,
            "silver": silver,
        },
        index=idx,
    )
    macro["cot_mm_net"] = align_to_bars(mm, idx, "3D21h")
    macro["regime_true"] = regime  # only for tests; never used as a feature
    return MarketData(bars, macro, events, source="synthetic")


# ---------------------------------------------------------------------------
# Real data loaders
# ---------------------------------------------------------------------------

YAHOO = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "silver": "SI=F",
    "vix": "^VIX",
    "spx": "^GSPC",
    "gld": "GLD",
}


def load_yahoo(ticker: str, interval: str = "1h", period: str = "729d") -> pd.DataFrame:
    """OHLCV from Yahoo Finance. Intraday history is capped at about two years."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install yfinance to load Yahoo data") from e
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = cal.ensure_utc(pd.DatetimeIndex(df.index))
    return df.dropna()


def load_fred(series_id: str) -> pd.Series:
    """A daily FRED series, e.g. DFII10 (10y real yield) or T10YIE (breakeven)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + urllib.parse.quote(series_id)
    with urllib.request.urlopen(url, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = df["date"]
    return s.dropna()


# CFTC Public Reporting Environment, Disaggregated Futures Only.
CFTC_DISAGG_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
GOLD_CFTC_CODE = "088691"


def load_cot_gold(limit: int = 1500) -> pd.DataFrame:
    """Weekly COMEX gold positioning: managed money long, short, net, open interest."""
    q = urllib.parse.urlencode(
        {
            "cftc_contract_market_code": GOLD_CFTC_CODE,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": str(limit),
        }
    )
    with urllib.request.urlopen(f"{CFTC_DISAGG_URL}?{q}", timeout=30) as r:
        rows = json.loads(r.read())
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "mm_long": pd.to_numeric(df["m_money_positions_long_all"]),
            "mm_short": pd.to_numeric(df["m_money_positions_short_all"]),
            "open_interest": pd.to_numeric(df["open_interest_all"]),
        }
    )
    out.index = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    out["mm_net"] = out["mm_long"] - out["mm_short"]
    return out.sort_index()


def load_csv(path: str) -> pd.DataFrame:
    """OHLCV from a CSV. Reads MT5 history exports (<DATE> <TIME> ...) and plain files.

    Plain files need a time column (time, datetime, date or timestamp) and
    open/high/low/close. Naive timestamps are taken as UTC; MT5 exports are in
    broker server time, so shift them first if your broker is on UTC+2/+3.
    """
    with open(path) as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        col = next(c for c in ("timestamp", "datetime", "time", "date") if c in df.columns)
        ts = pd.to_datetime(df[col])
    if "volume" not in df.columns:
        df["volume"] = df.get("tickvol", df.get("vol", 0))
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = cal.ensure_utc(pd.DatetimeIndex(ts))
    return out.sort_index()


def load_market(
    source: str = "synthetic",
    csv_path: str | None = None,
    interval: str = "1h",
    with_macro: bool = True,
    **kwargs,
) -> MarketData:
    """One call to get bars plus aligned macro context.

    source = "synthetic" | "yahoo" | "csv"
    """
    if source == "synthetic":
        return synthetic_market(freq=interval, **kwargs)
    if source == "yahoo":
        bars = load_yahoo(YAHOO["gold"], interval=interval)
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv source needs csv_path")
        bars = load_csv(csv_path)
    else:
        raise ValueError(f"unknown source {source!r}")

    return market_from_bars(bars, with_macro=with_macro, source=source)


def load_series_csv(path: str) -> pd.Series:
    """A daily series from a two column CSV (date, value), e.g. central bank
    gold purchases or GLD tonnes exported from their publishers."""
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna().sort_index()


def extra_series_spec(spec: str | None = None) -> list[tuple[str, str, str]]:
    """Parse AURUM_EXTRA_SERIES: "name=path.csv@lag;name2=path2.csv@lag2".

    Each becomes macro column x_<name>, joined `lag` after its date (default 1D).
    """
    spec = spec if spec is not None else os.environ.get("AURUM_EXTRA_SERIES", "")
    out = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, rest = part.split("=", 1)
        path, _, lag = rest.partition("@")
        out.append((name.strip(), path.strip(), lag.strip() or "1D"))
    return out


def events_with_extras(start: int, end: int) -> pd.DataFrame:
    """The built in NFP/FOMC calendar plus AURUM_EVENTS_CSV (timestamp,event), e.g. CPI dates."""
    extra = None
    path = os.environ.get("AURUM_EVENTS_CSV")
    if path and os.path.exists(path):
        extra = pd.read_csv(path)
    return cal.event_calendar(start, end, extra)


def market_from_bars(bars: pd.DataFrame, with_macro: bool = True, source: str = "bars",
                     quiet: bool = False) -> MarketData:
    """Wrap OHLCV bars with macro context aligned to their publication times.

    Used by load_market and by the live runner, so live features are built
    exactly the way the backtest built them.
    """
    idx = cal.ensure_utc(pd.DatetimeIndex(bars.index))
    bars = bars.set_axis(idx)
    macro = pd.DataFrame(index=idx)
    if with_macro:
        loaders = {
            "dxy": lambda: align_to_bars(load_yahoo(YAHOO["dxy"], "1d", "10y")["close"], idx, "22h"),
            "silver": lambda: align_to_bars(load_yahoo(YAHOO["silver"], "1d", "10y")["close"], idx, "22h"),
            "vix": lambda: align_to_bars(load_yahoo(YAHOO["vix"], "1d", "10y")["close"], idx, "22h"),
            "real_yield": lambda: align_to_bars(load_fred("DFII10"), idx, "46h"),
            "breakeven": lambda: align_to_bars(load_fred("T10YIE"), idx, "46h"),
            "cot_mm_net": lambda: align_to_bars(load_cot_gold()["mm_net"], idx, "3D21h"),
        }
        for name, path, lag in extra_series_spec():
            loaders[f"x_{name}"] = (lambda p=path, g=lag: align_to_bars(load_series_csv(p), idx, g))
        for name, fn in loaders.items():
            try:
                macro[name] = fn()
            except Exception as e:  # keep going with what we have
                if not quiet:
                    print(f"[aurum] macro series {name} unavailable: {e}")
    events = events_with_extras(idx[0].year, idx[-1].year + 1)
    return MarketData(bars, macro, events, source=source)
