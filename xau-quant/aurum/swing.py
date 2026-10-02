"""Daily swing system modelled on the friend's BTC chart, rebuilt for XAUUSD.

What his chart shows (BTCUSDT perpetual, 1D):
    * a trend flip engine with a stepped trailing line, a ribbon fill and
      triangle markers; calibrated against his marker dates, HalfTrend with
      amplitude 5 fits best (7 of 10 within 3 days), Supertrend 10/3 next
    * a slow black moving average (read here as EMA 200) as the regime line
    * thin fast averages near price (EMA 20 and EMA 50)
    * a multi timeframe table (5m, 15m, 1h, 4h, 1D direction, "Aligned 3/2")
      with the signal's own long and short win rates
    * a "D candle PO3" box: the current daily candle drawn beside price
    * trades planned with TradingView's position tool: about a 4% stop for
      a 29% target (7.15 R), held for weeks

Rules here:
    entry   the engine flips (direction change on a completed daily bar);
            optional regime filter (longs above EMA 200, shorts below) and
            optional multi timeframe alignment
    exit    one of the stop presets below, plus an opposite flip
    re-entry after a stop, wait for the next flip

Stop presets (pick with `stop_mode`) until his exact SL rule arrives:
    trail     stop starts at the engine line and trails it (classic swing)
    pct_r     fixed % stop and an R multiple target (his position tool: 4%, 7R)
    pct_be    fixed % stop, breakeven at 1R, then trail the engine line
    atr       2.5 ATR stop, trail the engine line
    swing_r   stop beyond the 10 bar swing high/low plus 0.5 ATR, 7R target:
              matches his BTC short of 28 Oct 2025 (stop 118,028; 10 day
              high 116,381 + 0.46 ATR; target 7.15R)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import calendar as cal
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .data import MarketData
from .indicators import engine, mtf_directions


def to_daily(bars: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily candles on the broker's trading day (rolls at 17:00 New York), indexed at 00:00 UTC.

    With `now`, a day that has not closed yet is dropped, so live decisions only
    ever use completed daily candles.
    """
    if len(bars) and (bars.index[1] - bars.index[0]) >= pd.Timedelta(hours=20):
        return bars
    day = pd.DatetimeIndex(pd.to_datetime(np.asarray(cal.trading_day(bars.index)))).tz_localize("UTC")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in bars:
        agg["volume"] = "sum"
    if "spread" in bars:
        agg["spread"] = "mean"
    d = bars.groupby(day).agg(agg)
    d.index.name = None
    if now is not None and len(d):
        # Trading day D closes at 17:00 New York on calendar day D.
        close_ny = pd.Timestamp(d.index[-1].date()).tz_localize(cal.NEW_YORK) + pd.Timedelta(hours=17)
        if now < close_ny.tz_convert("UTC"):
            d = d.iloc[:-1]
    return d


def bars_for(bars: pd.DataFrame, timeframe: str = "D") -> pd.DataFrame:
    """Candles for the swing timeframe: "D" uses the broker trading day, anything
    else ("4h", "12h") is a plain UTC resample."""
    if timeframe in ("D", "1D"):
        return to_daily(bars)
    from .indicators import resample_ohlc
    agg_cols = [c for c in ("spread",) if c in bars]
    out = resample_ohlc(bars, timeframe)
    for c in agg_cols:
        out[c] = bars[c].resample(timeframe).mean().reindex(out.index)
    return out


def daily_market(md: MarketData, timeframe: str = "D") -> MarketData:
    """The same market on swing timeframe candles; macro columns take each candle's last value."""
    d = bars_for(md.bars, timeframe)
    if len(d) == len(md.bars):
        return md
    if len(md.macro.columns):
        macro = md.macro.reindex(d.index, method="ffill")
    else:
        macro = pd.DataFrame(index=d.index)
    return MarketData(d, macro, md.events, md.source)


@dataclass
class SwingParams:
    engine: str = "halftrend"
    engine_params: dict = field(default_factory=lambda: {"amplitude": 5})
    regime_ema: int | None = 200        # None disables the regime filter
    allow_long: bool = True
    allow_short: bool = True
    mtf_rules: dict | None = None       # e.g. {"1D": "1D", "3D": "3D", "1W": "1W"} on daily data
    mtf_min: int = 0                    # minimum aligned timeframes in the trade's direction


def swing_components(d: pd.DataFrame, p: SwingParams) -> pd.DataFrame:
    """Everything the chart shows, per daily bar."""
    e = engine(d, p.engine, **p.engine_params)
    out = pd.DataFrame(index=d.index)
    out["dir"] = e["dir"]
    out["line"] = e["line"]
    out["ema_fast"] = d["close"].ewm(span=20, adjust=False).mean()
    out["ema_mid"] = d["close"].ewm(span=50, adjust=False).mean()
    if p.regime_ema:
        out["ema_regime"] = d["close"].ewm(span=p.regime_ema, adjust=False).mean()
    if p.mtf_rules:
        m = mtf_directions(d, p.mtf_rules, p.engine, **p.engine_params)
        out = out.join(m)
    return out


def swing_signal(d: pd.DataFrame, p: SwingParams) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """(signal, trailing line, components). Signal is the side to hold, 0 when filters say stand aside."""
    comp = swing_components(d, p)
    side = comp["dir"].copy()
    if p.regime_ema:
        above = d["close"] > comp["ema_regime"]
        side = side.where(~((side > 0) & ~above), 0.0)
        side = side.where(~((side < 0) & above), 0.0)
    if p.mtf_rules and p.mtf_min:
        side = side.where(~((side > 0) & (comp["aligned_up"] < p.mtf_min)), 0.0)
        side = side.where(~((side < 0) & (comp["aligned_down"] < p.mtf_min)), 0.0)
    if not p.allow_long:
        side = side.where(side <= 0, 0.0)
    if not p.allow_short:
        side = side.where(side >= 0, 0.0)
    # Only act on a fresh flip: hold while the engine keeps its direction,
    # but do not jump into a trend in the middle because a filter turned on.
    flip = comp["dir"].diff().fillna(0) != 0
    held = pd.Series(0.0, index=d.index)
    state = 0.0
    for i, (s, f, dr) in enumerate(zip(side.to_numpy(), flip.to_numpy(), comp["dir"].to_numpy())):
        if f:
            state = s
        elif state != 0 and np.sign(dr) != np.sign(state):
            state = 0.0
        held.iloc[i] = state
    return held, comp["line"], comp


def swing_stops(d: pd.DataFrame, lookback: int = 10, buffer_atr: float = 0.5) -> pd.DataFrame:
    """Structure stops: beyond the last `lookback` bars' extreme by `buffer_atr` ATR(14).

    Fits the friend's BTC short of 28 Oct 2025: stop 118,028 vs a 10 day high
    of 116,381 plus 0.46 ATR. Columns long (below the swing low) and short.
    """
    from .indicators import atr_rma

    a = atr_rma(d, 14)
    return pd.DataFrame({"long": d["low"].rolling(lookback).min() - buffer_atr * a,
                         "short": d["high"].rolling(lookback).max() + buffer_atr * a}, index=d.index)


def trade_plan(d: pd.DataFrame, side: int, equity: float, cfg: BacktestConfig,
               line: pd.Series | None = None, stops: pd.DataFrame | None = None,
               contract_size: float = 100.0) -> dict:
    """Entry, stop, target, R:R and size for a trade taken at the last close,
    laid out like TradingView's position tool."""
    from .backtest import initial_stop_distance, risk_lots
    from .indicators import atr_rma

    entry = float(d["close"].iloc[-1])
    a = float(atr_rma(d, 14).iloc[-1])
    if stops is not None:
        level = float(stops["long" if side > 0 else "short"].iloc[-1])
        dist = side * (entry - level)
        dist = dist if np.isfinite(dist) and dist > 0 else 2.0 * a
    else:
        line_dist = side * (entry - float(line.iloc[-1])) if line is not None else None
        dist = initial_stop_distance(cfg, a, entry, line_dist)
    lots, _ = risk_lots(equity, 1.0, a, entry, cfg, dist)
    stop = entry - side * dist
    target = entry + side * cfg.target_r * dist if cfg.target_r else None
    return {
        "side": "long" if side > 0 else "short", "entry": round(entry, 2), "stop": round(stop, 2),
        "stop_pct": round(100 * dist / entry, 3), "target": round(target, 2) if target else None,
        "target_pct": round(100 * abs(target - entry) / entry, 3) if target else None,
        "risk_reward": cfg.target_r, "lots": lots, "risk_usd": round(lots * dist * contract_size, 2),
        "trails": line is not None,
    }


STOP_PRESETS: dict[str, dict] = {
    "trail": {"stop_atr": None, "stop_pct": None, "use_line": True},
    "pct_r": {"stop_atr": None, "stop_pct": 0.04, "target_r": 7.0, "use_line": False},
    "pct_be": {"stop_atr": None, "stop_pct": 0.04, "breakeven_r": 1.0, "use_line": True},
    "atr": {"stop_atr": 2.5, "stop_pct": None, "use_line": True},
    # Swing high / low plus half an ATR, 7R target: the friend's position tool on BTC.
    "swing_r": {"stop_atr": None, "stop_pct": None, "target_r": 7.0, "use_line": False, "use_swing": True},
}


def swing_config(stop_mode: str = "trail", risk_per_trade: float = 0.01, **over) -> tuple[BacktestConfig, bool]:
    preset = dict(STOP_PRESETS[stop_mode])
    use_line = preset.pop("use_line")
    preset.pop("use_swing", None)
    base = {"risk_per_trade": risk_per_trade, "daily_loss_limit": 1.0, "max_drawdown_kill": 0.35}
    cfg = BacktestConfig(**{**base, **preset, **over})
    return cfg, use_line


def backtest_swing(md: MarketData, p: SwingParams | None = None, stop_mode: str = "trail",
                   timeframe: str = "D", **cfg_over) -> tuple[BacktestResult, pd.DataFrame]:
    p = p or SwingParams()
    dmd = daily_market(md, timeframe)
    sig, line, comp = swing_signal(dmd.bars, p)
    cfg, use_line = swing_config(stop_mode, **cfg_over)
    stops = swing_stops(dmd.bars) if STOP_PRESETS[stop_mode].get("use_swing") else None
    res = run_backtest(dmd, sig, cfg, stop_line=line if use_line else None, entry_stops=stops)
    return res, comp
