"""Gold strategies. Each turns MarketData plus features into a signal in [-1, 1].

A signal is the position you want after the bar closes: its sign is the side
and its magnitude is the conviction that scales risk. Zero means flat.

    london_breakout   Asian range breakout in the first hours of London, the
                      classic XAUUSD intraday setup: Asia builds a range on
                      physical flow, London's open picks a direction.
    trend             Time series momentum on one to five month horizons, with a
                      hysteresis band so it does not churn. Gold trends for
                      months when central banks or real yields drive it.
    macro_reversion   Fade gaps between gold and its dollar plus real yield fair
                      value (the GRAM idea from the World Gold Council, made
                      tradable with a walk forward fit).
    ensemble          Blend of the above, weighted by the causal HMM regime.

Each strategy also carries the backtest settings (stops, targets, time
stops) that match how it is meant to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import calendar as cal
from .backtest import BacktestConfig
from .data import MarketData
from .regime import detect_regimes


def hysteresis(z: pd.Series, enter: float, exit: float, fade: bool = True) -> pd.Series:
    """Stateful band signal. Enters beyond +/-enter, exits once back inside +/-exit.

    With fade=True a high z gives a short (mean reversion); False follows it.
    """
    vals = z.to_numpy()
    out = np.zeros(len(vals))
    state = 0
    for i, v in enumerate(vals):
        if np.isnan(v):
            out[i] = state
            continue
        if state == 0:
            if v > enter:
                state = -1 if fade else 1
            elif v < -enter:
                state = 1 if fade else -1
        elif abs(v) < exit:
            state = 0
        out[i] = state
    return pd.Series(out, index=z.index)


def flat_into_news(signal: pd.Series, md: MarketData, before: str = "30min", after: str = "15min") -> pd.Series:
    """Force flat around scheduled NFP and FOMC releases."""
    risk = cal.event_risk(signal.index, md.events, before, after)
    return signal.where(~risk, 0.0)


# ---------------------------------------------------------------------------

def london_breakout(md: MarketData, feats: pd.DataFrame, buffer_atr: float = 0.1,
                    max_range_atr: float = 6.0, entry_hours: tuple[float, float] = (8.0, 11.0),
                    exit_ny_hour: float = 16.0) -> pd.Series:
    idx = md.bars.index
    ld = idx.tz_convert(cal.LONDON)
    ld_hour = ld.hour + ld.minute / 60
    ny = idx.tz_convert(cal.NEW_YORK)
    ny_hour = ny.hour + ny.minute / 60
    day = np.asarray(cal.trading_day(idx))

    in_entry = (ld_hour >= entry_hours[0]) & (ld_hour < entry_hours[1])
    ok_range = feats["asia_range_atr"] <= max_range_atr
    up = (feats["dist_asia_high_atr"] > buffer_atr) & in_entry & ok_range
    dn = (feats["dist_asia_low_atr"] < -buffer_atr) & in_entry & ok_range
    trig = pd.Series(np.where(up, 1.0, np.where(dn, -1.0, np.nan)), index=idx)

    # First trigger of the day holds until the New York afternoon.
    first = trig.groupby(day).transform(lambda s: s.ffill())
    first_side = trig.groupby(day).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    active = first.notna() & (ny_hour < exit_ny_hour) & ~((ny.hour >= 17) | (ny_hour < 3))
    sig = pd.Series(np.where(active, first_side, 0.0), index=idx).fillna(0.0)
    return flat_into_news(sig, md)


def trend(md: MarketData, feats: pd.DataFrame, fast_days: int = 20, slow_days: int = 100,
          enter: float = 0.5, exit: float = 0.1) -> pd.Series:
    """Time series momentum. Defaults follow the 1 to 5 month horizons where the
    literature (Moskowitz, Ooi and Pedersen 2012; Hurst, Ooi and Pedersen 2017)
    finds trend in gold, expressed in bars of whatever frequency the data has."""
    bars_per_day = max(1, round(24 / md.freq_hours))
    fast, slow = fast_days * bars_per_day, slow_days * bars_per_day
    c = md.bars["close"]
    spread = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    vol = c.diff().rolling(slow).std()
    strength = spread / (vol * np.sqrt(fast))
    # Enter on a strong reading, hold until it has nearly faded. Without the
    # band the position flickers on and off around the threshold and pays the
    # spread each time.
    side = hysteresis(strength, enter, exit, fade=False)
    conviction = (strength.abs() / 2).clip(0.25, 1.0)
    return (side * conviction).fillna(0.0)


def macro_reversion(md: MarketData, feats: pd.DataFrame, enter: float = 1.75, exit: float = 0.25) -> pd.Series:
    z = feats["fv_gap_z"]
    if z.notna().sum() == 0:
        return pd.Series(0.0, index=md.bars.index)
    sig = hysteresis(z, enter, exit, fade=True)
    # Do not fade gold while the dollar is moving hard in the same direction.
    if "dxy_ret_120" in feats:
        dxy_z = (feats["dxy_ret_120"] / feats["dxy_ret_120"].rolling(2000, min_periods=500).std())
        against = ((sig > 0) & (dxy_z > 2)) | ((sig < 0) & (dxy_z < -2))
        sig = sig.where(~against, 0.0)
    return flat_into_news(sig, md)


# Regime weights: which strategy to trust in which volatility state.
REGIME_WEIGHTS = {
    "calm": {"london_breakout": 0.3, "trend": 0.3, "macro_reversion": 0.4},
    "normal": {"london_breakout": 0.3, "trend": 0.5, "macro_reversion": 0.2},
    "stressed": {"london_breakout": 0.0, "trend": 0.7, "macro_reversion": 0.0},
}


def ensemble(md: MarketData, feats: pd.DataFrame, regimes: pd.DataFrame | None = None,
             components: dict[str, pd.Series] | None = None, threshold: float = 0.15) -> pd.Series:
    if regimes is None:
        regimes = detect_regimes(md.bars["close"])
    if components is None:
        components = {name: STRATEGIES[name].signal(md, feats) for name in REGIME_WEIGHTS["calm"]}
    idx = md.bars.index
    probs = regimes.filter(like="p_").fillna(0.0)
    total = pd.Series(0.0, index=idx)
    for reg, weights in REGIME_WEIGHTS.items():
        p = probs.get(f"p_{reg}", pd.Series(0.0, index=idx))
        for name, w in weights.items():
            total += p * w * components[name].reindex(idx).fillna(0.0)
    sig = total.where(total.abs() >= threshold, 0.0)
    return (sig / sig.abs().max() if sig.abs().max() > 0 else sig).clip(-1, 1)


@dataclass
class Strategy:
    name: str
    description: str
    signal: Callable[[MarketData, pd.DataFrame], pd.Series]
    config: BacktestConfig = field(default_factory=BacktestConfig)


STRATEGIES: dict[str, Strategy] = {
    "london_breakout": Strategy(
        "london_breakout",
        "Asian range breakout at the London open, flat by the New York afternoon",
        london_breakout,
        BacktestConfig(stop_atr=1.5, target_atr=3.0, max_bars_in_trade=12),
    ),
    "trend": Strategy(
        "trend",
        "One to five month time series momentum",
        trend,
        BacktestConfig(stop_atr=25.0, risk_per_trade=0.01),
    ),
    "macro_reversion": Strategy(
        "macro_reversion",
        "Fade gaps from the DXY plus real yield fair value",
        macro_reversion,
        BacktestConfig(stop_atr=5.0, max_bars_in_trade=240, risk_per_trade=0.0075),
    ),
}
STRATEGIES["ensemble"] = Strategy(
    "ensemble",
    "Regime weighted blend of breakout, trend and macro reversion",
    ensemble,
    BacktestConfig(stop_atr=5.0, risk_per_trade=0.01),
)
