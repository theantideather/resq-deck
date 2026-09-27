"""Feature engineering for gold.

Every feature at bar t uses only information available at the close of bar t.
Macro columns arrive already lagged to their publication time (see
`data.align_to_bars`), and the fair value model is refit on a trailing window
and applied forward, never fit on the data it scores.

Groups:
    price      returns, realized vol, ATR, RSI, trend distance, efficiency
    session    hour, session flags, Asian range and London breakout distance,
               LBMA fix proximity, event proximity
    macro      dollar and real yield moves, VIX, gold/silver ratio, CFTC
               managed money positioning, beta adjusted residual vs the dollar,
               macro fair value gap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import calendar as cal
from .data import MarketData


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = bars["close"].shift()
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().rename("atr")


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman: net move over total path. Near 1 is a clean trend, near 0 is chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).rename(f"er_{n}")


def zscore(s: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    m = s.rolling(n, min_periods=min_periods or n // 2)
    return (s - m.mean()) / m.std().replace(0, np.nan)


def rolling_beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    cov = y.rolling(n, min_periods=n // 2).cov(x)
    var = x.rolling(n, min_periods=n // 2).var()
    return cov / var.replace(0, np.nan)


def asian_range(bars: pd.DataFrame) -> pd.DataFrame:
    """Running high and low of the current trading day's Asian session.

    During Asia the values grow bar by bar; after Asia they are fixed for the
    rest of the day. Bars before the day's Asian session starts get NaN.
    """
    sess = cal.session_flags(bars.index)
    day = cal.trading_day(bars.index)
    hi = bars["high"].where(sess["asia"])
    lo = bars["low"].where(sess["asia"])
    g_hi = hi.groupby(day.to_numpy()).cummax()
    g_lo = lo.groupby(day.to_numpy()).cummin()
    out = pd.DataFrame({"asia_high": g_hi, "asia_low": g_lo}, index=bars.index)
    return out.groupby(day.to_numpy()).ffill()


def rolling_fair_value(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
    refit_every: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk forward OLS of y on X. Returns (fitted value, coefficients).

    Coefficients are estimated on the `window` bars that end at a refit point
    and then used, unchanged, for the next `refit_every` bars. No bar is ever
    scored by a model that saw it.
    """
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()
    fitted = pd.Series(np.nan, index=y.index)
    coefs = pd.DataFrame(np.nan, index=y.index, columns=["const", *X.columns])
    if len(data) <= window:
        return fitted, coefs
    Y = data["_y"].to_numpy()
    A = np.column_stack([np.ones(len(data)), data[X.columns].to_numpy()])
    pos = y.index.get_indexer(data.index)
    for start in range(window, len(data), refit_every):
        sl = slice(start - window, start)
        beta, *_ = np.linalg.lstsq(A[sl], Y[sl], rcond=None)
        apply = slice(start, min(start + refit_every, len(data)))
        fitted.iloc[pos[apply]] = A[apply] @ beta
        coefs.iloc[pos[apply]] = beta
    return fitted, coefs


def macro_fair_value(md: MarketData, window: int = 2000, refit_every: int = 24) -> pd.DataFrame:
    """Gold's macro fair value from the dollar and real yields, in the spirit of
    the World Gold Council's GRAM attribution: log(gold) ~ log(DXY) + real yield.

    Returns fair (log price), gap (log price minus fair) and gap_z.
    """
    cols = [c for c in ("dxy", "real_yield") if c in md.macro and md.macro[c].notna().any()]
    if not cols:
        return pd.DataFrame(index=md.bars.index, columns=["fair", "gap", "gap_z"], dtype=float)
    X = pd.DataFrame(index=md.bars.index)
    if "dxy" in cols:
        X["log_dxy"] = np.log(md.macro["dxy"])
    if "real_yield" in cols:
        X["real_yield"] = md.macro["real_yield"]
    y = np.log(md.bars["close"])
    fair, _ = rolling_fair_value(y, X, window, refit_every)
    gap = y - fair
    return pd.DataFrame({"fair": fair, "gap": gap, "gap_z": zscore(gap, window // 2)})


# ---------------------------------------------------------------------------
# The feature matrix
# ---------------------------------------------------------------------------

def build_features(md: MarketData, fair_window: int = 2000) -> pd.DataFrame:
    b = md.bars
    c = b["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=b.index)

    # Price and volatility.
    for n in (1, 4, 24, 120):
        f[f"ret_{n}"] = lc.diff(n)
    r1 = lc.diff()
    f["rv_24"] = r1.rolling(24).std()
    f["rv_120"] = r1.rolling(120).std()
    f["vol_ratio"] = f["rv_24"] / f["rv_120"]
    a = atr(b)
    f["atr_pct"] = a / c
    f["rsi_14"] = rsi(c)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    f["dist_ema50_atr"] = (c - ema50) / a
    f["dist_ema200_atr"] = (c - ema200) / a
    f["ema20_slope"] = ema20.pct_change(5)
    f["er_24"] = efficiency_ratio(c, 24)
    hi120 = b["high"].rolling(120).max()
    lo120 = b["low"].rolling(120).min()
    f["donchian_pos"] = (c - lo120) / (hi120 - lo120).replace(0, np.nan)
    f["range_atr"] = (b["high"] - b["low"]) / a
    f["volume_z"] = zscore(b["volume"].astype(float), 120)

    # Clock.
    sess = cal.session_flags(b.index)
    for col in sess.columns:
        f[f"sess_{col}"] = sess[col].astype(float)
    hour = b.index.hour + b.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = b.index.dayofweek.astype(float)
    f["near_fix"] = cal.near_lbma_fix(b.index).astype(float)
    f["hours_to_event"] = cal.hours_to_next_event(b.index, md.events)
    f["event_window"] = cal.event_risk(b.index, md.events, "2h", "2h").astype(float)
    ar = asian_range(b)
    f["asia_range_atr"] = (ar["asia_high"] - ar["asia_low"]) / a
    f["dist_asia_high_atr"] = (c - ar["asia_high"]) / a
    f["dist_asia_low_atr"] = (c - ar["asia_low"]) / a

    # Macro and cross asset.
    m = md.macro
    if "dxy" in m and m["dxy"].notna().any():
        ldxy = np.log(m["dxy"])
        f["dxy_ret_24"] = ldxy.diff(24)
        f["dxy_ret_120"] = ldxy.diff(120)
        beta = rolling_beta(r1, ldxy.diff(), 500)
        f["beta_dxy"] = beta
        # Gold's move after stripping the dollar: positive means gold is strong
        # in its own right (central bank or safe haven demand), not just a weak dollar.
        f["resid_vs_dxy_24"] = (r1 - beta * ldxy.diff()).rolling(24).sum()
    if "real_yield" in m and m["real_yield"].notna().any():
        f["ry_chg_24"] = m["real_yield"].diff(24)
        f["ry_chg_120"] = m["real_yield"].diff(120)
        f["ry_level"] = m["real_yield"]
    if "breakeven" in m and m["breakeven"].notna().any():
        f["be_chg_120"] = m["breakeven"].diff(120)
    if "vix" in m and m["vix"].notna().any():
        f["vix_z"] = zscore(m["vix"], 500)
    if "silver" in m and m["silver"].notna().any():
        f["gsr_z"] = zscore(c / m["silver"], 2000)
    if "cot_mm_net" in m and m["cot_mm_net"].notna().any():
        bars_per_week = max(1, round(24 / md.freq_hours)) * 5
        f["cot_mm_z"] = zscore(m["cot_mm_net"], bars_per_week * 104, min_periods=bars_per_week * 26)

    fv = macro_fair_value(md, window=fair_window)
    f["fv_gap"] = fv["gap"]
    f["fv_gap_z"] = fv["gap_z"]
    return f.replace([np.inf, -np.inf], np.nan)
