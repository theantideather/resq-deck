"""Bar by bar backtester for XAUUSD with realistic costs and risk controls.

Execution model
    A signal computed on bar t's close is traded at bar t+1's open. Stops and
    targets are checked against each later bar's high and low; if both are
    touched in the same bar the stop is assumed to fill first. A bar that
    gaps through a stop fills at its open, not at the stop price.

Costs, from `GoldInstrument`
    half spread plus slippage on every fill (spread widened around NFP/FOMC
    and at the 17:00 New York rollover), commission per lot, and overnight
    swap at each rollover with the Wednesday triple charge.

Sizing
    Each entry risks `risk_per_trade` of equity between entry and the ATR
    stop, scaled by the signal's conviction |signal| in (0, 1], rounded down
    to the broker's lot step and capped by margin.

Risk controls
    daily loss limit (flat for the rest of the trading day), max drawdown kill
    switch (flat for good), optional time stop, and a re-entry lock after a
    stop out until the signal changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import calendar as cal
from .data import MarketData
from .features import atr as _atr
from .instrument import XAUUSD, GoldInstrument


@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005
    stop_atr: float | None = 2.0
    target_atr: float | None = None
    atr_period: int = 14
    max_bars_in_trade: int | None = None
    daily_loss_limit: float = 0.03
    max_drawdown_kill: float = 0.25
    max_margin_utilisation: float = 0.5
    event_spread_window: tuple[str, str] = ("30min", "30min")
    instrument: GoldInstrument = field(default_factory=lambda: XAUUSD)
    # Swing trading stop rules (all optional):
    stop_pct: float | None = None       # fixed stop, fraction of entry price (0.04 = 4%); overrides stop_atr
    target_r: float | None = None       # take profit at this many R (multiples of the initial stop distance)
    breakeven_r: float | None = None    # move the stop to entry once price has gone this many R in favour


@dataclass
class Trade:
    entry_time: pd.Timestamp
    side: int
    lots: float
    entry_price: float
    stop: float | None
    target: float | None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""
    gross_pnl: float = 0.0
    costs: float = 0.0
    swap: float = 0.0
    risk_usd: float = 0.0
    init_dist: float = 0.0
    best: float = float("nan")

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs + self.swap

    @property
    def r_multiple(self) -> float:
        return self.net_pnl / self.risk_usd if self.risk_usd else np.nan


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    stats: dict
    config: BacktestConfig

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"{'Net return':<22}{s['total_return']:>10.1%}",
            f"{'CAGR':<22}{s['cagr']:>10.1%}",
            f"{'Sharpe':<22}{s['sharpe']:>10.2f}",
            f"{'Sortino':<22}{s['sortino']:>10.2f}",
            f"{'Max drawdown':<22}{s['max_drawdown']:>10.1%}",
            f"{'Calmar':<22}{s['calmar']:>10.2f}",
            f"{'Trades':<22}{s['trades']:>10d}",
            f"{'Win rate':<22}{s['win_rate']:>10.1%}",
            f"{'Profit factor':<22}{s['profit_factor']:>10.2f}",
            f"{'Avg R per trade':<22}{s['avg_r']:>10.2f}",
            f"{'Time in market':<22}{s['exposure']:>10.1%}",
            f"{'Spread+slip+comm':<22}{s['costs']:>10,.0f}",
            f"{'Swap':<22}{s['swap']:>10,.0f}",
        ]
        return "\n".join(lines)


def risk_lots(equity: float, conviction: float, atr_value: float, price: float,
              cfg: BacktestConfig, stop_dist: float | None = None) -> tuple[float, float]:
    """Lots for a new entry and the stop distance used to size it.

    Risks `risk_per_trade * conviction` of equity between entry and an ATR
    stop, capped by margin and rounded down to the broker's lot step. Without
    a stop, sizes as if the stop were 2 ATR away. The backtester and the live
    runner both call this, so they size identically.
    """
    ins = cfg.instrument
    if stop_dist is None:
        stop_dist = initial_stop_distance(cfg, atr_value, price)
    if not np.isfinite(stop_dist) or stop_dist <= 0 or conviction <= 0:
        return 0.0, float(stop_dist)
    lots = equity * cfg.risk_per_trade * conviction / (stop_dist * ins.contract_size_oz)
    max_lots = equity * cfg.max_margin_utilisation / ins.margin_required(1.0, price)
    return ins.round_lots(min(lots, max_lots)), float(stop_dist)


def initial_stop_distance(cfg: BacktestConfig, atr_value: float, price: float,
                          line_dist: float | None = None) -> float:
    """Distance from entry to the first stop: fixed %, else ATR multiple, else the
    trailing line, else 2 ATR."""
    if cfg.stop_pct:
        return price * cfg.stop_pct
    if cfg.stop_atr:
        return cfg.stop_atr * atr_value
    if line_dist is not None and np.isfinite(line_dist) and line_dist > 0:
        return line_dist
    return 2.0 * atr_value


def run_backtest(
    md: MarketData,
    signal: pd.Series,
    config: BacktestConfig | None = None,
    stop_line: pd.Series | None = None,
    entry_stops: pd.DataFrame | None = None,
) -> BacktestResult:
    """`stop_line` (optional) is a trailing level such as a Supertrend or HalfTrend
    line. Its value at bar t-1 becomes the stop for bar t, ratcheting only in
    the trade's favour; with no stop_pct or stop_atr it also sets the first stop.

    `entry_stops` (optional) has columns long and short: absolute stop prices
    known at each bar's close (e.g. swing low minus half an ATR). When given,
    the first stop of a trade opened at bar t is taken from bar t-1."""
    cfg = config or BacktestConfig()
    ins = cfg.instrument
    bars = md.bars
    idx = bars.index
    n = len(bars)
    sig = signal.reindex(idx).fillna(0.0).clip(-1, 1).to_numpy()
    line = stop_line.reindex(idx).to_numpy(dtype=float) if stop_line is not None else None
    es_long = entry_stops["long"].reindex(idx).to_numpy(dtype=float) if entry_stops is not None else None
    es_short = entry_stops["short"].reindex(idx).to_numpy(dtype=float) if entry_stops is not None else None
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = _atr(bars, cfg.atr_period).to_numpy()
    ev = cal.event_risk(idx, md.events, *cfg.event_spread_window).to_numpy()
    roll = cal.is_rollover(idx).to_numpy()
    day = np.asarray(cal.trading_day(idx))
    oz = ins.contract_size_oz

    equity = np.empty(n)
    pos_lots = np.zeros(n)
    cash = cfg.initial_equity
    peak = cash
    day_start_equity = cash
    halted_today = False
    killed = False
    locked_side = 0  # side we were stopped out of, blocked until the signal changes
    trade: Trade | None = None
    trades: list[Trade] = []

    def fill_cost(lots: float, i: int) -> float:
        spread = ins.spread_at(event_risk=bool(ev[i]), rollover=bool(roll[i]))
        return lots * oz * (spread / 2 + ins.slippage_usd) + lots * ins.commission_per_lot / 2

    def close_trade(i: int, price: float, reason: str) -> None:
        nonlocal cash, trade
        t = trade
        t.exit_time = idx[i]
        t.exit_price = price
        t.reason = reason
        t.gross_pnl = t.side * (price - t.entry_price) * t.lots * oz
        exit_cost = fill_cost(t.lots, i)
        t.costs += exit_cost
        cash += t.gross_pnl - exit_cost
        trades.append(t)
        trade = None

    for i in range(n):
        # 1. New trading day: swap on open positions, reset the daily limit.
        if i > 0 and day[i] != day[i - 1]:
            if trade is not None:
                nights = 3 if pd.Timestamp(day[i - 1]).weekday() == ins.triple_swap_weekday else 1
                rate = ins.swap_long_per_lot if trade.side > 0 else ins.swap_short_per_lot
                sw = rate * trade.lots * nights
                trade.swap += sw
                cash += sw
            mtm = cash + (trade.side * (c[i - 1] - trade.entry_price) * trade.lots * oz if trade else 0.0)
            day_start_equity = mtm
            halted_today = False

        # 2. Act at this bar's open on the previous bar's signal.
        if i > 0 and not killed:
            want = int(np.sign(sig[i - 1]))
            if locked_side and want != locked_side:
                locked_side = 0
            if halted_today:
                want = 0
            if trade is not None and want != trade.side:
                close_trade(i, o[i], "signal")
            if trade is None and want != 0 and want != locked_side and np.isfinite(a[i - 1]):
                line_dist = want * (o[i] - line[i - 1]) if line is not None else None
                if es_long is not None:
                    level = es_long[i - 1] if want > 0 else es_short[i - 1]
                    sd = want * (o[i] - level)
                    dist = sd if np.isfinite(sd) and sd > 0 else 2.0 * a[i - 1]
                else:
                    dist = initial_stop_distance(cfg, a[i - 1], o[i], line_dist)
                lots, stop_dist = risk_lots(cash, abs(sig[i - 1]), a[i - 1], o[i], cfg, dist)
                if lots > 0:
                    has_stop = bool(cfg.stop_pct or cfg.stop_atr or line is not None or es_long is not None)
                    stop = o[i] - want * stop_dist if has_stop else None
                    if cfg.target_r:
                        target = o[i] + want * cfg.target_r * stop_dist
                    else:
                        target = o[i] + want * cfg.target_atr * a[i - 1] if cfg.target_atr else None
                    trade = Trade(idx[i], want, lots, o[i], stop, target, risk_usd=lots * stop_dist * oz,
                                  init_dist=stop_dist, best=o[i])
                    entry_cost = fill_cost(lots, i)
                    trade.costs += entry_cost
                    cash -= entry_cost
                    entry_bar = i

        # 3. Stops, targets and time stop inside this bar.
        if trade is not None and i > entry_bar:
            # Ratchet the stop with what was known at the previous close.
            s = trade.side
            trade.best = max(trade.best, h[i - 1]) if s > 0 else min(trade.best, l[i - 1])
            if line is not None and np.isfinite(line[i - 1]) and trade.stop is not None:
                trade.stop = max(trade.stop, line[i - 1]) if s > 0 else min(trade.stop, line[i - 1])
            if cfg.breakeven_r and trade.stop is not None and s * (trade.best - trade.entry_price) >= cfg.breakeven_r * trade.init_dist:
                trade.stop = max(trade.stop, trade.entry_price) if s > 0 else min(trade.stop, trade.entry_price)
        if trade is not None:
            s = trade.side
            hit_stop = trade.stop is not None and ((s > 0 and l[i] <= trade.stop) or (s < 0 and h[i] >= trade.stop))
            hit_tgt = trade.target is not None and ((s > 0 and h[i] >= trade.target) or (s < 0 and l[i] <= trade.target))
            if hit_stop:
                gapped = (s > 0 and o[i] < trade.stop) or (s < 0 and o[i] > trade.stop)
                close_trade(i, o[i] if gapped else trade.stop, "stop")
                locked_side = s
            elif hit_tgt:
                gapped = (s > 0 and o[i] > trade.target) or (s < 0 and o[i] < trade.target)
                close_trade(i, o[i] if gapped else trade.target, "target")
                locked_side = s
            elif cfg.max_bars_in_trade and i - entry_bar >= cfg.max_bars_in_trade:
                close_trade(i, c[i], "time")
                locked_side = s

        # 4. Mark to market and risk limits at the close.
        open_pnl = trade.side * (c[i] - trade.entry_price) * trade.lots * oz if trade else 0.0
        eq = cash + open_pnl
        peak = max(peak, eq)
        if not halted_today and eq <= day_start_equity * (1 - cfg.daily_loss_limit):
            halted_today = True
            if trade is not None:
                close_trade(i, c[i], "daily_limit")
                eq = cash
        if not killed and eq <= peak * (1 - cfg.max_drawdown_kill):
            killed = True
            if trade is not None:
                close_trade(i, c[i], "kill_switch")
                eq = cash
        equity[i] = eq
        pos_lots[i] = trade.side * trade.lots if trade else 0.0

    if trade is not None:
        close_trade(n - 1, c[-1], "end")
        equity[-1] = cash

    eq_s = pd.Series(equity, index=idx, name="equity")
    pos_s = pd.Series(pos_lots, index=idx, name="lots")
    tdf = trades_frame(trades)
    return BacktestResult(eq_s, pos_s, tdf, compute_stats(eq_s, pos_s, tdf, md.freq_hours), cfg)


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    cols = ["entry_time", "exit_time", "side", "lots", "entry_price", "exit_price", "reason",
            "gross_pnl", "costs", "swap", "net_pnl", "r_multiple", "initial_stop", "target"]
    rows = [
        {**{k: getattr(t, k) for k in cols if k not in ("net_pnl", "r_multiple", "initial_stop")},
         "net_pnl": t.net_pnl, "r_multiple": t.r_multiple,
         "initial_stop": t.entry_price - t.side * t.init_dist if t.init_dist else t.stop}
        for t in trades
    ]
    return pd.DataFrame(rows, columns=cols)


def compute_stats(equity: pd.Series, lots: pd.Series, trades: pd.DataFrame, freq_hours: float) -> dict:
    bars_per_year = 24 * 5.2 * 52 / freq_hours
    r = equity.pct_change().fillna(0.0)
    years = max(len(equity) / bars_per_year, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    sd = r.std()
    downside = r[r < 0].std()
    sharpe = r.mean() / sd * np.sqrt(bars_per_year) if sd > 0 else 0.0
    sortino = r.mean() / downside * np.sqrt(bars_per_year) if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    mdd = dd.min()
    wins = trades["net_pnl"][trades["net_pnl"] > 0] if len(trades) else pd.Series(dtype=float)
    losses = trades["net_pnl"][trades["net_pnl"] <= 0] if len(trades) else pd.Series(dtype=float)
    pf = wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else (np.inf if len(wins) else 0.0)
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(sd * np.sqrt(bars_per_year)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(cagr / -mdd) if mdd < 0 else 0.0,
        "trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
        "profit_factor": float(pf),
        "avg_r": float(trades["r_multiple"].mean()) if len(trades) else 0.0,
        "exposure": float((lots != 0).mean()),
        "costs": float(trades["costs"].sum()) if len(trades) else 0.0,
        "swap": float(trades["swap"].sum()) if len(trades) else 0.0,
        "years": float(years),
    }
