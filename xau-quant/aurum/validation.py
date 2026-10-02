"""Tests a backtest has to pass before anyone risks money on it.

The XAUUSD bot market is full of "Sharpe 4.8, 2% drawdown" claims. Almost all
of them come from trying many variants on one history and reporting the best.
These tools put a number on that:

    probabilistic_sharpe   P(true Sharpe > benchmark) given sample length,
                           skew and fat tails (Bailey and Lopez de Prado 2012)
    deflated_sharpe        the same, after correcting for how many strategy
                           variants were tried (Bailey and Lopez de Prado 2014)
    bootstrap_trades       resample trades to get the drawdown you should
                           plan for, not the one history happened to deal
    cost_stress            rerun with spread, slippage and swap multiplied;
                           an edge that dies at 1.5x costs is not an edge
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _st  # scipy ships with scikit-learn

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .data import MarketData

EULER_GAMMA = 0.5772156649


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna().pct_change().dropna()


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """PSR on per period returns. benchmark_sr is per period, not annualised."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew = _st.skew(r)
    kurt = _st.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(_st.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best Sharpe among n_trials skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = _st.norm.ppf(1 - 1 / n_trials)
    z2 = _st.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """DSR: PSR against the Sharpe you would expect from luck alone after n_trials.

    `trial_sharpes` are the per period Sharpe ratios of every variant you
    tried; if omitted their variance is approximated by 1 / sample length.
    Above 0.95 is good evidence of skill. Report n_trials honestly.
    """
    r = returns.dropna()
    var = np.var(trial_sharpes, ddof=1) if trial_sharpes and len(trial_sharpes) > 1 else 1 / max(len(r), 1)
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var))


def bootstrap_trades(trades: pd.DataFrame, initial_equity: float, n_paths: int = 2000,
                     seed: int = 0) -> dict:
    """Resample trade P&L with replacement; return drawdown and return percentiles."""
    pnl = trades["net_pnl"].to_numpy()
    if len(pnl) < 5:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(pnl, size=(n_paths, len(pnl)), replace=True)
    eq = initial_equity + np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.full((n_paths, 1), initial_equity), eq], axis=1), axis=1)[:, 1:]
    mdd = (eq / peak - 1).min(axis=1)
    ret = eq[:, -1] / initial_equity - 1
    return {
        "max_dd_median": float(np.median(mdd)),
        "max_dd_p95": float(np.percentile(mdd, 5)),
        "return_p5": float(np.percentile(ret, 5)),
        "return_median": float(np.median(ret)),
        "prob_loss": float((ret < 0).mean()),
    }


def cost_stress(md: MarketData, signal: pd.Series, config: BacktestConfig,
                multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
                stop_line: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    ins = config.instrument
    for m in multipliers:
        stressed = ins.with_overrides(
            spread_usd=ins.spread_usd * m,
            slippage_usd=ins.slippage_usd * m,
            commission_per_lot=ins.commission_per_lot * m,
            swap_long_per_lot=ins.swap_long_per_lot * m,
            swap_short_per_lot=ins.swap_short_per_lot if ins.swap_short_per_lot < 0 else ins.swap_short_per_lot / m,
        )
        cfg = BacktestConfig(**{**config.__dict__, "instrument": stressed})
        st = run_backtest(md, signal, cfg, stop_line=stop_line).stats
        rows.append({"cost_x": m, "return": st["total_return"], "sharpe": st["sharpe"],
                     "max_dd": st["max_drawdown"], "profit_factor": st["profit_factor"]})
    return pd.DataFrame(rows).set_index("cost_x")


@dataclass
class Scorecard:
    sharpe: float
    psr: float
    dsr: float
    n_trials: int
    bootstrap: dict
    cost_table: pd.DataFrame | None
    verdict: str

    def text(self) -> str:
        lines = [
            f"{'Sharpe (annual)':<26}{self.sharpe:>8.2f}",
            f"{'P(Sharpe > 0)':<26}{self.psr:>8.1%}",
            f"{'Deflated Sharpe':<26}{self.dsr:>8.1%}   ({self.n_trials} trials)",
        ]
        if self.bootstrap:
            b = self.bootstrap
            lines += [
                f"{'Max DD, median path':<26}{b['max_dd_median']:>8.1%}",
                f"{'Max DD, 1 in 20 path':<26}{b['max_dd_p95']:>8.1%}",
                f"{'P(losing money)':<26}{b['prob_loss']:>8.1%}",
            ]
        if self.cost_table is not None:
            lines.append("Cost stress (return / Sharpe):")
            for m, row in self.cost_table.iterrows():
                lines.append(f"  {m:>4.1f}x costs  {row['return']:>8.1%}  {row['sharpe']:>6.2f}")
        lines.append(f"Verdict: {self.verdict}")
        return "\n".join(lines)


def scorecard(md: MarketData, signal: pd.Series, result: BacktestResult, n_trials: int = 1,
              stress: bool = True, stop_line: pd.Series | None = None) -> Scorecard:
    daily = _daily_returns(result.equity)
    psr = probabilistic_sharpe(daily)
    dsr = deflated_sharpe(daily, n_trials)
    boot = bootstrap_trades(result.trades, result.config.initial_equity)
    table = cost_stress(md, signal, result.config, stop_line=stop_line) if stress else None
    survives = table is None or table.loc[1.5, "sharpe"] > 0.3
    if result.stats["trades"] < 30:
        verdict = "not enough trades to judge"
    elif dsr >= 0.95 and survives:
        verdict = "passes: evidence of an edge after costs and multiple testing; paper trade next"
    elif psr >= 0.9 and survives:
        verdict = "promising, but not significant once the number of trials is counted"
    else:
        verdict = "fails: indistinguishable from luck or killed by costs"
    return Scorecard(result.stats["sharpe"], psr, dsr, n_trials, boot, table, verdict)
