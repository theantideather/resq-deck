"""Parameter search that tells the truth about itself.

    walk_forward_optimize   pick parameters on a trailing window, trade them on the
                            next one, stitch the out of sample pieces together.
                            The trial count feeds the deflated Sharpe.
    pbo_cscv                probability of backtest overfitting (Bailey, Borwein,
                            Lopez de Prado and Zhu, 2017): how often the in sample
                            winner lands in the bottom half out of sample.
    reality_check           White's (2000) reality check with the Politis and Romano
                            stationary bootstrap: is the best of N strategies better
                            than zero once you account for having looked at N?
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .data import MarketData


def _sharpe(r: np.ndarray, periods: float) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


@dataclass
class WFOResult:
    oos_returns: pd.Series           # stitched out of sample daily returns
    oos_equity: pd.Series
    chosen: pd.DataFrame             # per window: dates and the parameters picked
    returns_matrix: pd.DataFrame     # daily returns of every combo over the full sample
    combos: list[dict]
    n_trials: int

    def sharpe(self) -> float:
        return _sharpe(self.oos_returns.to_numpy(), 252)


def walk_forward_optimize(
    md: MarketData,
    feats: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    grid: dict[str, list],
    config: BacktestConfig,
    train_days: int = 365,
    test_days: int = 91,
    metric: Callable[[np.ndarray], float] | None = None,
) -> WFOResult:
    """Walk forward optimisation over `grid` (kwargs for signal_fn).

    Every combo is backtested once over the whole sample (signals are causal,
    so that is safe), giving a matrix of daily returns. Each window then picks
    the combo with the best train metric and takes that combo's returns for
    the following test window. Positions are not carried across a window
    switch, which slightly understates switching costs.
    """
    combos = param_grid(grid)
    metric = metric or (lambda r: _sharpe(r, 252))
    # Account kill switches belong to live trading. In a parameter search they
    # freeze a candidate's equity after a drawdown and make it look safer
    # than it is, so they are switched off here.
    config = BacktestConfig(**{**config.__dict__, "max_drawdown_kill": 1.0, "daily_loss_limit": 1.0})
    cols = {}
    for i, kw in enumerate(combos):
        sig = signal_fn(md, feats, **kw)
        eq = run_backtest(md, sig, config).equity
        cols[i] = eq.resample("1D").last().dropna().pct_change().fillna(0.0)
    R = pd.DataFrame(cols).fillna(0.0)
    days = R.index
    out, rows = [], []
    start = days[0] + pd.Timedelta(days=train_days)
    while start < days[-1]:
        end = start + pd.Timedelta(days=test_days)
        train = R[(days >= start - pd.Timedelta(days=train_days)) & (days < start)]
        test = R[(days >= start) & (days < end)]
        if len(train) < 20 or test.empty:
            break
        scores = [metric(train[c].to_numpy()) for c in R.columns]
        best = int(np.argmax(scores))
        out.append(test[best])
        rows.append({"test_start": start.date(), "test_end": min(end - pd.Timedelta(days=1), days[-1]).date(),
                     "train_sharpe": round(float(scores[best]), 3), **combos[best]})
        start = end
    oos = pd.concat(out) if out else pd.Series(dtype=float)
    return WFOResult(oos, (1 + oos).cumprod(), pd.DataFrame(rows), R, combos, len(combos))


def pbo_cscv(returns: pd.DataFrame | np.ndarray, n_splits: int = 10,
             metric: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross validation.

    `returns` is T periods x N strategies (e.g. WFOResult.returns_matrix). The
    rows are cut into n_splits blocks; every half/half split of the blocks
    is tried. For each, the best strategy in sample is ranked out of sample.
    PBO is the share of splits where it ranks below the median. Near 0.5 or
    above means the selection process is picking noise.
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least two strategies")
    n_splits = n_splits - n_splits % 2
    blocks = np.array_split(np.arange(T), n_splits)
    metric = metric or (lambda X: X.mean(axis=0) / np.where(X.std(axis=0) > 0, X.std(axis=0), np.inf))
    logits, degradation = [], []
    for is_blocks in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_blocks])
        is_perf, oos_perf = metric(M[is_idx]), metric(M[oos_idx])
        best = int(np.argmax(is_perf))
        rank = (oos_perf < oos_perf[best]).sum() + 0.5 * ((oos_perf == oos_perf[best]).sum() - 1) + 1
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        degradation.append((is_perf[best], oos_perf[best]))
    logits = np.array(logits)
    deg = np.array(degradation)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)),
            "median_logit": float(np.median(logits)),
            "is_best_mean": float(deg[:, 0].mean()), "oos_of_is_best_mean": float(deg[:, 1].mean())}


def _stationary_bootstrap_indices(T: int, n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n)
    new_block = rng.random((n, T)) < p
    jumps = rng.integers(0, T, (n, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def reality_check(returns: pd.DataFrame | np.ndarray, n_boot: int = 1000, mean_block: float = 10.0,
                  seed: int = 0) -> dict:
    """White's reality check: p-value that the best strategy's mean return beats zero by luck.

    `returns` is T x N excess returns (daily is typical). A p-value under 0.05
    says the best strategy is unlikely to be a product of searching N of them.
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    means = M.mean(axis=0)
    v = np.sqrt(T) * means.max()
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(T, n_boot, mean_block, rng)
    boot_means = np.stack([M[i].mean(axis=0) for i in idx])       # n_boot x N
    v_star = np.sqrt(T) * (boot_means - means).max(axis=1)
    return {"p_value": float((v_star >= v).mean()), "best_index": int(means.argmax()),
            "best_mean": float(means.max()), "n_strategies": N, "n_boot": n_boot}


# Parameter grids for the built in strategies, used by `aurum optimize`.
GRIDS: dict[str, dict[str, list]] = {
    "london_breakout": {"buffer_atr": [0.0, 0.1, 0.25, 0.5], "max_range_atr": [3.0, 6.0]},
    "trend": {"fast_days": [10, 20, 40], "slow_days": [60, 100, 150]},
    "macro_reversion": {"enter": [1.25, 1.75, 2.25], "exit": [0.0, 0.25, 0.5]},
    "asian_reversion": {"enter": [0.8, 1.2, 1.6], "exit": [0.0, 0.2]},
    "fix_fade": {"start_hour": [12.0, 13.0, 14.0], "end_hour": [15.0, 16.0]},
}
