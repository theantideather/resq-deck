import numpy as np
import pandas as pd
import pytest

from aurum.data import synthetic_market
from aurum.features import build_features
from aurum.ml import meta_label, triple_barrier_labels
from aurum.research import GRIDS, param_grid, pbo_cscv, reality_check, walk_forward_optimize
from aurum.strategies import STRATEGIES


def test_param_grid():
    g = param_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "z"} in g
    for name, grid in GRIDS.items():
        assert name in STRATEGIES and len(param_grid(grid)) >= 4


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (1000, 30))
    assert pbo_cscv(noise)["pbo"] > 0.3
    edge = noise.copy()
    edge[:, 7] += 0.004  # one strategy genuinely better everywhere
    r = pbo_cscv(edge)
    assert r["pbo"] < 0.05 and r["oos_of_is_best_mean"] > 0


def test_reality_check_noise_vs_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (750, 20))
    assert reality_check(noise, n_boot=300)["p_value"] > 0.1
    edge = noise.copy()
    edge[:, 3] += 0.003
    r = reality_check(edge, n_boot=300)
    assert r["p_value"] < 0.05 and r["best_index"] == 3


def test_walk_forward_optimize_runs_and_counts_trials():
    md = synthetic_market(start="2021-01-01", end="2022-12-31", seed=4)
    feats = build_features(md, fair_window=1000)
    s = STRATEGIES["london_breakout"]
    res = walk_forward_optimize(md, feats, s.signal, {"buffer_atr": [0.0, 0.25], "max_range_atr": [3.0, 6.0]},
                                s.config, train_days=180, test_days=90)
    assert res.n_trials == 4 and res.returns_matrix.shape[1] == 4
    assert len(res.chosen) >= 4 and {"buffer_atr", "max_range_atr"} <= set(res.chosen.columns)
    assert res.oos_returns.index.is_monotonic_increasing
    assert np.isfinite(res.sharpe())


def test_meta_label_learns_when_primary_is_right():
    md = synthetic_market(start="2021-01-01", end="2022-12-31", seed=5)
    rng = np.random.default_rng(3)
    idx = md.bars.index
    tb = triple_barrier_labels(md.bars, 12, 1.0)
    primary = pd.Series(rng.choice([-1.0, 1.0], len(idx)), index=idx)
    # A feature that knows, noisily, whether the primary side will win.
    wins = (tb == primary).astype(float)
    feats = pd.DataFrame({"hint": wins + rng.normal(0, 0.6, len(idx)), "noise": rng.normal(size=len(idx))}, index=idx)
    res = meta_label(md, feats, primary, horizon=12, k_atr=1.0, min_train=3000, test_bars=2000)
    base, kept = res.precision(0.55)
    assert res.folds >= 3
    assert kept > base + 0.1
    sig = res.signal(primary)
    assert sig.abs().max() <= 1 and (sig != 0).mean() < (primary != 0).mean()
