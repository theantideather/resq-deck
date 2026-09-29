"""Machine learning signal for gold with the safeguards most retail bots skip.

    labels        Triple barrier (Lopez de Prado, Advances in Financial ML,
                  ch. 3): for each bar, did price hit +k ATR or -k ATR first
                  within `horizon` bars? That is the question a stop and a
                  target actually ask, unlike "is the next bar green".
    validation    Purged walk forward. A label at t uses bars up to
                  t + horizon, so training rows whose label window reaches
                  into the test block are dropped. Training only ever uses
                  bars before the test block, so no embargo is needed (that
                  matters for k-fold CV, where training data follows the test).
    model         sklearn HistGradientBoostingClassifier: fast, handles the
                  NaNs that macro features have early on, no GPU needed.
                  Swap in XGBoost or LightGBM through `model_factory`.
    signal        p(up) - p(down) from out of sample predictions only.

Everything returned by `walk_forward` is out of sample: every prediction
comes from a model that never saw that bar or any bar whose label overlapped it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import MarketData
from .features import atr as _atr


def triple_barrier_labels(bars: pd.DataFrame, horizon: int = 24, k_atr: float = 1.5,
                          atr_period: int = 14) -> pd.Series:
    """+1 if the upper barrier is touched first, -1 for the lower, 0 if neither.

    If both are touched inside the same bar the label is 0, since the order is
    unknowable from OHLC. The last `horizon` bars get NaN.
    """
    c = bars["close"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    a = _atr(bars, atr_period).to_numpy()
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        if not np.isfinite(a[i]):
            continue
        up, dn = c[i] + k_atr * a[i], c[i] - k_atr * a[i]
        hh = h[i + 1 : i + 1 + horizon]
        ll = lo[i + 1 : i + 1 + horizon]
        up_hit = np.flatnonzero(hh >= up)
        dn_hit = np.flatnonzero(ll <= dn)
        tu = up_hit[0] if len(up_hit) else horizon
        td = dn_hit[0] if len(dn_hit) else horizon
        if tu < td:
            out[i] = 1
        elif td < tu:
            out[i] = -1
        else:
            out[i] = 0
    return pd.Series(out, index=bars.index, name="label")


def default_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=0,
    )


@dataclass
class WalkForwardResult:
    proba: pd.DataFrame           # out of sample class probabilities, columns -1, 0, 1
    labels: pd.Series
    folds: list[dict]
    feature_names: list[str]

    def edge(self) -> pd.Series:
        p = self.proba
        return (p.get(1.0, 0.0) - p.get(-1.0, 0.0)).rename("edge")

    def signal(self, threshold: float = 0.1, cap: float = 0.4) -> pd.Series:
        """Enter when |edge| clears `threshold`, hold until it falls below a third
        of that. Conviction scales with |edge| up to `cap`."""
        from .strategies import hysteresis

        e = self.edge()
        side = hysteresis(e, threshold, threshold / 3, fade=False)
        conviction = (e.abs().clip(lower=threshold, upper=cap) / cap)
        return (side * conviction).fillna(0.0)

    def hit_rate(self) -> float:
        e = self.edge()
        m = self.labels.notna() & e.notna() & (self.labels != 0) & (e != 0)
        return float((np.sign(e[m]) == self.labels[m]).mean()) if m.any() else float("nan")


def walk_forward(
    md: MarketData,
    feats: pd.DataFrame,
    horizon: int = 24,
    k_atr: float = 1.5,
    train_bars: int | None = None,
    test_bars: int = 2000,
    min_train: int = 6000,
    model_factory: Callable[[], object] = default_model,
    drop_features: tuple[str, ...] = (),
) -> WalkForwardResult:
    """Purged walk forward. `train_bars=None` means an expanding window."""
    labels = triple_barrier_labels(md.bars, horizon, k_atr)
    X = feats.drop(columns=[c for c in drop_features if c in feats], errors="ignore")
    names = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    yv = labels.to_numpy()
    n = len(X)
    proba = pd.DataFrame(np.nan, index=X.index, columns=[-1.0, 0.0, 1.0])
    folds = []
    start = min_train
    while start < n - horizon:
        end = min(start + test_bars, n)
        # Purge: training labels must resolve before the test block starts.
        tr_end = start - horizon
        tr_start = 0 if train_bars is None else max(0, tr_end - train_bars)
        tr = np.arange(tr_start, tr_end)
        tr = tr[np.isfinite(yv[tr])]
        if len(tr) < 1000:
            start = end
            continue
        model = model_factory()
        model.fit(Xv[tr], yv[tr])
        te = np.arange(start, end)
        p = model.predict_proba(Xv[te])
        for j, cls in enumerate(model.classes_):
            proba.iloc[te, proba.columns.get_loc(float(cls))] = p[:, j]
        folds.append({
            "train_start": X.index[tr_start], "train_end": X.index[tr_end - 1],
            "test_start": X.index[start], "test_end": X.index[end - 1],
            "n_train": int(len(tr)),
        })
        start = end
    proba = proba.fillna({-1.0: 0.0, 0.0: 0.0, 1.0: 0.0}).where(proba.notna().any(axis=1))
    return WalkForwardResult(proba, labels, folds, names)


def permutation_importance_oos(md: MarketData, feats: pd.DataFrame, result: WalkForwardResult,
                               n_repeats: int = 3, sample: int = 5000, seed: int = 0) -> pd.Series:
    """Permutation importance of the last fold's model on its own test block.

    Refits the last fold (cheap) and measures how much log loss worsens when
    each feature is shuffled. Features near zero or negative are noise.
    """
    from sklearn.inspection import permutation_importance

    last = result.folds[-1]
    X = feats[result.feature_names]
    y = result.labels
    tr = (X.index >= last["train_start"]) & (X.index <= last["train_end"]) & y.notna().to_numpy()
    te = (X.index >= last["test_start"]) & (X.index <= last["test_end"]) & y.notna().to_numpy()
    model = default_model().fit(X[tr].to_numpy(float), y[tr].to_numpy())
    Xte, yte = X[te].to_numpy(float), y[te].to_numpy()
    if len(Xte) > sample:
        pick = np.random.default_rng(seed).choice(len(Xte), sample, replace=False)
        Xte, yte = Xte[pick], yte[pick]
    r = permutation_importance(model, Xte, yte, n_repeats=n_repeats, random_state=seed, scoring="neg_log_loss")
    return pd.Series(r.importances_mean, index=result.feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Meta-labeling
# ---------------------------------------------------------------------------

@dataclass
class MetaResult:
    proba: pd.Series          # out of sample P(primary trade wins), NaN where primary is flat
    labels: pd.Series         # 1 if the primary side hit its target barrier first, else 0
    folds: int

    def signal(self, primary: pd.Series, threshold: float = 0.55) -> pd.Series:
        """Keep the primary side only where P(win) clears `threshold`, sized by that probability."""
        p = self.proba.reindex(primary.index)
        size = ((p - threshold) / (1 - threshold)).clip(0, 1)
        return (np.sign(primary) * size.where(p >= threshold, 0.0)).fillna(0.0)

    def precision(self, threshold: float = 0.55) -> tuple[float, float]:
        """Hit rate of all primary bets vs of those the meta model keeps."""
        m = self.proba.notna() & self.labels.notna()
        base = float(self.labels[m].mean()) if m.any() else float("nan")
        kept = m & (self.proba >= threshold)
        return base, float(self.labels[kept].mean()) if kept.any() else float("nan")


def meta_label(md: MarketData, feats: pd.DataFrame, primary: pd.Series, horizon: int = 24,
               k_atr: float = 1.5, test_bars: int = 2000, min_train: int = 6000,
               calibrate: bool = True) -> MetaResult:
    """Meta-labeling (Lopez de Prado, ch. 3.6): the primary strategy picks the side,
    a secondary model learns when to believe it.

    Trained with the same purged walk forward as `walk_forward`, on bars where
    the primary signal is non zero. With `calibrate`, probabilities go through
    isotonic calibration so a 0.6 means roughly 60 percent.
    """
    from sklearn.calibration import CalibratedClassifierCV

    tb = triple_barrier_labels(md.bars, horizon, k_atr)
    side = np.sign(primary.reindex(feats.index).fillna(0.0))
    active = side != 0
    y = pd.Series(np.where(tb.notna() & active, (tb == side).astype(float), np.nan), index=feats.index)
    X = feats.assign(primary_side=side).to_numpy(dtype=float)
    yv = y.to_numpy()
    proba = pd.Series(np.nan, index=feats.index)
    n, folds, start = len(feats), 0, min_train
    while start < n - horizon:
        end = min(start + test_bars, n)
        tr = np.arange(0, start - horizon)
        tr = tr[np.isfinite(yv[tr])]
        te = np.arange(start, end)
        te = te[active.to_numpy()[te]]
        if len(tr) >= 500 and len(np.unique(yv[tr])) == 2 and len(te):
            base = default_model()
            model = CalibratedClassifierCV(base, method="isotonic", cv=3) if calibrate else base
            model.fit(X[tr], yv[tr])
            pos = list(model.classes_).index(1.0)
            proba.iloc[te] = model.predict_proba(X[te])[:, pos]
            folds += 1
        start = end
    return MetaResult(proba, y, folds)
