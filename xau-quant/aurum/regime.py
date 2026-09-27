"""Market regime detection with a Gaussian hidden Markov model, in plain numpy.

Several of the open source gold bots use hmmlearn for this. Writing the
forward filter ourselves keeps the dependency list short and, more
importantly, makes it easy to guarantee that the regime reported at bar t
uses only bars up to t (filtered, not smoothed, probabilities). hmmlearn's
`predict` runs Viterbi over the whole sample, which leaks the future into
every past label.

Observations are a small vector per bar, by default the bar's return and its
rolling volatility, both standardized on the training window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class GaussianHMM:
    n_states: int = 3
    n_iter: int = 50
    tol: float = 1e-4
    seed: int = 0
    min_var: float = 1e-3

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        # Diagonal covariance Gaussians.
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.sum(diff**2 / self.vars_[None], axis=2)
            + np.sum(np.log(2 * np.pi * self.vars_), axis=1)[None]
        )

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_states
        rng = np.random.default_rng(self.seed)
        # Initialise means on quantiles of the last column (volatility), which
        # gives stable, interpretable states.
        order = np.argsort(X[:, -1])
        chunks = np.array_split(order, k)
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks]) + rng.normal(0, 1e-3, (k, d))
        self.vars_ = np.array([X[c].var(axis=0) + self.min_var for c in chunks])
        self.startprob_ = np.full(k, 1 / k)
        self.transmat_ = np.full((k, k), 0.02 / (k - 1))
        np.fill_diagonal(self.transmat_, 0.98)

        prev = -np.inf
        for _ in range(self.n_iter):
            log_b = self._log_emission(X)
            log_a = np.log(self.transmat_)
            # Forward.
            la = np.empty((n, k))
            la[0] = np.log(self.startprob_) + log_b[0]
            for t in range(1, n):
                la[t] = _logsumexp(la[t - 1][:, None] + log_a, axis=0) + log_b[t]
            # Backward.
            lb = np.zeros((n, k))
            for t in range(n - 2, -1, -1):
                lb[t] = _logsumexp(log_a + log_b[t + 1][None] + lb[t + 1][None], axis=1)
            ll = _logsumexp(la[-1], axis=0)
            gamma = np.exp(la + lb - ll)
            xi = np.zeros((k, k))
            for t in range(n - 1):
                x = la[t][:, None] + log_a + log_b[t + 1][None] + lb[t + 1][None] - ll
                xi += np.exp(x)
            # M step.
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi / xi.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            self.vars_ = (gamma.T @ X**2) / w[:, None] - self.means_**2 + self.min_var
            if ll - prev < self.tol:
                break
            prev = ll
        self.loglik_ = float(ll)
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t) for every t. Causal, safe to trade on."""
        X = np.asarray(X, dtype=float)
        log_b = self._log_emission(X)
        log_a = np.log(self.transmat_)
        n, k = log_b.shape
        out = np.empty((n, k))
        la = np.log(self.startprob_) + log_b[0]
        out[0] = np.exp(la - _logsumexp(la, axis=0))
        for t in range(1, n):
            la = _logsumexp(la[:, None] + log_a, axis=0) + log_b[t]
            la -= _logsumexp(la, axis=0)
            out[t] = np.exp(la)
        return out


REGIME_NAMES = {3: ["calm", "normal", "stressed"], 2: ["calm", "stressed"]}


def detect_regimes(
    close: pd.Series,
    train_end: int | None = None,
    n_states: int = 3,
    vol_window: int = 24,
    step: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit the HMM on bars [0, train_end) and filter the whole series.

    Features are the mean return and volatility over `vol_window` bars, sampled
    every `step` bars to speed up EM. States are sorted by volatility so column
    names are stable: calm, normal, stressed. Use `train_end` to keep the fit
    out of sample; default is the first half.
    """
    r = np.log(close).diff()
    feat = pd.DataFrame(
        {"ret": r.rolling(vol_window).mean(), "vol": np.log(r.rolling(vol_window).std())},
        index=close.index,
    ).dropna()
    if train_end is None:
        train_end = len(feat) // 2
    train = feat.iloc[:train_end:step]
    mu, sd = train.mean(), train.std().replace(0, 1)
    z = ((feat - mu) / sd).to_numpy()
    hmm = GaussianHMM(n_states=n_states, seed=seed).fit(z[:train_end:step])
    # Reorder states by volatility.
    order = np.argsort(hmm.means_[:, 1])
    probs = hmm.filter(z)[:, order]
    names = REGIME_NAMES.get(n_states, [f"s{i}" for i in range(n_states)])
    out = pd.DataFrame(probs, index=feat.index, columns=[f"p_{n}" for n in names])
    out["regime"] = np.array(names)[probs.argmax(axis=1)]
    return out.reindex(close.index)
