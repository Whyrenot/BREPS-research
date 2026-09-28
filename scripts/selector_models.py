"""
selector_models.py
==================
The estimators, and the one thing they all have to agree on: a *score* per
candidate, higher meaning "pick me". Everything downstream -- argmax selection,
early stopping, per-group tables -- reads only that score, so CatBoost and a
logistic regression are interchangeable at the policy layer.

Four ways to turn "which of these masks is best" into a supervised problem, all
implemented for both families so the comparison is apples to apples:

  reg    regress target_iou. Wastes most of its capacity on case difficulty,
         which is constant inside a pool and therefore irrelevant to the choice.
  regc   regress target_iou minus the pool mean. Same features, the nuisance
         removed from the LABEL. Usually the cheapest real win.
  rank   pairwise. CatBoost: YetiRank. Linear: Bradley-Terry logistic on feature
         DIFFERENCES within a pool, no intercept, weighted by the IoU gap the
         pair is worth -- which makes it a regret-weighted ranker, not just an
         accuracy-weighted one. This is the formulation logistic regression is
         actually for; plain per-row logistic on raw features cannot express
         "best of this pool" at all.
  best   binary "is this the pool's argmax". Blunter than rank (it throws away
         the ordering of the losers) but it is the one head whose output is
         directly a probability, which the stop rule can threshold.

Plus one head that only makes sense for early stopping:

  cont   binary "does any LATER checkpoint beat the best seen so far". Features
         are causal; the LABEL looks into the future, which is exactly what
         supervision is for. Thresholding this is a genuine stopping rule,
         where thresholding a quality regression is only a proxy for one.

Missing values. CatBoost takes NaN natively and splits on it, which matters
here: `f_agree_*` is NaN for the undefended mask because it has no pool to
agree with, and that is information, not absence. The linear models cannot, so
they get median imputation plus an explicit was-missing indicator per column,
both fitted on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CB_DEFAULTS = dict(depth=6, learning_rate=0.06, random_seed=0, verbose=0,
                   allow_writing_files=False)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def pool_centered(y: pd.Series, case: pd.Series) -> np.ndarray:
    return (y - y.groupby(case).transform("mean")).values


def is_pool_best(y: pd.Series, case: pd.Series, eps: float = 1e-9) -> np.ndarray:
    return (y >= y.groupby(case).transform("max") - eps).astype(int).values


def improves_later(y: pd.Series, case: pd.Series, eps: float = 0.0) -> np.ndarray:
    """1 if some checkpoint after this one beats the best seen up to and
    including this one. Rows must be in (case, step) order."""
    g = y.groupby(case, sort=False)
    best_so_far = g.cummax().values
    # reverse cummax of the strictly later rows
    rev = y.iloc[::-1].groupby(case.iloc[::-1], sort=False).cummax().iloc[::-1].values
    later_max = np.r_[rev[1:], -np.inf]
    boundary = case.values != np.r_[case.values[1:], None]
    later_max = np.where(boundary, -np.inf, later_max)
    return (later_max > best_so_far + eps).astype(int)


# --------------------------------------------------------------------------
# CatBoost
# --------------------------------------------------------------------------
def fit_predict_catboost(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                         mode: str, iterations: int, threads: int,
                         seed: int = 0, **over) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

    kw = dict(CB_DEFAULTS, iterations=iterations, thread_count=threads,
              random_seed=seed, **over)
    if mode in ("reg", "regc"):
        y = tr.target_iou.values if mode == "reg" else pool_centered(tr.target_iou, tr.case)
        m = CatBoostRegressor(loss_function="RMSE", **kw)
        m.fit(tr[cols], y)
        return m.predict(te[cols])
    if mode in ("best", "cont"):
        y = tr["_y_" + mode].values
        if y.min() == y.max():                      # degenerate fold
            return np.zeros(len(te))
        m = CatBoostClassifier(loss_function="Logloss", **kw)
        m.fit(tr[cols], y)
        return m.predict(te[cols], prediction_type="RawFormulaVal")
    if mode == "rank":
        gtr = pd.factorize(tr.case)[0]
        gte = pd.factorize(te.case)[0]
        m = CatBoostRanker(loss_function="YetiRank", **kw)
        m.fit(Pool(tr[cols], tr.target_iou.values, group_id=gtr))
        return m.predict(Pool(te[cols], group_id=gte))
    raise ValueError(mode)


def catboost_importance(tr: pd.DataFrame, cols: list[str], mode: str,
                        iterations: int, threads: int, seed: int = 0,
                        **over) -> pd.Series:
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

    kw = dict(CB_DEFAULTS, iterations=iterations, thread_count=threads,
              random_seed=seed, **over)
    if mode in ("reg", "regc"):
        y = tr.target_iou.values if mode == "reg" else pool_centered(tr.target_iou, tr.case)
        m = CatBoostRegressor(loss_function="RMSE", **kw).fit(tr[cols], y)
    elif mode in ("best", "cont"):
        m = CatBoostClassifier(loss_function="Logloss", **kw).fit(tr[cols], tr["_y_" + mode].values)
    else:
        m = CatBoostRanker(loss_function="YetiRank", **kw)
        m.fit(Pool(tr[cols], tr.target_iou.values, group_id=pd.factorize(tr.case)[0]))
    return pd.Series(m.get_feature_importance(), index=cols).sort_values(ascending=False)


# --------------------------------------------------------------------------
# linear / logistic
# --------------------------------------------------------------------------
class LinearPrep:
    """Median impute + was-missing flags + standardise, fitted on train only."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame):
        A = X[self.cols].to_numpy(dtype=np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        self.med_ = np.nanmedian(A, axis=0)
        self.med_[~np.isfinite(self.med_)] = 0.0
        self.flag_ = np.isnan(A).any(axis=0)
        self.scaler_ = StandardScaler().fit(self._raw(X))
        return self

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.cols].to_numpy(dtype=np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        miss = np.isnan(A)
        A = np.where(miss, self.med_, A)
        if self.flag_.any():
            A = np.hstack([A, miss[:, self.flag_].astype(np.float64)])
        return A

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self._raw(X))


def _case_bounds(case: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start offset and size of each contiguous run of `case`."""
    chg = np.r_[True, case[1:] != case[:-1]]
    start = np.flatnonzero(chg)
    size = np.diff(np.r_[start, len(case)])
    return start, size


def sample_pairs(case: np.ndarray, y: np.ndarray, n_pairs: int, rng,
                 min_gap: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """`n_pairs` random within-pool pairs per case, drawn in one vectorised go.

    Rows must be grouped by case (they are: every builder sorts on it). Pairs
    with a negligible IoU gap are dropped -- they carry no ranking signal and
    only blur the decision boundary.
    """
    start, size = _case_bounds(case)
    reps = np.repeat(np.arange(len(start)), n_pairs)
    s = start[reps]
    n = size[reps]
    i = s + (rng.random(len(reps)) * n).astype(np.int64)
    j = s + (rng.random(len(reps)) * n).astype(np.int64)
    keep = np.abs(y[i] - y[j]) > min_gap
    return i[keep], j[keep]


class LogRegRanker:
    """Bradley-Terry: P(i beats j) = sigmoid(w . (x_i - x_j)), no intercept.

    Sample weight |iou_i - iou_j| makes the fit care about pairs in proportion
    to the regret of getting them backwards, so a coin-flip between two masks
    that differ in the 4th decimal costs the model nothing.
    """

    def __init__(self, cols, C=1.0, n_pairs=32, seed=0, max_iter=1000):
        self.cols, self.C, self.n_pairs, self.seed, self.max_iter = (
            cols, C, n_pairs, seed, max_iter)

    def fit(self, tr: pd.DataFrame):
        self.prep_ = LinearPrep(self.cols).fit(tr)
        X = self.prep_.transform(tr)
        y = tr.target_iou.to_numpy()
        rng = np.random.default_rng(self.seed)
        i, j = sample_pairs(tr.case.to_numpy(), y, self.n_pairs, rng)
        flip = rng.random(len(i)) < 0.5                  # balance the labels
        i, j = np.where(flip, j, i), np.where(flip, i, j)
        D = X[i] - X[j]
        lab = (y[i] > y[j]).astype(int)
        w = np.abs(y[i] - y[j])
        self.clf_ = LogisticRegression(C=self.C, fit_intercept=False,
                                       max_iter=self.max_iter, solver="lbfgs",
                                       tol=1e-3)
        self.clf_.fit(D, lab, sample_weight=w)
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        return self.prep_.transform(te) @ self.clf_.coef_.ravel()


class LogRegBinary:
    """Plain logistic on a binary label, scored by the decision function."""

    def __init__(self, cols, label: str, C=1.0, max_iter=1000, balanced=True):
        self.cols, self.label, self.C, self.max_iter = cols, label, C, max_iter
        self.balanced = balanced

    def fit(self, tr: pd.DataFrame):
        self.prep_ = LinearPrep(self.cols).fit(tr)
        y = tr["_y_" + self.label].to_numpy()
        self.const_ = y.min() == y.max()
        if self.const_:
            return self
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs", tol=1e-3,
            class_weight="balanced" if self.balanced else None)
        self.clf_.fit(self.prep_.transform(tr), y)
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        if self.const_:
            return np.zeros(len(te))
        return self.clf_.decision_function(self.prep_.transform(te))

    def predict_proba(self, te: pd.DataFrame) -> np.ndarray:
        if self.const_:
            return np.zeros(len(te))
        return self.clf_.predict_proba(self.prep_.transform(te))[:, 1]


def fit_predict_linear(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                       mode: str, C: float, n_pairs: int, seed: int = 0) -> np.ndarray:
    if mode == "rank":
        return LogRegRanker(cols, C=C, n_pairs=n_pairs, seed=seed).fit(tr).predict(te)
    if mode in ("best", "cont"):
        return LogRegBinary(cols, mode, C=C).fit(tr).predict(te)
    if mode in ("reg", "regc"):
        from sklearn.linear_model import Ridge
        prep = LinearPrep(cols).fit(tr)
        y = tr.target_iou.values if mode == "reg" else pool_centered(tr.target_iou, tr.case)
        return Ridge(alpha=1.0 / max(C, 1e-9)).fit(prep.transform(tr), y).predict(prep.transform(te))
    raise ValueError(mode)
