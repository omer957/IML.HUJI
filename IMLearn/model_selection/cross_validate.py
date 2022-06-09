from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

# from tqdm import trange


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    n_samples, *_ = X.shape

    folds = np.remainder(np.arange(n_samples), cv)
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)

    # for k in trange(cv, desc="k-fold"):
    for k in range(cv):
        X_train = X[folds != k]
        y_train = y[folds != k]
        X_validation = X[folds == k]
        y_validation = y[folds == k]
        estimator.fit(X_train, y_train)
        train_scores[k] = scoring(estimator.predict(X_train), y_train)
        validation_scores[k] = scoring(estimator.predict(X_validation), y_validation)

    return train_scores.mean(), validation_scores.mean()
