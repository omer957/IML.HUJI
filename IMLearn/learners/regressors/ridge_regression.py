from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from typing import Callable
from IMLearn.metrics.loss_functions import mean_square_error


def adjust_to_intercept(method: Callable):
    def wrapper(self, X, *args, **kwargs):
        if self.include_intercept_:
            X = np.concatenate((np.ones((len(X), 1), dtype=int), X), axis=1)
        # X = np.concatenate((np.ones((len(X), 1), dtype=int), X), axis=1) if self.include_intercept_ else X
        return method(self, X, *args, **kwargs)

    return wrapper


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    @adjust_to_intercept
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        valid_S_indexes = S > 0
        S_lam = np.zeros(S.size)
        S_lam[valid_S_indexes] = S / (S ** 2 + self.lam_)
        if self.include_intercept_ and S[0] != 0:
            S_lam[0] = 1 / S[0]
        self.coefs_ = Vt.T @ (np.diag(S_lam) @ (U.T @ y))

        # I_d = np.identity(X.shape[1])
        # if self.include_intercept_:
        #     I_d[0][0] = 0
        # self.coefs_ = np.linalg.inv(X.T @ X + self.lam_ * I_d) @ X.T @ y

        self.fitted_ = True

    @adjust_to_intercept
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self.predict(X))

