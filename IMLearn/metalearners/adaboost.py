# import numpy as np
# from ..base import BaseEstimator
# from typing import Callable, NoReturn
# from IMLearn.metrics.loss_functions import misclassification_error
#
#
# class AdaBoost(BaseEstimator):
#     """
#     AdaBoost class for boosting a specified weak learner
#
#     Attributes
#     ----------
#     self.wl_: Callable[[], BaseEstimator]
#         Callable for obtaining an instance of type BaseEstimator
#
#     self.iterations_: int
#         Number of boosting iterations to perform
#
#     self.models_: List[BaseEstimator]
#         List of fitted estimators, fitted along the boosting iterations
#     """
#
#     def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
#         """
#         Instantiate an AdaBoost class over the specified base estimator
#
#         Parameters
#         ----------
#         wl: Callable[[], BaseEstimator]
#             Callable for obtaining an instance of type BaseEstimator
#
#         iterations: int
#             Number of boosting iterations to perform
#         """
#         super().__init__()
#         self.wl_ = wl
#         self.iterations_ = iterations
#         self.models_, self.weights_, self.D_ = None, None, None
#
#     def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
#         """
#         Fit an AdaBoost classifier over given samples
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to fit an estimator for
#
#         y : ndarray of shape (n_samples, )
#             Responses of input data to fit to
#         """
#         n_samples, n_features = X.shape
#         self.D_ = np.ones(n_samples) / n_samples
#         self.models_ = np.empty(self.iterations_, dtype=BaseEstimator)
#         self.weights_ = np.empty(self.iterations_, dtype=float)
#
#         for t in range(self.iterations_):
#             curr_model = self.wl_()  # creates an instance of weak learner
#             curr_model.fit(X, y * self.D_)
#             self.models_[t] = curr_model
#             pred_t = curr_model.predict(X)
#             e_t = np.sum(self.D_ * (y != pred_t).astype(int))
#             self.weights_[t] = 0.5 * np.log(1 / e_t - 1)
#             self.D_ *= np.exp(-y * self.weights_[t] * pred_t)
#             self.D_ /= np.sum(self.D_)
#
#     def _predict(self, X):
#         """
#         Predict responses for given samples using fitted estimator
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to predict responses for
#
#         Returns
#         -------
#         responses : ndarray of shape (n_samples, )
#             Predicted responses of given samples
#         """
#         self.partial_predict(X, self.iterations_)
#
#     def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
#         """
#         Evaluate performance under misclassification loss function
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Test samples
#
#         y : ndarray of shape (n_samples, )
#             True labels of test samples
#
#         Returns
#         -------
#         loss : float
#             Performance under missclassification loss function
#         """
#         return self.partial_loss(X, y, self.iterations_)
#
#     def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
#         """
#         Predict responses for given samples using fitted estimators
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to predict responses for
#
#         T: int
#             The number of classifiers (from 1,...,T) to be used for prediction
#
#         Returns
#         -------
#         responses : ndarray of shape (n_samples, )
#             Predicted responses of given samples
#         """
#         p_pred = np.zeros(X.shape[0])
#         for t in range(T):
#             p_pred += self.models_[t].predict(X) * self.weights_[t]
#         return np.where(p_pred >= 0, 1, -1)
#
#
#     def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
#         """
#         Evaluate performance under misclassification loss function
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Test samples
#
#         y : ndarray of shape (n_samples, )
#             True labels of test samples
#
#         T: int
#             The number of classifiers (from 1,...,T) to be used for prediction
#
#         Returns
#         -------
#         loss : float
#             Performance under missclassification loss function
#         """
#         y_pred = self.partial_predict(X, T)
#         y_true = np.where(y >= 0, 1, -1)
#         return misclassification_error(y_true=y_true, y_pred=y_pred)

#----------------------------------------------------

import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner
    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator
    self.iterations_: int
        Number of boosting iterations to perform
    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator
        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator
        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        self.D_ = np.ones(shape=m) / m
        self.models_ = np.empty(shape=self.iterations_, dtype=BaseEstimator)
        self.weights_ = np.empty(shape=self.iterations_)
        for t in range(self.iterations_):
            wl_t = self.wl_().fit(X, y * self.D_)
            self.models_[t] = wl_t
            pred_t = wl_t.predict(X)
            epsilon_t = np.sum(self.D_ * (pred_t != y).astype(int))
            self.weights_[t] = .5 * np.log((1/epsilon_t) - 1)
            self.D_ *= np.exp(-y * (self.weights_[t] * pred_t))
            self.D_ /= np.sum(self.D_)

    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_ or T <= 0:
            raise ValueError("T must be in range [1, number of fitted models].")
        pred = np.zeros(shape=X.shape[0])
        for t in range(T):
            pred += self.weights_[t] * self.models_[t].predict(X)
        return np.where(pred >= 0, 1, -1)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        if T > self.iterations_ or T <= 0:
            raise ValueError("T must be in range [1, number of fitted models].")
        return misclassification_error(np.where(y >= 0, 1, -1),
                                       self.partial_predict(X, T))