from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_features = X.shape[1]
        X_t = X.T
        best_err = 2

        for sign, f_index in product([-1, 1], range(n_features)):
            curr_feature = X_t[f_index]
            f_TH, curr_ER = self._find_threshold(curr_feature, y, sign)
            if best_err > curr_ER:
                best_err = curr_ER
                self.sign_ = sign
                self.j_ = f_index
                self.threshold_ = f_TH

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indexes = values.argsort()
        values_sorted = values[sorted_indexes]
        labels_sorted = labels[sorted_indexes]

        # purpose - find the index i where all the next index-s, including i, are with the minimum values that are below TH
        # reverse labels with opposite labels:
        #  - reverse in order to see in the opposite side when cumsum
        #  - multiply - for calculate the balance of under TH against g/e TH (favour for g/e TH)
        #   ** meaning - if all of them will be sign - how many are correct after 'KIZUZUIM'
        prep1 = (labels_sorted * sign)[::-1]
        prep2 = np.cumsum(prep1)  # calculate the balance from the end to the start
        TH_balance = prep2[::-1]  # reverse ->  the balance of each index (including) until the end = number of good labeling until the index.

        min_val_ind = np.argmax(TH_balance)

        y_pred = np.concatenate((-sign * np.ones(min_val_ind), sign * np.ones(labels_sorted.shape[0] - min_val_ind)))
        # multiply with abs('labels_sorted') in order to make weighted loss
        loss = np.sum(np.abs(labels_sorted) * (np.where(labels_sorted >= 0, 1, -1) != y_pred).astype(int))

        if min_val_ind > 0:
            return values_sorted[min_val_ind], loss
        else:
            return -np.inf, loss

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

        y_pred = self.predict(X)
        y_true = np.where(y >= 0, 1, -1)
        return misclassification_error(y_true=y_true, y_pred=y_pred)
