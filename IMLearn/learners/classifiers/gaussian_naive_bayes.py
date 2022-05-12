from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self.pi_ = np.zeros(n_classes)
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            x_c = X[y == c]
            self.mu_[i] = np.mean(x_c, axis=0)
            self.vars_[i] = np.var(x_c, axis=0, ddof=1)
            self.pi_[i] = x_c.shape[0] / n_samples

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
        a = np.argmax(self.likelihood(X), axis=1)
        # print("gnb: \n", np.unique(a))
        return a

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        res = np.zeros((n_samples, n_classes))

        for j in range(n_classes):
            curr_cov = np.diag(self.vars_[j])
            inv_curr_cov = np.linalg.inv(curr_cov)
            z = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(curr_cov))
            for i in range(n_samples):
                x_m = X[i] - self.mu_[j]
                x_ij_likelihood = (1 / z) * np.exp(-0.5 * (x_m @ inv_curr_cov) @ x_m.T)
                res[i, j] = x_ij_likelihood * self.pi_[j]
        return res

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
        return misclassification_error(y, y_pred)
