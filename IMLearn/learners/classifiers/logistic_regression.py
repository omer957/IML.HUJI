from typing import NoReturn
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2
from IMLearn.metrics import misclassification_error


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        if self.include_intercept_:
            X = np.insert(X, 0, 1, axis=1) if len(X.shape) > 1 else np.insert(X, 0, 1, axis=0)

        n_samples, n_features = X.shape
        weights = np.random.normal(0, 1, n_features) / np.sqrt(n_features)

        if self.penalty_ == "none":
            self.coefs_ = self.solver_.fit(LogisticModule(weights), X, y)
        else:
            regularization_type = {"l1": L1, "l2": L2}
            regularization_module = regularization_type[self.penalty_](weights)
            regularized_module = RegularizedModule(fidelity_module=LogisticModule(weights),
                                                   regularization_module=regularization_module,
                                                   lam=self.lam_,
                                                   weights=weights,
                                                   include_intercept=self.include_intercept_)
            self.coefs_ = self.solver_.fit(f=regularized_module, X=X, y=y)

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
        return np.where(self.predict_proba(X) >= self.alpha_, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        if self.include_intercept_:
            X = np.insert(X, 0, 1, axis=1) if len(X.shape) > 1 else np.insert(X, 0, 1, axis=0)
        exp_part = np.exp(X @ self.coefs_)
        return exp_part / (1 + exp_part)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
