from __future__ import annotations
import numpy as np
import pandas as pd
from typing import NoReturn
from sklearn import datasets
from sklearn.linear_model import Lasso
from IMLearn.base import BaseEstimator
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression

from tqdm import trange

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MyLasso(BaseEstimator):
    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        super().__init__()
        self.lam_ = lam
        self.lasso = Lasso(self.lam_, max_iter=1000)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.lasso.fit(X, y)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.lasso.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return mean_square_error(y, y_pred)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # raise NotImplementedError()

    X = np.linspace(-1.2, 2, n_samples)
    y_clean = np.array(list(map(lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2), X)))
    y_noise = y_clean + np.random.normal(0, noise, n_samples)

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y_noise), train_proportion=(2 / 3))
    train_X, train_y, test_X, test_y = train_X.sort_index().iloc[:, 0].to_numpy(), train_y.sort_index().to_numpy(), \
                                       test_X.sort_index().iloc[:, 0].to_numpy(), test_y.sort_index().to_numpy()

    fig1 = go.Figure(
        data=[
            go.Scatter(x=X, y=y_clean, mode='markers', name="noise_less", marker=dict(size=5, opacity=0.6)),
            go.Scatter(x=train_X, y=train_y, mode='markers', name="train with noise", marker=dict(size=5, opacity=0.6)),
            go.Scatter(x=test_X, y=test_y, mode='markers', name="test with noise", marker=dict(size=5, opacity=0.6)),

        ],
        layout=go.Layout(
            title=fr"$\text{{noiseless data and train and test sets with noise}} \sigma^2 = {noise}$",
            xaxis_title='x',
            yaxis_title='y',
            height=1000,
            width=1000)
    )
    # fig1.show()
    fig1.write_image("./q1.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = np.zeros(11)
    test_scores = np.zeros(11)
    for k in range(11):
        train_scores[k], test_scores[k] = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
        print(f"\t{k} - validation error = {np.round_(test_scores[k], decimals = 2)}")

    best_k = np.argmin(test_scores)
    selected_error = test_scores[best_k]

    fig2 = go.Figure(
        data=[
            go.Scatter(x=np.linspace(0, 10, 11), y=train_scores, mode='markers+lines',
                       name="train", marker=dict(size=7.5, opacity=0.6)),
            go.Scatter(x=np.linspace(0, 10, 11), y=test_scores, mode='markers+lines',
                       name="test", marker=dict(size=7.5, opacity=0.6)),
            go.Scatter(x=[best_k], y=[selected_error], mode='markers',
                       name='Best Model', marker=dict(color='darkred', symbol="x", size=10))
        ],
        layout=go.Layout(
            title=fr'5-fold cross validation. n_samles: {n_samples}, noise: {noise}',
            xaxis_title='k',
            yaxis_title='score - mean square error',
            height=1000,
            width=1000)
    )
    # fig2.show()
    fig2.write_image("./q2.png")

    # raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()

    poly_model = PolynomialFitting(best_k)
    y_pred = poly_model.fit(train_X, train_y).predict(test_X)
    reported_error = mean_square_error(test_y, y_pred)
    print(f"noise : {noise}, n_samples: {n_samples} | \ttest error for k* = {best_k} : {np.round_(reported_error, decimals = 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
       Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
       values for Ridge and Lasso regressions

       Parameters
       ----------
       n_samples: int, default=50
           Number of samples to generate

       n_evaluations: int, default = 500
           Number of regularization parameter values to evaluate for each of the algorithms
       """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X, y = datasets.load_diabetes(return_X_y=True)
    train_samples = 50
    train_X, train_y, test_X, test_y = X[:train_samples], y[:train_samples], X[train_samples:], y[train_samples:]

    # # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    l_min = 0.001
    l_max = 3
    l_range = np.linspace(l_min, l_max, n_evaluations)

    ridge_scores = np.zeros(shape=(n_evaluations, 2))
    lasso_scores = np.zeros(shape=(n_evaluations, 2))

    for i in trange(n_evaluations, desc="find best lambda"):
    # for i in range(n_evaluations):
        curr_l = l_range[i];
        ridge_scores[i] = cross_validate(RidgeRegression(curr_l), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(MyLasso(curr_l), train_X, train_y, mean_square_error)

    models = ["Ridge", "Lasso"]
    scores = [ridge_scores, lasso_scores]

    fig3 = make_subplots(rows=2,
                         cols=1,
                         subplot_titles=[f"train and validation errors: {m}" for m in models],
                         horizontal_spacing=0.1,
                         vertical_spacing=0.1,
                         )

    for i in range(2):
        fig3.add_traces([go.Scatter(x=l_range,
                                    y=scores[i][:, 0],
                                    mode='markers+lines',
                                    name=f"train of {models[i]}",
                                    marker=dict(size=4, opacity=0.6)),
                         go.Scatter(x=l_range,
                                    y=scores[i][:, 1],
                                    mode='markers+lines',
                                    name=f"validation of {models[i]}",
                                    marker=dict(size=4, opacity=0.6))
                         ],
                        rows=i + 1, cols=1)
    # fig3.show()
    fig3.write_image("./q7.png")

    best_lambda_ridge = l_range[np.argmin(scores[0][:, 1])]
    best_lambda_lasso = l_range[np.argmin(scores[1][:, 1])]

    print(f"best lambda ridge: {best_lambda_ridge}")
    print(f"best lambda lasso: {best_lambda_lasso}")
    print()
    ridge_model = RidgeRegression(best_lambda_ridge)
    lasso_model = MyLasso(best_lambda_lasso)
    linear_model = LinearRegression()

    ridge_model.fit(train_X, train_y)
    lasso_model.fit(train_X, train_y)
    linear_model.fit(train_X, train_y)

    ridge_test_error = ridge_model.loss(test_X, test_y)
    lasso_test_error = lasso_model.loss(test_X, test_y)
    linear_test_error = linear_model.loss(test_X, test_y)

    print(f"ridge test error: {np.round_(ridge_test_error, decimals = 2)}")
    print(f"lasso test error: {np.round_(lasso_test_error, decimals = 2)}")
    print(f"least squares test error: {np.round_(linear_test_error, decimals = 2)}")

    # # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()  # q1,2,3
    select_polynomial_degree(noise=0)  # 4
    select_polynomial_degree(n_samples=1500, noise=10)  # 5
    select_regularization_parameter()  # q6,7,8

