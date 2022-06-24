import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error, mean_square_error
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black",
                                 line=dict(width=1),
                                 marker=dict(size=2))],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values = []
    weights = []

    def callback(**kwargs) -> None:
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


titles = ["L1", "L2"]


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for i, m in enumerate([L1, L2]):
        fig_model = go.Figure()
        min_val = float('inf')
        for j, eta in enumerate(etas):
            callback, values, weights = get_gd_state_recorder_callback()
            module = m(weights=init.copy())
            weights.append(init.copy())
            GD = GradientDescent(callback=callback, learning_rate=FixedLR(eta))
            GD.fit(f=module, X=None, y=None)
            path_title = f" for {titles[i]} norm with eta = {eta}"
            fig_path = plot_descent_path(module=m, descent_path=np.array(weights), title=path_title)
            file_title = "./" + path_title + ".png"
            fig_path.write_image(file_title)

            # trace for convergence rate of current eta
            fig_model.add_traces(data=[go.Scatter(x=np.arange(len(values)), y=values,
                                                  name=eta,
                                                  mode="markers+lines",
                                                  line=dict(width=0.5),
                                                  marker=dict(size=2))])
            min_val = min(min_val, np.min(values))

        # q4
        print(f"the lowest loss achieved for {titles[i]}: {min_val}")

        fig_title = f"module {titles[i]} norm - convergence rate for different etas "
        fig_model.update_layout(title=fig_title)
        fig_file_name = "./" + fig_title + ".png"
        fig_model.write_image(fig_file_name)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    fig = go.Figure()

    # for i, m in enumerate([L1, L2]):
    min_val = float('inf')

    for j, gamma in enumerate(gammas):
        callback, values, weights = get_gd_state_recorder_callback()

        module = L1(weights=init.copy())
        weights.append(init.copy())

        GD = GradientDescent(callback=callback, learning_rate=ExponentialLR(0.1, gamma))
        GD.fit(f=module, X=None, y=None)

        path_title = f" for L1 norm with eta = 1, gamma = {gamma}"
        fig_path = plot_descent_path(module=L1, descent_path=np.array(weights), title=path_title)
        file_title = "./expLR - " + path_title + ".png"
        fig_path.write_image(file_title)
        # if gamma == 0.95:
        #     fig_path.show()

        # trace for convergence rate of current eta
        fig.add_traces(data=[go.Scatter(x=np.arange(len(values)), y=values,
                                        name=f"L1 - gamma: {gamma}",
                                        mode="markers+lines",
                                        line=dict(width=0.5),
                                        marker=dict(size=2))])
        min_val = min(min_val, np.min(values))

    fig_title = f"q5 - convergence rate for all decay rates"
    fig.update_layout(title=fig_title)
    fig_file_name = "./" + fig_title + ".png"
    fig.write_image(fig_file_name)
    print(f"the lowest loss achieved for L1: {min_val}")

    # Plot algorithm's convergence for the different values of gamma

    # Plot descent path for gamma=0.95


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset

    train_X, train_y, test_X, test_y = load_data()
    train_X, train_y, test_X, test_y = train_X.sort_index().iloc[:, :].to_numpy(), train_y.sort_index().to_numpy(), \
                                       test_X.sort_index().iloc[:, :].to_numpy(), test_y.sort_index().to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data

    from sklearn.metrics import roc_curve, auc

    logistic_regression = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                                    max_iter=2 * 10 ** 4))  # default learning rate
    logistic_regression.fit(train_X, train_y)
    y_prob_pred = logistic_regression.predict_proba(train_X)
    fpr, tpr, thresholds = roc_curve(train_y, y_prob_pred)

    fig8 = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr,
                         mode='markers+lines',
                         text=thresholds, name="",
                         showlegend=False,
                         marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    # fig8.show()
    fig8.write_image("./q8.png")

    best_alpha_index = np.argmax(tpr - fpr)
    best_alpha = thresholds[best_alpha_index]
    model_error_best_alpha = misclassification_error(test_y, np.where(logistic_regression.predict_proba(test_X) >= best_alpha, 1, 0))
    # print(f"q9: best alpha: {np.round_(best_alpha, decimals = 2)}, its error over test: {np.round_(model_error_best_alpha, decimals = 2)}")
    print(f"q9: best alpha: {best_alpha}, its error over test: {model_error_best_alpha}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    print("\nq 10 + 11:")

    l_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for penalty in ["l1", "l2"]:
        scores = np.zeros(shape=(len(l_values), 2))
        for i, l in enumerate(l_values):
            regularized_logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                                             max_iter=2 * 10 ** 4),
                                                      penalty="l1",
                                                      alpha=0.5,
                                                      lam=l)
            scores[i] = cross_validate(regularized_logistic, train_X, train_y, misclassification_error)

        # go.Figure(
        #     data=[
        #         go.Scatter(
        #             x=l_values,
        #             y=scores[:, 1],
        #             name=f'{penalty} penalty Validation',
        #             mode='markers+lines'
        #         ),
        #         go.Scatter(
        #             x=l_values,
        #             y=scores[:, 0],
        #             name=f'{penalty} penalty Train',
        #             mode='markers+lines'
        #         )
        #     ]
        # ).write_image(f"{penalty}-K-CV.png")

        best_lambda = l_values[np.argmin(scores[:, 1])]
        regularized_logistic = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=2 * 10 ** 4),
                                                  penalty="l1",
                                                  alpha=0.5,
                                                  lam=best_lambda)
        regularized_logistic.fit(train_X, train_y)
        error = regularized_logistic.loss(test_X, test_y)
        print(f"the test-error of {penalty} regularization with {best_lambda} (best lambda) is: {error}")


if __name__ == '__main__':
    np.random.seed(0)
    print("\nq: 1,2,3,4")
    compare_fixed_learning_rates()
    print("\nq: 5, 6, 7")
    compare_exponential_decay_rates()
    print("\nq: 8, 9, 10, 11")
    fit_logistic_regression()
