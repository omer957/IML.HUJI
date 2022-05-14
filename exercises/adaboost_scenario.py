import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (X_train, Y_train), (X_test, Y_test) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    print("Q1 - start")
    adaboost_model = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost_model.fit(X_train, Y_train)

    print('\ngenerating graph...')
    fig1 = go.Figure(
        data=[
            go.Scatter(
                x=np.linspace(1, n_learners, n_learners),
                y=list(map(lambda x: adaboost_model.partial_loss(X_train, Y_train, int(x)), np.linspace(1, n_learners, n_learners))),
                mode='markers+lines',
                name="training error",
                marker=dict(size=5, opacity=0.6),
                line=dict(width=3)
            ),
            go.Scatter(
                x=np.linspace(1, n_learners, n_learners),
                y=list(map(lambda x: adaboost_model.partial_loss(X_test, Y_test, int(x)), np.linspace(1, n_learners, n_learners))),
                mode='markers+lines',
                name="test error",
                marker=dict(size=5, opacity=0.6),
                line=dict(width=3)
            )
        ],
        layout=go.Layout(
            title=f"Loss as function of num of learnesr; with {noise} noise.",
            xaxis_title={'text': "$\\text{Num of learners}$"},
            yaxis_title={'text': "$\\text{Misclassification error}$"}
        )
    )
    fig1.write_image("./q1.png")
    print("\nQ1 - end ************ \n\n")

    # Question 2: Plotting decision surfaces
    print("Q2 - start")

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[X_train, X_test].min(axis=0), np.r_[X_train, X_test].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])

    print('\ngenerating graph...')
    fig2 = make_subplots(rows=2,
                         cols=2,
                         subplot_titles=[f"Decision boundary for ensemble with {t} weak learners" for t in T],
                         horizontal_spacing=0.1,
                         vertical_spacing=0.1)

    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: adaboost_model.partial_predict(x, t), lims[0], lims[1], showscale=False),
                         go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=Y_test,
                                                symbol='diamond',
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(title_text=f"Decision boundary obtained by using the weighted ensembles of different sizes; with {noise} noise.",
                       font_size=15, margin=dict(t=100))

    fig2.write_image("./q2.png")
    print("\nQ2 - end ************ \n\n")

    # Question 3: Decision surface of best performing ensemble
    print("Q3 - start")
    min_ind = np.argmin(np.array([adaboost_model.partial_loss(X_test, Y_test, t) for t in range(1, 251)]))
    best_ensemble_size = min_ind + 1

    acc = accuracy(Y_test, adaboost_model.partial_predict(X_test, best_ensemble_size))
    print('\ngenerating graph...')
    fig3 = go.Figure(
        [decision_surface(lambda x: adaboost_model.partial_predict(x, best_ensemble_size), lims[0], lims[1], showscale=False),
         go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=Y_test,
                                symbol='diamond',
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))],
        layout=go.Layout(
            title=f"the ensemble that achieves the lowest test error is ensemble of size {best_ensemble_size}, with accuracy of: {acc}; with {noise} noise.",
            font_size=15
        )
    )

    fig3.write_image("./q3.png")
    print("\nQ3 - end ************ \n\n")

    # Question 4: Decision surface with weighted samples
    print("Q4 - start")
    D_normal = 5 * adaboost_model.D_ / np.max(adaboost_model.D_)
    print('\ngenerating graph...')
    fig4 = go.Figure(
        [decision_surface(adaboost_model.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=X_train[:, 0], y=X_train[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=Y_test,
                                symbol='diamond',
                                size=D_normal,
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))],
        layout=go.Layout(
            title=f"decision boundary obtained by using the weighted ensembles of size 250; with {noise} noise.",
            font_size=15
        )
    )
    fig4.write_image("./q4.png")
    print("\nQ4 - end ************ \n\n")
    print('*******************************************************************')


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)  # q-5


