from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from utils import *
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    full_data = np.load(filename)
    return full_data[:, 0:2], full_data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    data_set_directory = "../datasets/"

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset(data_set_directory + f)
        losses = []

        def my_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, Y))

        # Fit Perceptron and record loss in each fit iteration
        p = Perceptron(callback=my_callback)
        p.fit(X, Y)
        # print(losses)

        # Plot figure
        dots = np.linspace(1, len(losses), len(losses))
        fig = go.Figure().add_trace(go.Scatter(x=dots, y=losses, mode='lines'))
        fig.update_xaxes(title_text="misclassification error loss", title_font_size=15)
        fig.update_yaxes(title_text="number of updates", title_font_size=15)
        fig.update_layout(title_text=f"Loss as  function of iteration - {n}", title_font_size=30, width=1200, height=700)
        file_name = f"{n}_graph.jpg"
        output_path = "./ex3/perceptron/"
        fig.write_image(output_path + file_name)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    data_set_directory = "../datasets/"
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset

        X, Y = load_dataset(data_set_directory + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, Y)
        lda_pred = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, Y)
        gnb_pred = gnb.predict(X)

        from IMLearn.metrics import accuracy
        lda_acc = accuracy(Y, lda_pred)
        gnb_acc = accuracy(Y, gnb_pred)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian naive bayes - accuracy: {gnb_acc}", f"Linear discriminant analysis - accuracy: {lda_acc}"),
                            horizontal_spacing=0.2,
                            column_widths=[15, 15],
                            row_heights=[6])
        # GNB
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color=gnb_pred, symbol=Y, opacity=.75)),
                      row=1, col=1)
        # theLDA
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers",
                                 showlegend=False,
                                 marker=dict(color=lda_pred, symbol=Y, opacity=.75)),
                      row=1, col=2)
        fig.update_annotations(font_size=10)
        fig.update_layout(title_text=f"Bayes Classifiers Comparison over data set: {f}",
                          font_size=10,
                          margin=dict(t=100))

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()
        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()

        for plot_col, model in enumerate([gnb, lda]):
            for c in range(model.classes_.shape[0]):
                fig.add_trace(go.Scatter(x=[model.mu_[c][0]], y=[model.mu_[c][1]],
                                         mode="markers",
                                         showlegend=False,
                                         marker=dict(color="black", symbol="x", size=8)),
                              row=1, col=plot_col + 1)

                cov = lda.cov_ if model == lda else np.diag(gnb.vars_[c])
                fig.add_trace(get_ellipse(model.mu_[c], cov), row=1, col=plot_col + 1)

        file_name = f"{f}_graph_pred_comparison.jpg"
        output_path = "./ex3/comparison/"
        fig.write_image(output_path + file_name)
    # print("finish")

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
