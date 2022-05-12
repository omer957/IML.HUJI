from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

REMOVES_FEATURES = ['id', 'date', 'lat', 'long']
NON_NEGATIVE = ["bathrooms", "floors", "sqft_above", "sqft_basement", "yr_renovated", "view"]
POSITIVE = ["price", "sqft_living", "sqft_lot", "floors", "yr_built", "sqft_lot15", 'zipcode']
CURR_YEAR = 2022


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # read all the data
    full_data = pd.read_csv(filename).drop_duplicates()

    # remove Unnecessary columns
    data = full_data.drop(REMOVES_FEATURES, axis=1)
    data = data.dropna()

    # remove unrational values
    for feature in POSITIVE:
        data = data[data[feature] >= 0]

    for feature in POSITIVE:
        data = data[data[feature] > 0]

    # handle categorical columns - zipcode (only this one, because it is the most indecative)
    data = pd.get_dummies(data, prefix='zipcode ', columns=['zipcode'])

    # check how new the renovation compare to year it was built
    data["renovated_rate"] = (data["yr_renovated"] - data["yr_built"]) / (CURR_YEAR - data["yr_built"])
    data["renovated_rate"] = np.where(data.renovated_rate >= 0, data.renovated_rate, 0)

    response = data['price']
    samples = data.drop("price", axis=1)

    data = data.dropna()

    return samples, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    corr_arr = []
    y_std = np.std(y)
    for feature in X.columns:
        feature_std = np.std(X[feature])
        feature_corr = (np.cov(X[feature], y)[0][1]) / (y_std * feature_std)
        corr_arr.append([feature, feature_corr])

        fig = go.Figure().add_trace(go.Scatter(x=X[feature], y=y, mode='markers',
                                               marker=dict(color="blue",
                                                           opacity=0.15)
                                               )
                                    )
        fig.update_xaxes(title_text=f"feature {feature}", title_font_size=15)
        fig.update_yaxes(title_text="response", title_font_size=15)
        fig.update_layout(
            title_text=f"Peareson correlation of feature {feature} and price ({feature_corr})",
            title_font_size=30, width=1200, height=700)
        # fig.show()

        # file_name = f"/{feature_corr*100}_correlation_of_{feature}_and_price.jpg"
        # fig.write_image(output_path + file_name)
        print("|", end="")  # show proccess


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    samples, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response

    print("\n*** start evaluate ***\n")
    feature_evaluation(samples, response, "plots_feature_evaluation")
    print("\n*** evaluation ended ***\n")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(samples, response, 0.75)
    print("\n*** finish splitting ***\n")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss = []
    std_loss = []

    for p in range(10, 101):

        loss_p = []

        for i in range(10):
            # indexes = np.random.permutation(X_train.index)
            # indexes = indexes[:int(len(indexes) * p / 100)]
            # X_train_p = X_train.loc[indexes]
            # y_train_p = y_train.loc[indexes]

            X_train_p = X_train.sample(frac=p / 100)
            y_train_p = y_train[X_train_p.index]
            # X_train_p = X_train_p.to_numpy()

            est = LinearRegression()
            est._fit(X_train_p.to_numpy(), y_train_p.to_numpy())

            loss_p.append(est._loss(X_test.to_numpy(), y_test.to_numpy()))

        # mean loss
        mean_loss.append(np.mean(loss_p))
        # std_loss
        std_loss.append(np.std(loss_p))

        print("|", end="")  # show proccess

    np_loss_std = np.array(std_loss)
    confidence_interval_p = np.array(mean_loss) + 2 * np_loss_std
    confidence_interval_m = np.array(mean_loss) + (-2) * np_loss_std

    p_values = np.linspace(10, 101, 91)

    fig = go.Figure((go.Scatter(x=p_values, y=mean_loss, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=p_values, y=confidence_interval_p, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=p_values, y=confidence_interval_m, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)))

    fig.update_xaxes(title_text="percentage of samples", title_font_size=15)
    fig.update_yaxes(title_text="mse", title_font_size=15)
    fig.update_layout(
        title_text="MSE as function of percentage of samples",
        title_font_size=30, width=1200, height=700
    )

    fig.show()
    fig.write_image("q4 plot - MSE as function of percentage of samples.png")
    print("\n*** finish the job ***\n")
