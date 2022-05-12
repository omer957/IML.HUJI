import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    data = pd.read_csv(filename, parse_dates=['Date'])

    # remove Unnecessary columns

    data = data.drop_duplicates().dropna()

    data['DayOfYear'] = data['Date'].dt.day_of_year

    # removing no rational samples
    for f in ['Month', 'Day', 'Year']:
        data = data[data[f] > 0]

    data = data[data['Month'] < 13]
    data = data[data['Day'] < 32]
    data = data[data['Temp'] > -20]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    print("\n *** finish q 1 ***")

    # Question 2 - Exploring data for specific country
    data_israel = data[data["Country"] == 'Israel']

    # data_israel['Year'] = data_israel['Year'].astype(str)

    fig1 = px.scatter(data_israel, x='DayOfYear', y='Temp', color='Year')

    fig1.update_xaxes(title_text="day of the year", title_font_size=15)
    fig1.update_yaxes(title_text="temp", title_font_size=15)
    fig1.update_layout(
        title_text="Daily Temperatue in Israel (function of day of the year)",
        title_font_size=30, width=1200, height=700
    )
    fig1.write_image("q2.1 - Daily Temperatue in Israel (function of day of the year).png")
    # fig1.show()

    data_israel_month = data_israel.groupby("Month").agg(np.std)

    fig2 = px.bar(data_israel_month, y='Temp')
    fig2.update_layout(
        title_text="monthly standard deviation of the daily temperatures in Israel",
        title_font_size=30, width=1200, height=700
    )
    fig2.write_image("q2.2 - monthly Temperatue in Israel.png")
    # fig2.show()

    print("\n *** finish q 2 ***")


    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()

    data_country_month = data.groupby(['Country', 'Month'])["Temp"].agg([np.mean, np.std]).reset_index()

    fig3 = px.line(data_country_month, x='Month', y='mean', error_y='std', color='Country')
    fig3.update_layout(
        title_text="Average monthly temperature, with error in different countries",
        title_font_size=30, width=1200, height=700
    )
    fig3.write_image("q3 - Average monthly temperature, with error in different.png")
    # fig3.show()

    print("\n *** finish q 3 ***")
    # Question 4 - Fitting model for different values of `k`

    X_train, y_train, X_test, y_test = split_train_test(data_israel["DayOfYear"], data_israel["Temp"], 0.75)

    loss = []

    for k in range(1, 11):
        poly_est = PolynomialFitting(k)
        poly_est._fit(X_train.to_numpy(), y_train.to_numpy())
        k_loss = poly_est._loss(X_test.to_numpy(), y_test.to_numpy())
        loss.append(k_loss)
        print(f"test error for degree k = {k} is: ", k_loss)

    dots = np.linspace(1, 10, 10)

    fig4 = px.bar(x=dots, y=loss)

    fig4.update_xaxes(title_text="degree of the polynom", title_font_size=15)
    fig4.update_yaxes(title_text="loss", title_font_size=15)
    fig4.update_layout(
        title_text="q4 - Loss as function of the degree of the polynom",
        title_font_size=30, width=1200, height=700
    )
    fig4.write_image("q4 - Loss as function of the degree of the polynom.png")
    # fig4.show()

    print("\n *** finish q 4 ***")
    # Question 5 - Evaluating fitted model on different countries

    countries = ['Jordan', 'South Africa', 'The Netherlands']
    countries_loss = []

    final_k = 5
    poly_est = PolynomialFitting(final_k)
    poly_est._fit(data_israel["DayOfYear"], data_israel["Temp"])

    for country in countries:
        data_country = data[data["Country"] == country]
        loss = poly_est._loss(data_country["DayOfYear"].to_numpy(), data_country["Temp"].to_numpy())
        countries_loss.append(loss)

    fig5 = px.bar(x=countries, y=countries_loss)
    fig5.update_xaxes(title_text="countries", title_font_size=15)
    fig5.update_yaxes(title_text="loss", title_font_size=15)
    fig5.update_layout(
        title_text="q5 - Loss in different countries when using model with k=5 fitted on Israel",
        title_font_size=20, width=1200, height=700
    )
    fig5.write_image("q5.png")

    print("\n *** finish q 5 ***")







