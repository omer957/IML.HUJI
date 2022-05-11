from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

NUMBER_OF_SAMPLES = 1000

def test_univariate_gaussian():

    est = UnivariateGaussian()

    # Question 1 - Draw samples and print fitted model

    X = np.random.normal(10, 1, NUMBER_OF_SAMPLES)
    est.fit(X)
    print("expectation: ", est.mu_, " , variavce: ", est.var_)
    # for the quiz:
    # print("\nrounded expectation: ", np.round(est.mu_, 3), " , rounded variance: ", np.round(est.var_, 3))


    # Question 2 - Empirically showing sample mean is consistent
    prev_mu_ = est.mu_

    ms = np.linspace(1, NUMBER_OF_SAMPLES, NUMBER_OF_SAMPLES // 10)
    estimated_mean = []
    for i in range(1, 1 + NUMBER_OF_SAMPLES // 10):
        Y = X[:i*10]
        est.fit(Y)
        estimated_mean.append(est.mu_)

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$'),
               go.Scatter(x=ms, y=[prev_mu_] * len(ms), mode='lines', name=r'$\mu$')],
              layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
               xaxis_title="$m\\text{ - number of samples}$", yaxis_title="r$\hat\mu$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    est.fit(X)
    X_pdf = est.pdf(X)

    go.Figure([go.Scatter(x=X, y=X_pdf, mode='markers', name='Empirical PDF', marker=dict(opacity=0.15,
                                                                                          color='LightSkyBlue',
                                                                                          size=7,
                                                                                          line=dict(color="DarkBlue",
                                                                                                    width=2, )
                                                                                          ),
                          )
               ],
              layout=go.Layout(title=r"$\text{Empirical PDF function under the fitted model}$",
                               xaxis_title="sample value", yaxis_title="PDF value",
                               height=300, width=1200)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, NUMBER_OF_SAMPLES)

    est = MultivariateGaussian()
    est.fit(X)

    print("expectation:\n", est.mu_)
    print("variance:\n", est.cov_)
    # for the quiz:
    # print("\nrounded expectation:\n", np.round(est.mu_, 3))
    # print("rounded variance:\n", np.round(est.cov_, 3))

    # Question 5 - Likelihood evaluation

    dot_num = 200
    dots = np.linspace(-10, 10, dot_num)
    combinations = np.array(np.meshgrid(dots, dots)).T.reshape(-1, 2)

    l = lambda x: est.log_likelihood(np.array([x[0], 0, x[1], 0]), sigma, X)

    A = np.apply_along_axis(l, 1, combinations)

    go.Figure(go.Heatmap(x=dots, y=dots, z=A.reshape(dot_num, dot_num)),
              layout=go.Layout(title="heatmap of log-likelihood based on f1, f3 values ",
                               xaxis_title="f3", yaxis_title="f1", height=1000, width=1000)).show()

    # Question 6 - Maximum likelihood

    a = np.argmax(A)
    b = combinations[a, :]
    print("\nthe values for f1 anf f3 that achieved the maximum log-likelihood: ", np.round(b, 3))

    # for checking:
    # print("the calculation for the argmax: ", est.log_likelihood(np.array([b[0], 0, b[1], 0]), sigma, X))
    # print("max in A: ", np.max(A))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
