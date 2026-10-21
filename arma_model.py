import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_arma_model():
    # X[t] = ar * X[t-1] + noise[t] + ma * noise[t-1]
    # statsmodels uses the opposite sign for AR polynomial coefficients.
    model = arma_generate_sample(
        ar=[1, -0.6, -0.2],
        ma=[1, 0.2],
        nsample=1000,
        scale=np.sqrt(1),
        distrvs=np.random.default_rng().standard_normal,
        burnin=500,
    )

    plt.figure()
    plt.plot(model)
    # Autocorrelation and Ljung-Box test (jointly tests lags 1 through 50)
    ljung_box = acorr_ljungbox(model, lags=[50], return_df=True).loc[50]
    plot_acf(model, lags=50, alpha=0.05,
             title=f"ARMA ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    # McLeod-Li: Ljung-Box test on squared, mean-centered simulated values
    squared_model = (model - model.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_model, lags=[50], return_df=True).loc[50]
    plot_acf(squared_model, lags=50, alpha=0.05,
             title=f"Squared Centered ARMA ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    plt.show()


if __name__ == "__main__":
    plot_arma_model()
