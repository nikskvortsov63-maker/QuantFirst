import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_garch_model():
    # X[t] = sigma[t] * noise[t], where noise is standard normal.
    # sigma[t]^2 = omega + alpha * X[t-1]^2 + beta * sigma[t-1]^2
    omega = 0.05
    alpha = 0.1
    beta = 0.85
    garch = arch_model(None, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    simulation = garch.simulate([omega, alpha, beta], nobs=1000, burn=500)
    model = simulation["data"]

    plt.figure()
    plt.plot(model)

    # Autocorrelation and Ljung-Box test (jointly tests lags 1 through 50)
    ljung_box = acorr_ljungbox(model, lags=[50], return_df=True).loc[50]
    plot_acf(model, lags=50, alpha=0.05,
             title=f"GARCH(1,1) ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    # McLeod-Li: Ljung-Box test on squared, mean-centered simulated values
    squared_model = (model - model.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_model, lags=[50], return_df=True).loc[50]
    plot_acf(squared_model, lags=50, alpha=0.05,
             title=f"Squared Centered GARCH(1,1) ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    plt.show()


if __name__ == "__main__":
    plot_garch_model()
