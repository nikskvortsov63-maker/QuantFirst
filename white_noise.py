import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_noise():
    noise = np.random.default_rng().normal(size=1000)
    plt.figure()
    plt.plot(noise)

    # Autocorrelation and Ljung-Box test (jointly tests lags 1 through 50)
    ljung_box = acorr_ljungbox(noise, lags=[50], return_df=True).loc[50]
    plot_acf(noise, lags=50, alpha=0.05,
             title=f"White Noise ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    # McLeod-Li: Ljung-Box test on squared, mean-centered noise
    squared_noise = noise * noise
    mcleod_li = acorr_ljungbox(squared_noise, lags=[50], return_df=True).loc[50]
    plot_acf(squared_noise, lags=50, alpha=0.05,
             title=f"Squared Centered White Noise ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    plt.show()