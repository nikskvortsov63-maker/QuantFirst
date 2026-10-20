import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_returns():
    sp_data = yf.download("^GSPC", period="10y", auto_adjust=False, progress=False)
    aapl_data = yf.download("AAPL", period="10y", auto_adjust=False, progress=False)
    tsla_data = yf.download("TSLA", period="10y", auto_adjust=False, progress=False)
    msft_data = yf.download("MSFT", period="10y", auto_adjust=False, progress=False)

    # Close prices

    #fig, axes = plt.subplots(4, 1, sharex=True)

    #axes[0].plot(sp_data.index, sp_data["Close"])
    #axes[0].set_ylabel("S&P Close")

    #axes[1].plot(aapl_data.index, aapl_data["Close"])
    #axes[1].set_ylabel("AAPL Close")

    #axes[2].plot(tsla_data.index, tsla_data["Close"])
    #axes[2].set_ylabel("TSLA Close")

    #axes[3].plot(msft_data.index, msft_data["Close"])
    #axes[3].set_ylabel("MSFT Close")

    # Log returns

    fig, axes = plt.subplots(4, 1, sharex=True)

    sp_log_returns = np.log(sp_data["Close"] / sp_data["Close"].shift(1))
    aapl_log_returns = np.log(aapl_data["Close"] / aapl_data["Close"].shift(1))
    tsla_log_returns = np.log(tsla_data["Close"] / tsla_data["Close"].shift(1))
    msft_log_returns = np.log(msft_data["Close"] / msft_data["Close"].shift(1))

    axes[0].plot(sp_log_returns.index, sp_log_returns)
    axes[0].set_ylabel("S&P Log return")

    axes[1].plot(aapl_log_returns.index, aapl_log_returns)
    axes[1].set_ylabel("AAPL Log return")

    axes[2].plot(tsla_log_returns.index, tsla_log_returns)
    axes[2].set_ylabel("TSLA Log return")

    axes[3].plot(msft_log_returns.index, msft_log_returns)
    axes[3].set_ylabel("MSFT Log return")

    # Autocorrelation and Ljung-Box test (jointly tests lags 1 through 50)

    returns = sp_log_returns["^GSPC"].dropna()
    ljung_box = acorr_ljungbox(returns, lags=[50], return_df=True).loc[50]

    plot_acf(returns, lags=50, alpha=0.05,
             title=f"S&P 500 ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    # McLeod-Li: Ljung-Box test on squared, mean-centered returns

    squared_returns = returns * returns
    mcleod_li = acorr_ljungbox(squared_returns, lags=[50], return_df=True).loc[50]

    plot_acf(squared_returns, lags=50, alpha=0.05,
             title=f"S&P 500 Squared Centered Log Returns ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    plt.show()
