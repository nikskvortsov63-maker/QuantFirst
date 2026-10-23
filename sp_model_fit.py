import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA


def fit_sp_arma(close, p=2, q=2):
    log_returns = (100 * np.log(close / close.shift(1))).dropna()
    if log_returns.empty:
        raise ValueError("Yahoo Finance returned no usable S&P 500 log_returns")

    # ARIMA with d=0 is an ARMA model. log_returns are in percentage points.
    fit = ARIMA(log_returns.to_numpy(), order=(p, 0, q), trend="c").fit()
    if not fit.mle_retvals["converged"]:
        raise RuntimeError("ARMA fit did not converge")

    print(fit.summary())

    plt.figure()
    plt.plot(log_returns.index, log_returns, label="S&P 500 log log_returns")
    plt.plot(log_returns.index, fit.fittedvalues, label=f"ARMA({p},{q}) fitted mean")
    plt.ylabel("Log return (%)")
    plt.title(f"In-sample ARMA({p},{q}) fit | AIC={fit.aic:.2f}, BIC={fit.bic:.2f}")
    plt.legend()

    residuals = fit.resid

    plt.figure()
    plt.plot(log_returns.index, residuals)
    plt.title(f"ARMA({p},{q}) Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual (percentage points)")
    ljung_box = acorr_ljungbox(residuals, lags=[50], model_df=p + q, return_df=True).loc[50]
    plot_acf(residuals, lags=50, alpha=0.05,
             title=f"ARMA({p},{q}) Residual ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    squared_residuals = (residuals - residuals.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_residuals, lags=[50], return_df=True).loc[50]
    plot_acf(squared_residuals, lags=50, alpha=0.05,
             title=f"Squared Centered ARMA({p},{q}) Residual ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    #plt.show()
    return fit


def fit_sp_garch(residuals):
    # Residuals should use the same percentage-point units as the ARMA fit.
    # Consider Student-t errors (dist="t") for heavy tails; also update forecast noise.
    model = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    fit = model.fit(disp="off")
    if fit.convergence_flag != 0:
        raise RuntimeError("GARCH fit did not converge")

    print(fit.summary())

    plt.figure()
    plt.plot(fit.conditional_volatility)
    plt.title("S&P 500 ARMA Residuals: Fitted GARCH(1,1) Volatility")
    plt.xlabel("Observation")
    plt.ylabel("Conditional standard deviation (percentage points)")

    standardized_residuals = np.asarray(fit.std_resid)
    standardized_residuals = standardized_residuals[np.isfinite(standardized_residuals)]
    plt.figure()
    plt.plot(standardized_residuals)
    plt.title("GARCH Standardized Residuals")
    plt.xlabel("Observation")
    plt.ylabel("Residual / conditional standard deviation")

    # Diagnostic p-values are approximate after two-stage ARMA-GARCH estimation.
    ljung_box = acorr_ljungbox(standardized_residuals, lags=[50], return_df=True).loc[50]
    plot_acf(standardized_residuals, lags=50, alpha=0.05,
             title=f"Standardized Residual ACF\nLjung-Box Q(50)={ljung_box['lb_stat']:.2f}, p={ljung_box['lb_pvalue']:.3g}")

    squared_residuals = (standardized_residuals - standardized_residuals.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_residuals, lags=[50], return_df=True).loc[50]
    plot_acf(squared_residuals, lags=50, alpha=0.05,
             title=f"Squared Centered Standardized Residual ACF\nMcLeod-Li Q(50)={mcleod_li['lb_stat']:.2f}, p={mcleod_li['lb_pvalue']:.3g}")

    #plt.show()
    return fit


def forecast_sp_prices(arma_fit, garch_fit, last_close, days=500):
    # One simulated future path, conditional on the fitted models and history.
    returns = list(np.asarray(arma_fit.model.endog).ravel())
    errors = list(np.asarray(arma_fit.resid))
    mean = arma_fit.params[arma_fit.param_names.index("const")]
    omega = garch_fit.params["omega"]
    alpha = garch_fit.params["alpha[1]"]
    beta = garch_fit.params["beta[1]"]
    variance = float(np.asarray(garch_fit.conditional_volatility)[-1]) ** 2
    price = float(last_close)
    closing_prices = []

    for day in range(days):
        # ARMA: m[t] = mu + sum(phi[i] * (r[t-i] - mu)) + sum(theta[j] * e[t-j])
        return_mean = mean
        for lag, coefficient in enumerate(arma_fit.arparams, start=1):
            return_mean += coefficient * (returns[-lag] - mean)
        for lag, coefficient in enumerate(arma_fit.maparams, start=1):
            return_mean += coefficient * errors[-lag]

        # GARCH(1,1): sigma[t]^2 = omega + alpha * e[t-1]^2 + beta * sigma[t-1]^2
        variance = omega + alpha * errors[-1] ** 2 + beta * variance
        # z[t] ~ N(0, 1). Student-t noise may be considered for heavy tails.
        noise = np.random.normal(loc=0, scale=1)
        # e[t] = sigma[t] * z[t]; r[t] = m[t] + e[t]
        error = np.sqrt(variance) * noise
        log_return = return_mean + error
        # r[t] = 100 * log(P[t] / P[t-1]), so P[t] = P[t-1] * exp(r[t] / 100).
        price = price * np.exp(log_return / 100)

        returns.append(log_return)
        errors.append(error)
        closing_prices.append(price)

    return np.array(closing_prices)


def plot_sp_forecasts_vs_actual(arma_fit, garch_fit, close_prices, actual_close, num_simulations=1000, days=500):
    last_close = close_prices.iloc[-1]
    actual_close = actual_close.dropna()
    actual_close = actual_close.loc[actual_close.index > close_prices.index[-1]].iloc[:days]

    plt.figure()
    for _ in range(num_simulations):
        prices = forecast_sp_prices(arma_fit, garch_fit, last_close, days)
        prices_with_start = np.concatenate(([last_close], prices))
        plt.plot(range(days + 1), prices_with_start)

    actual_with_start = np.concatenate(([last_close], actual_close.to_numpy()))
    plt.plot(range(len(actual_with_start)), actual_with_start,
             color="black", linewidth=2, label="Actual S&P 500 close")
    plt.xlabel("Trading steps after last fitted close")
    plt.ylabel("Closing index level")
    plt.title(f"{num_simulations} Simulated S&P 500 ARMA-GARCH Paths")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = yf.download("^GSPC", start="2012-01-01", auto_adjust=False, progress=False)
    all_close_prices = data["Close"]["^GSPC"].dropna()
    close_prices = all_close_prices.loc["2012":"2016"]
    actual_close = all_close_prices.loc[all_close_prices.index > close_prices.index[-1]]

    arma_fit = fit_sp_arma(close_prices)
    garch_fit = fit_sp_garch(arma_fit.resid)

    plot_sp_forecasts_vs_actual(arma_fit, garch_fit, close_prices, actual_close)
