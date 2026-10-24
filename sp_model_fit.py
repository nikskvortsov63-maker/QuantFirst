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

    residuals = fit.resid

    ljung_box = acorr_ljungbox(residuals, lags=[50], model_df=p + q, return_df=True).loc[50]

    squared_residuals = (residuals - residuals.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_residuals, lags=[50], return_df=True).loc[50]

    return fit


def fit_sp_garch(residuals):
    # Residuals should use the same percentage-point units as the ARMA fit.
    # Consider Student-t errors (dist="t") for heavy tails; also update forecast noise.
    model = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    fit = model.fit(disp="off")
    if fit.convergence_flag != 0:
        raise RuntimeError("GARCH fit did not converge")

    print(fit.summary())

    standardized_residuals = np.asarray(fit.std_resid)
    standardized_residuals = standardized_residuals[np.isfinite(standardized_residuals)]

    # Diagnostic p-values are approximate after two-stage ARMA-GARCH estimation.
    ljung_box = acorr_ljungbox(standardized_residuals, lags=[50], return_df=True).loc[50]

    squared_residuals = (standardized_residuals - standardized_residuals.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_residuals, lags=[50], return_df=True).loc[50]

    return fit


def fit_sp_gjr_garch(residuals):
    # Residuals should use the same percentage-point units as the ARMA fit.
    # Consider Student-t errors (dist="t") for heavy tails; also update forecast noise.
    model = arch_model(residuals, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
    fit = model.fit(disp="off")
    if fit.convergence_flag != 0:
        raise RuntimeError("GJR-GARCH fit did not converge")

    print(fit.summary())

    standardized_residuals = np.asarray(fit.std_resid)
    standardized_residuals = standardized_residuals[np.isfinite(standardized_residuals)]

    # Diagnostic p-values are approximate after two-stage ARMA-GARCH estimation.
    ljung_box = acorr_ljungbox(standardized_residuals, lags=[50], return_df=True).loc[50]

    squared_residuals = (standardized_residuals - standardized_residuals.mean()) ** 2
    mcleod_li = acorr_ljungbox(squared_residuals, lags=[50], return_df=True).loc[50]

    return fit


def forecast_sp_prices(arma_fit, garch_fit, last_close, days=500, gamma=0, num_simulations=1):
    # Rows are trading days; columns are independent simulated paths.
    p = len(arma_fit.arparams)
    q = len(arma_fit.maparams)
    error_lags = max(q, 1)
    returns = np.empty((p + days, num_simulations))
    errors = np.empty((error_lags + days, num_simulations))
    if p:
        returns[:p] = np.asarray(arma_fit.model.endog).ravel()[-p:, None]
    errors[:error_lags] = np.asarray(arma_fit.resid)[-error_lags:, None]

    mean = arma_fit.params[arma_fit.param_names.index("const")]
    omega = garch_fit.params["omega"]
    alpha = garch_fit.params["alpha[1]"]
    beta = garch_fit.params["beta[1]"]
    last_variance = float(np.asarray(garch_fit.conditional_volatility)[-1]) ** 2
    variances = np.full(num_simulations, last_variance)
    prices = np.full(num_simulations, last_close, dtype=float)
    closing_prices = np.empty((days, num_simulations))

    for day in range(days):
        # ARMA mean for every path, using each path's own history.
        return_means = np.full(num_simulations, mean)
        for lag, coefficient in enumerate(arma_fit.arparams, start=1):
            return_means += coefficient * (returns[p + day - lag] - mean)
        for lag, coefficient in enumerate(arma_fit.maparams, start=1):
            return_means += coefficient * errors[error_lags + day - lag]

        # GJR variance; gamma=0 gives standard GARCH.
        previous_errors = errors[error_lags + day - 1]
        variances = omega + (alpha + gamma * (previous_errors < 0)) * previous_errors ** 2 + beta * variances
        noise = np.random.normal(loc=0, scale=1, size=num_simulations)
        new_errors = np.sqrt(variances) * noise
        log_returns = return_means + new_errors
        prices = prices * np.exp(log_returns / 100)

        returns[p + day] = log_returns
        errors[error_lags + day] = new_errors
        closing_prices[day] = prices

    return closing_prices


def forecast_sp_gjr_prices(arma_fit, gjr_fit, last_close, days=500, num_simulations=1):
    return forecast_sp_prices(arma_fit, gjr_fit, last_close, days,
                              gamma=gjr_fit.params["gamma[1]"], num_simulations=num_simulations)


def plot_sp_forecasts_vs_actual(arma_fit, garch_fit, close_prices, actual_close, num_simulations=1000, days=500, forecast_method=forecast_sp_prices):
    last_close = close_prices.iloc[-1]
    actual_close = actual_close.dropna()
    actual_close = actual_close.loc[actual_close.index > close_prices.index[-1]].iloc[:days]

    plt.figure()
    prices = forecast_method(arma_fit, garch_fit, last_close, days, num_simulations=num_simulations)
    starting_prices = np.full((1, num_simulations), last_close)
    prices_with_start = np.concatenate((starting_prices, prices), axis=0)
    plt.plot(range(days + 1), prices_with_start)

    actual_with_start = np.concatenate(([last_close], actual_close.to_numpy()))
    plt.plot(range(len(actual_with_start)), actual_with_start,
             color="black", linewidth=2, label="Actual S&P 500 close")
    plt.xlabel("Trading steps after last fitted close")
    plt.ylabel("Closing index level")
    plt.title(f"{num_simulations} Simulated S&P 500 ARMA-{garch_fit.model.volatility.name} Paths")
    plt.legend()


if __name__ == "__main__":
    data = yf.download("^GSPC", start="2012-01-01", auto_adjust=False, progress=False)
    all_close_prices = data["Close"]["^GSPC"].dropna()
    close_prices = all_close_prices.loc["2012":"2016"]
    actual_close = all_close_prices.loc[all_close_prices.index > close_prices.index[-1]]

    arma_fit = fit_sp_arma(close_prices)

    # ARMA-GARCH
    garch_fit = fit_sp_garch(arma_fit.resid)
    plot_sp_forecasts_vs_actual(arma_fit, garch_fit, close_prices, actual_close, forecast_method=forecast_sp_prices)

    # ARMA-GTR-GARCH
    gjr_fit = fit_sp_gjr_garch(arma_fit.resid)
    plot_sp_forecasts_vs_actual(arma_fit, gjr_fit, close_prices, actual_close, forecast_method=forecast_sp_gjr_prices)

    plt.show()

