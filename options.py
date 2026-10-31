import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def black_scholes_merton(stock_price, strike, years, volatility, rate, dividend_yield):
    # d1 = [ln(S/K) + (r-q+sigma^2/2)*T] / (sigma*sqrt(T)); d2 = d1-sigma*sqrt(T).
    years = np.maximum(years, 0)
    sigma_sqrt_time = volatility * np.sqrt(years)
    d1 = (np.log(stock_price / strike) + (rate - dividend_yield + volatility ** 2 / 2) * years) / np.maximum(sigma_sqrt_time, 1e-12)
    d2 = d1 - sigma_sqrt_time
    discounted_stock = stock_price * np.exp(-dividend_yield * years)
    discounted_strike = strike * np.exp(-rate * years)
    call = discounted_stock * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put = discounted_strike * norm.cdf(-d2) - discounted_stock * norm.cdf(-d1)
    return call, put


def plot_options(risk_free_rate=0.04, dividend_yield=0.005):
    call_symbol = "AAPL261127C00345000"
    put_symbol = "AAPL261127P00345000"

    stock = yf.Ticker("AAPL").history(period="3mo", auto_adjust=False)
    call = yf.Ticker(call_symbol).history(period="1mo", auto_adjust=False)
    put = yf.Ticker(put_symbol).history(period="1mo", auto_adjust=False)
    if stock.empty or call.empty or put.empty:
        raise ValueError("Yahoo Finance returned no price history for AAPL or one of the option contracts")

    # Extra history supplies the rolling window before the displayed month.
    log_returns = np.log(stock["Adj Close"] / stock["Adj Close"].shift(1))
    volatility = log_returns.rolling(20).std() * np.sqrt(252) * 100
    start = (pd.Timestamp.now(tz=stock.index.tz) - pd.DateOffset(months=1)).normalize()
    stock = stock.loc[start:]
    volatility = volatility.loc[start:]

    # Assumed continuous rates; historical volatility is a proxy for future volatility.
    expiry = pd.Timestamp("2026-11-27 16:00", tz="America/New_York")
    observation_times = stock.index.normalize() + pd.Timedelta(hours=16)
    years = (expiry - observation_times).total_seconds() / (365 * 24 * 3600)
    bsm_call, bsm_put = black_scholes_merton(
        stock["Close"], 345, years, volatility / 100, risk_free_rate, dividend_yield,
    )

    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.suptitle(f"BSM comparison: assumed r={risk_free_rate:.1%}, q={dividend_yield:.1%}")

    axes[0].plot(stock.index, stock["Close"])
    axes[0].set_title("AAPL Stock")
    axes[0].set_ylabel("Close (USD/share)")

    axes[1].plot(call.index, call["Close"], label="Observed close")
    axes[1].plot(stock.index, bsm_call, label="BSM European estimate (historical vol)")
    axes[1].legend()
    axes[1].set_title(f"AAPL $345 Call | Expiry 2026-11-27 | {call_symbol}")
    axes[1].set_ylabel("Close (USD/share)")

    axes[2].plot(put.index, put["Close"], label="Observed close")
    axes[2].plot(stock.index, bsm_put, label="BSM European estimate (historical vol)")
    axes[2].legend()
    axes[2].set_title(f"AAPL $345 Put | Expiry 2026-11-27 | {put_symbol}")
    axes[2].set_ylabel("Close (USD/share)")

    axes[3].plot(volatility.index, volatility)
    axes[3].set_title("AAPL 20-Day Historical Volatility (Annualized)")
    axes[3].set_ylabel("Volatility (%)")
    axes[3].set_xlabel("Date")

    
    strike = 345
    expiry_prices = np.linspace(strike * 0.7, strike * 1.3, 301)
    call_payoff = np.maximum(expiry_prices - strike, 0)
    put_payoff = np.maximum(strike - expiry_prices, 0)

    plt.figure()
    plt.plot(expiry_prices, call_payoff, label="Long call")
    plt.plot(expiry_prices, put_payoff, label="Long put")
    plt.axvline(strike, color="gray", linestyle="--", label="Strike $345")
    plt.title("AAPL $345 Options: Payoff at Expiration (Before Premium)")
    plt.xlabel("AAPL stock price at expiration (USD)")
    plt.ylabel("Payoff (USD/share)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_options()
