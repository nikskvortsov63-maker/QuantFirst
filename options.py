import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


def black_scholes_merton(
    stock_prices: float | pd.Series | np.ndarray,
    strike: float,
    time_remaining: float | np.ndarray | pd.Index,
    volatility: float | pd.Series | np.ndarray,
    rate: float,
    dividend_yield: float,
) -> tuple[float | pd.Series | np.ndarray, float | pd.Series | np.ndarray]:
    # d1 = [ln(S/K) + (r-q+sigma^2/2)*T] / (sigma*sqrt(T)); d2 = d1-sigma*sqrt(T).
    time_remaining = np.maximum(time_remaining, 0)
    sigma_sqrt_time = volatility * np.sqrt(time_remaining)
    d1 = (np.log(stock_prices / strike) + (rate - dividend_yield + volatility ** 2 / 2) * time_remaining) / np.maximum(sigma_sqrt_time, 1e-12)
    d2 = d1 - sigma_sqrt_time
    discounted_stock = stock_prices * np.exp(-dividend_yield * time_remaining)
    discounted_strike = strike * np.exp(-rate * time_remaining)
    call = discounted_stock * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put = discounted_strike * norm.cdf(-d2) - discounted_stock * norm.cdf(-d1)
    return call, put


def binomial_european(
    stock_price: float,
    strike: float,
    time_remaining: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
    steps: int = 500,
) -> tuple[float, float]:
    if steps < 1 or not isinstance(steps, int):
        raise ValueError("steps must be a positive integer")
    if stock_price <= 0 or strike <= 0 or time_remaining < 0 or volatility < 0:
        raise ValueError("Prices must be positive; time and volatility must be non-negative")
    if time_remaining == 0:
        return max(stock_price - strike, 0), max(strike - stock_price, 0)
    if volatility == 0:
        final_price = stock_price * np.exp((rate - dividend_yield) * time_remaining)
        discount = np.exp(-rate * time_remaining)
        return discount * max(final_price - strike, 0), discount * max(strike - final_price, 0)

    # 1. Stock moves up by u or down by d during each time step.
    dt = time_remaining / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up

    # 2. Risk-neutral probability, not a prediction of the real up probability.
    probability_up = (np.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not 0 <= probability_up <= 1:
        raise ValueError("Increase steps so the risk-neutral probability is between 0 and 1")
    discount = np.exp(-rate * dt)

    # 3. Start with option payoffs at expiration: node = number of up moves.
    call_values = np.zeros(steps + 1)
    put_values = np.zeros(steps + 1)
    for node in range(steps + 1):
        final_price = stock_price * up ** node * down ** (steps - node)
        call_values[node] = max(final_price - strike, 0)
        put_values[node] = max(strike - final_price, 0)

    # 4. Work backward: V = exp(-r*dt) * (p*V_up + (1-p)*V_down).
    # European options have no early-exercise decision.
    for step in range(steps - 1, -1, -1):
        for node in range(step + 1):
            call_values[node] = discount * (probability_up * call_values[node + 1]
                                           + (1 - probability_up) * call_values[node])
            put_values[node] = discount * (probability_up * put_values[node + 1]
                                          + (1 - probability_up) * put_values[node])

    return call_values[0], put_values[0]


def black_scholes_merton_greeks(
    stock_price: float,
    strike: float,
    time_remaining: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    if time_remaining <= 0 or volatility <= 0:
        raise ValueError("Greeks require positive time to expiry and volatility")

    sqrt_time = np.sqrt(time_remaining)
    d1 = (np.log(stock_price / strike) + (rate - dividend_yield + volatility ** 2 / 2) * time_remaining) / (volatility * sqrt_time)
    d2 = d1 - volatility * sqrt_time
    stock_discount = np.exp(-dividend_yield * time_remaining)
    strike_discount = np.exp(-rate * time_remaining)
    density = norm.pdf(d1)

    call_delta = stock_discount * norm.cdf(d1)
    put_delta = stock_discount * (norm.cdf(d1) - 1)
    gamma = stock_discount * density / (stock_price * volatility * sqrt_time)
    vega = stock_price * stock_discount * density * sqrt_time / 100

    time_decay = -stock_price * stock_discount * density * volatility / (2 * sqrt_time)
    call_theta = (time_decay - rate * strike * strike_discount * norm.cdf(d2)
                  + dividend_yield * stock_price * stock_discount * norm.cdf(d1)) / 365
    put_theta = (time_decay + rate * strike * strike_discount * norm.cdf(-d2)
                 - dividend_yield * stock_price * stock_discount * norm.cdf(-d1)) / 365

    call_rho = strike * time_remaining * strike_discount * norm.cdf(d2) / 100
    put_rho = -strike * time_remaining * strike_discount * norm.cdf(-d2) / 100

    # Per share: theta per calendar day; vega/rho per 1 percentage-point change.
    return pd.DataFrame({
        "Call": [call_delta, gamma, call_theta, vega, call_rho],
        "Put": [put_delta, gamma, put_theta, vega, put_rho],
    }, index=["Delta", "Gamma", "Theta / day", "Vega / 1pp IV", "Rho / 1pp rate"])


def plot_options(
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.005,
) -> None:
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
    time_remaining = (expiry - observation_times).total_seconds() / (365 * 24 * 3600)
    bsm_call, bsm_put = black_scholes_merton(
        stock["Close"], 345, time_remaining, volatility / 100, risk_free_rate, dividend_yield,
    )

    greeks = black_scholes_merton_greeks(
        float(stock["Close"].iloc[-1]), 345, float(time_remaining[-1]),
        float(volatility.iloc[-1] / 100), risk_free_rate, dividend_yield,
    )
    print(f"\nBSM Greeks as of {stock.index[-1]:%Y-%m-%d} (per share; European approximation)")
    print(greeks.round(6))

    tree_calls = []
    tree_puts = []
    for day in range(len(stock)):
        tree_call, tree_put = binomial_european(
            float(stock["Close"].iloc[day]), 345, float(time_remaining[day]),
            float(volatility.iloc[day] / 100), risk_free_rate, dividend_yield,
        )
        tree_calls.append(tree_call)
        tree_puts.append(tree_put)

    comparison = pd.DataFrame({
        "BSM": [float(bsm_call.iloc[-1]), float(bsm_put.iloc[-1])],
        "European tree (500 steps)": [tree_calls[-1], tree_puts[-1]],
    }, index=["Call", "Put"])
    print("\nEuropean pricing comparison (USD/share):")
    print(comparison.round(6))

    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.suptitle(f"European pricing comparison: assumed r={risk_free_rate:.1%}, q={dividend_yield:.1%}")

    axes[0].plot(stock.index, stock["Close"])
    axes[0].set_title("AAPL Stock")
    axes[0].set_ylabel("Close (USD/share)")

    axes[1].plot(call.index, call["Close"], label="Observed close")
    axes[1].plot(stock.index, bsm_call, label="BSM European estimate (historical vol)")
    axes[1].plot(stock.index, tree_calls, linestyle="--", label="European tree (500 steps)")
    axes[1].legend()
    axes[1].set_title(f"AAPL $345 Call | Expiry 2026-11-27 | {call_symbol}")
    axes[1].set_ylabel("Close (USD/share)")

    axes[2].plot(put.index, put["Close"], label="Observed close")
    axes[2].plot(stock.index, bsm_put, label="BSM European estimate (historical vol)")
    axes[2].plot(stock.index, tree_puts, linestyle="--", label="European tree (500 steps)")
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
