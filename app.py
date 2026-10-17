import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


sp_data = yf.download("^GSPC", period="10y", auto_adjust=False, progress=False)
aapl_data = yf.download("AAPL", period="10y", auto_adjust=False, progress=False)
tsla_data = yf.download("TSLA", period="10y", auto_adjust=False, progress=False)
msft_data = yf.download("MSFT", period="10y", auto_adjust=False, progress=False)

fig, axes = plt.subplots(4, 1, sharex=True)

axes[0].plot(sp_data.index, sp_data["Close"])
axes[0].set_ylabel("S&P Close")

axes[1].plot(aapl_data.index, aapl_data["Close"])
axes[1].set_ylabel("AAPL Close")

axes[2].plot(tsla_data.index, tsla_data["Close"])
axes[2].set_ylabel("TSLA Close")

axes[3].plot(msft_data.index, msft_data["Close"])
axes[3].set_ylabel("MSFT Close")

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

plt.show()
