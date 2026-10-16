import matplotlib.pyplot as plt
import yfinance as yf


data = yf.download("^GSPC", period="50y", auto_adjust=False, progress=False)
plt.plot(data.index, data["Close"])
plt.show()
