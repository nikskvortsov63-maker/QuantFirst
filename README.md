# QuantFirst

Download 50 years of S&P 500 index (`^GSPC`) data from Yahoo Finance and plot daily closing prices, simple returns, and log returns. Returns are decimals: 0.01 means 1%.

## Setup (PowerShell)

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

In VS Code, select `.venv\Scripts\python.exe` using **Python: Select Interpreter**, then run `app.py`.
