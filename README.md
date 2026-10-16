# QuantFirst

Download one year of S&P 500 index (`^GSPC`) data from Yahoo Finance and plot its daily closing price.

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
