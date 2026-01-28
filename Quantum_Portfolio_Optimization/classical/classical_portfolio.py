import yfinance as yf
import pandas as pd
import numpy as np
import itertools

# -----------------------------
# Step 1: Select stocks
# -----------------------------
stocks = ["AAPL", "MSFT", "GOOG", "AMZN"]

# -----------------------------
# Step 2: Download data
# -----------------------------
raw_data = yf.download(
    stocks,
    start="2022-01-01",
    end="2024-01-01"
)

# -----------------------------
# Step 3: Robust Close price extraction
# -----------------------------
if isinstance(raw_data.columns, pd.MultiIndex):
    # Detect where 'Close' exists
    if "Close" in raw_data.columns.get_level_values(0):
        # ('Close', 'AAPL') format
        data = raw_data["Close"]
    else:
        # ('AAPL', 'Close') format
        data = raw_data.xs("Close", level=1, axis=1)
else:
    data = raw_data["Close"]

print("\nStock Close Prices:")
print(data.head())

# -----------------------------
# Step 4: Daily returns
# -----------------------------
returns = data.pct_change().dropna()

# -----------------------------
# Step 5: Expected returns & covariance
# -----------------------------
mu = returns.mean().values
Sigma = returns.cov().values

# -----------------------------
# Step 6: Classical brute-force optimization
# -----------------------------
K = 2
lambda_risk = 0.5
N = len(mu)

best_value = float("inf")
best_portfolio = None

for combo in itertools.combinations(range(N), K):
    x = np.zeros(N)
    x[list(combo)] = 1
    objective = x @ Sigma @ x - lambda_risk * mu @ x

    if objective < best_value:
        best_value = objective
        best_portfolio = x

print("\nBest Classical Portfolio:")
print(best_portfolio)
print("Objective Value:", best_value)
