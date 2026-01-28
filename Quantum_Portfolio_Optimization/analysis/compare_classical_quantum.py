import numpy as np
import itertools
import time

# -----------------------------
# Financial Data (from Step 1)
# -----------------------------
mu = np.array([0.001, 0.0008, 0.0007, 0.0006])
Sigma = np.array([
    [0.01, 0.002, 0.001, 0.003],
    [0.002, 0.02, 0.004, 0.001],
    [0.001, 0.004, 0.03, 0.002],
    [0.003, 0.001, 0.002, 0.025]
])

K = 2
lambda_risk = 0.5
N = len(mu)

# -----------------------------
# Classical Brute Force
# -----------------------------
start_time = time.time()

best_value = float("inf")
best_x_classical = None

for combo in itertools.combinations(range(N), K):
    x = np.zeros(N)
    x[list(combo)] = 1
    value = x @ Sigma @ x - lambda_risk * mu @ x
    if value < best_value:
        best_value = value
        best_x_classical = x

classical_time = time.time() - start_time

# Metrics
classical_return = best_x_classical @ mu
classical_risk = np.sqrt(best_x_classical @ Sigma @ best_x_classical)
classical_sharpe = classical_return / classical_risk

# -----------------------------
# Quantum Result (from Step 3)
# -----------------------------
best_x_quantum = np.array([1, 1, 0, 0])  # replace if needed

quantum_return = best_x_quantum @ mu
quantum_risk = np.sqrt(best_x_quantum @ Sigma @ best_x_quantum)
quantum_sharpe = quantum_return / quantum_risk

# -----------------------------
# Display Comparison
# -----------------------------
print("\n====== CLASSICAL vs QUANTUM COMPARISON ======\n")

print("Classical Solution:")
print("Portfolio:", best_x_classical)
print("Return:", classical_return)
print("Risk:", classical_risk)
print("Sharpe Ratio:", classical_sharpe)
print("Runtime (s):", classical_time)

print("\nQuantum Solution:")
print("Portfolio:", best_x_quantum)
print("Return:", quantum_return)
print("Risk:", quantum_risk)
print("Sharpe Ratio:", quantum_sharpe)
