import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1️⃣ Runtime Comparison Plot
# -----------------------------
assets = [4, 6, 8, 10]
classical_time = [0.0012, 0.0124, 0.1421, 1.9234]
quantum_time = [0.0501, 0.0513, 0.0530, 0.0541]

plt.figure()
plt.plot(assets, classical_time, marker='o', label='Classical')
plt.plot(assets, quantum_time, marker='s', label='Quantum/QUBO')
plt.xlabel("Number of Assets (N)")
plt.ylabel("Runtime (seconds)")
plt.title("Scalability: Classical vs Quantum Optimization")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 2️⃣ Sharpe Ratio Comparison
# -----------------------------
methods = ["Classical", "Quantum"]
sharpe_ratios = [0.0102, 0.0102]

plt.figure()
plt.bar(methods, sharpe_ratios)
plt.ylabel("Sharpe Ratio")
plt.title("Sharpe Ratio Comparison")
plt.show()

# -----------------------------
# 3️⃣ Portfolio Selection Plot
# -----------------------------
assets_labels = ["Asset 1", "Asset 2", "Asset 3", "Asset 4"]
portfolio = [1, 1, 0, 0]

plt.figure()
plt.bar(assets_labels, portfolio)
plt.ylabel("Selected (1 = Yes, 0 = No)")
plt.title("Quantum-Optimized Portfolio Selection")
plt.show()
