import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.converters import QuadraticProgramToQubo

Q = np.array([
    [-29.9905, 20.002, 20.001, 20.003],
    [0.002, -29.9804, 20.004, 20.001],
    [0.001, 0.004, -29.97035, 20.002],
    [0.003, 0.001, 0.002, -29.9753]
])

num_assets = Q.shape[0]

qp = QuadraticProgram()

for i in range(num_assets):
    qp.binary_var(name=f"x{i}")

linear = {f"x{i}": Q[i, i] for i in range(num_assets)}
quadratic = {}

for i in range(num_assets):
    for j in range(i + 1, num_assets):
        quadratic[(f"x{i}", f"x{j}")] = Q[i, j]

qp.minimize(linear=linear, quadratic=quadratic)

qubo = QuadraticProgramToQubo().convert(qp)

exact_solver = NumPyMinimumEigensolver()
optimizer = MinimumEigenOptimizer(exact_solver)

result = optimizer.solve(qubo)

print("\nOptimal Portfolio (Binary Selection):")
print(result.x)
print("Objective Value:", result.fval)

mu = np.array([0.001, 0.0008, 0.0007, 0.0006])
Sigma = np.array([
    [0.01, 0.002, 0.001, 0.003],
    [0.002, 0.02, 0.004, 0.001],
    [0.001, 0.004, 0.03, 0.002],
    [0.003, 0.001, 0.002, 0.025]
])

x = result.x

portfolio_return = x @ mu
portfolio_risk = np.sqrt(x @ Sigma @ x)
sharpe_ratio = portfolio_return / portfolio_risk

print("\n--- Financial Performance ---")
print("Expected Return:", portfolio_return)
print("Portfolio Risk:", portfolio_risk)
print("Sharpe Ratio:", sharpe_ratio)
