import numpy as np
import itertools
import time
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.converters import QuadraticProgramToQubo

def classical_bruteforce(mu, Sigma, K, lam):
    N = len(mu)
    best = float("inf")
    for combo in itertools.combinations(range(N), K):
        x = np.zeros(N)
        x[list(combo)] = 1
        value = x @ Sigma @ x - lam * mu @ x
        best = min(best, value)
    return best

def quantum_solver(mu, Sigma, K, lam):
    N = len(mu)
    Q = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            Q[i, j] += Sigma[i, j]
        Q[i, i] -= lam * mu[i]
        Q[i, i] += 10 * (1 - 2 * K)
        for j in range(i + 1, N):
            Q[i, j] += 20

    qp = QuadraticProgram()
    for i in range(N):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": Q[i, i] for i in range(N)}
    quadratic = {(f"x{i}", f"x{j}"): Q[i, j]
                 for i in range(N) for j in range(i + 1, N)}

    qp.minimize(linear=linear, quadratic=quadratic)
    qubo = QuadraticProgramToQubo().convert(qp)

    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    solver.solve(qubo)

# -----------------------------
# Run Scalability Test
# -----------------------------
lambda_risk = 0.5
K = 2

print("\nN | Classical Time (s) | Quantum Time (s)")
print("----------------------------------------")

for N in [4, 6, 8, 10]:
    mu = np.random.rand(N) * 0.001
    Sigma = np.random.rand(N, N)
    Sigma = Sigma @ Sigma.T  # make positive definite

    start = time.time()
    classical_bruteforce(mu, Sigma, K, lambda_risk)
    classical_time = time.time() - start

    start = time.time()
    quantum_solver(mu, Sigma, K, lambda_risk)
    quantum_time = time.time() - start

    print(f"{N} | {classical_time:.4f} | {quantum_time:.4f}")
