import numpy as np

def build_qubo(mu, Sigma, K, lambda_risk=0.5, penalty=10):
    """
    Builds QUBO matrix for portfolio optimization
    """
    N = len(mu)
    Q = np.zeros((N, N))

    # Risk term
    for i in range(N):
        for j in range(N):
            Q[i, j] += Sigma[i, j]

    # Return term
    for i in range(N):
        Q[i, i] -= lambda_risk * mu[i]

    # Constraint penalty: (sum x_i - K)^2
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * K)
        for j in range(i + 1, N):
            Q[i, j] += 2 * penalty

    return Q

if __name__ == "__main__":
    mu = np.array([0.001, 0.0008, 0.0007, 0.0006])
    Sigma = np.array([
        [0.01, 0.002, 0.001, 0.003],
        [0.002, 0.02, 0.004, 0.001],
        [0.001, 0.004, 0.03, 0.002],
        [0.003, 0.001, 0.002, 0.025]
    ])

    Q = build_qubo(mu, Sigma, K=2)
    print("QUBO Matrix:\n", Q)
