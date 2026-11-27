import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    size = len(A[0])
    x = [0.0] * size
    x_next = [0.0] * size
    for sweep in range(n):
        x_next = [0.0] * size
        for i in range(size):
            x_next[i] = (b[i] - sum(A[i][j] * x[j] for j in range(size) if j != i)) / A[i][i]
        x = x_next
    return [round(value, 4) for value in x]