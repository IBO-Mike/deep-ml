import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A
    (b11, b12), (b21, b22) = B
    theta = np.arctan2(2 * b12, b11 - b22) / 2
    V = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    Vt = V.T
    W = A @ V
    S = np.sqrt(np.sum(W ** 2, axis=0))
    U = W / S
    if (S[0] < S[1]):
        S = S[1, 0]
        U = U[:, [1, 0]]
        Vt = Vt[[1, 0], :]

    return (U, S, Vt)