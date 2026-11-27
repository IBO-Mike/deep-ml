def determinant_4x4(matrix: list[list[int|float]]) -> float:
    d = len(matrix)
    if (d == 1):
        return matrix[0][0]

    det = 0
    f = 1
    for i in range(d):
        subMatrix = [[matrix[row][col] for col in range(d) if col != i] for row in range(1, d)]
        det += f * matrix[0][i] * determinant_4x4(subMatrix)
        f = -f
    return det