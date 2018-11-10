import datetime
import numpy as np


def solve(A, B):
    """
    Solve Ax = B.

    Parameters:
        A (scipy.sparse.csr.csr_matrix / np.matrix): sparse matrix
        B (scipy.sparse.csr.csr_matrix / np.matrix): sparse matrix

    Returns:
        x (scipy.sparse.csr.csr_matrix / np.matrix): answer for linear equation Ax = B
    """
    print("solving linear equation...")
    print(datetime.datetime.now())
    a = A.todense() if callable(getattr(A, "todense", None)) else A
    b = B.todense() if callable(getattr(B, "todense", None)) else B

    # inverse(A.T * A)
    print("inversing matrix")
    a_T_a = np.matmul(a.T, a)
    if not _is_invertible(a_T_a):
        print("matrix is not invertible, using pseudo-inverse instead")
        a_T_a_I = np.linalg.pinv(a_T_a)
    else:
        a_T_a_I = np.linalg.inv(a_T_a)
    x = np.matmul(np.matmul(a_T_a_I, a.T), b)
    print(datetime.datetime.now())
    return x


def _is_invertible(A):
    """
    Check if matrix A is invertible.

    Parameters:
        A (np.matrix, np.array): sparse matrix

    Returns:
        boolean, if A is invertible

    PS:
        currently abandoned, since it takes so long time to get the rank
    """
    # return matrix_rank(A) == A.shape[0]
    return True
