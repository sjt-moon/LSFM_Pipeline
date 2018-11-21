import datetime
import numpy as np
from scipy import sparse
from scikits import umfpack


class Solver:
    """
    Solve X = argmin ||AX-B||2, where ||.||2 is Frobenius norm.

    """
    def __init__(self, A, B, eps=1e-3, max_iter=10, solver="umfpack", verbose=True):
        assert solver in {"umfpack", "naive"}, "Unknown solver, use umfpack or naive."

        A = A.todense() if callable(getattr(A, "todense", None)) else np.array(A)
        B = B.todense() if callable(getattr(B, "todense", None)) else np.array(B)

        assert len(A.shape) == 2 and len(B.shape) == 2, "Shape of matrix A and B should be 2D array."
        assert A.shape[0] == B.shape[0], "Shape of matrix A and B does not match."

        self.A = A
        self.B = B
        self.eps = eps
        self.max_iter = max_iter
        self.solver = solver
        self.verbose = verbose

    def solve(self):
        """
        Solve Ax = B.

        Returns:
            x (np.array): answer for linear equation Ax = B
        """
        A, B = self.A, self.B
        m, n = A.shape[1], B.shape[1]
        X = np.zeros((m, n))
        for iter_ in range(1, self.max_iter + 1):
            error = np.sum((A @ X - B) ** 2)
            if self.verbose:
                print("iter: {}, error: {:.2f}".format(iter_, error))
            if error <= self.eps:
                break
            # iterate
            X = Solver.linear_solver(A.T @ A, A.T @ B, self.solver)
        return X

    @staticmethod
    def linear_solver(A, B, solver="umfpack"):
        if solver == "umfpack":
            return Solver._linear_umfpack_solver(A, B)
        elif solver == "naive":
            return Solver._linear_naive_solver(A, B)
        else:
            print("Unknown solver")

    @staticmethod
    def _linear_umfpack_solver(A, B):
        """
        Solve Ax = B with UMFPAXK library.

        Parameters:
            A (np.array): 2d matrix
            B (np.array): 2d matrix

        Returns:
            X (np.array): X
        """
        return umfpack.spsolve(sparse.csr_matrix(A), sparse.csr_matrix(B))

    @staticmethod
    def _linear_naive_solver(A, B):
        """
        Solve Ax = B.

        Parameters:
            A (np.array): 2d matrix
            B (np.array): 2d matrix

        Returns:
            x (np.array): answer for linear equation Ax = B
        """
        print("solving linear equation...")
        print(datetime.datetime.now())

        # inverse(A.T * A)
        print("inversing matrix")
        a_T_a = np.matmul(A.T, A)
        if not _is_invertible(a_T_a):
            print("matrix is not invertible, using pseudo-inverse instead")
            a_T_a_I = np.linalg.pinv(a_T_a)
        else:
            a_T_a_I = np.linalg.inv(a_T_a)
        x = np.matmul(np.matmul(a_T_a_I, A.T), B)
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
