import datetime
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from scikits import umfpack


class Solver:
    """
    Solve X = argmin ||AX-B||2, where ||.||2 is Frobenius norm.

    """

    @staticmethod
    def linear_solver(A, B, solver="umfpack"):
        """
        Solve Ax = B , where A, B, x may be all matrices.

        Parameters:
            A (scipy.sparse.csr_matrix): sparse 2d matrix
            B (scipy.sparse.csr_matrix): sparse 2d matrix
            solver (string): umfpack or naive

        Returns:
            X (scipy.sparse.csr_matrix): X
        """

        if solver == "umfpack":
            return Solver._linear_umfpack_solver(A, B)
        elif solver == "naive":
            return Solver._linear_naive_solver(A, B)
        else:
            raise ValueError("Unknown solver, use 'umfpack' or 'naive'")

    @staticmethod
    def _linear_umfpack_solver(A, B):
        """
        Solve Ax = B with UMFPAXK library.

        spsolve needs A to be a square matrix, if not, solve A.T @ A @ x = A.T @ B instead.

        Parameters:
            A (scipy.sparse.csr_matrix): sparse 2d matrix
            B (scipy.sparse.csr_matrix): sparse 2d matrix

        Returns:
            X (scipy.sparse.csr_matrix): X
        """
        if A.shape[0] != A.shape[1]:
            t = A.transpose()
            A = t.dot(A)
            B = t.dot(B)
        return umfpack.spsolve(A, B)

    @staticmethod
    def _linear_naive_solver(A, B):
        """
        Solve Ax = B.

        Parameters:
            A (scipy.sparse.csr_matrix): sparse 2d matrix
            B (scipy.sparse.csr_matrix): sparse 2d matrix

        Returns:
            x (scipy.sparse.csr_matrix): answer for linear equation Ax = B
        """
        print("solving linear equation...")
        print(datetime.datetime.now())

        # inverse(A.T * A)
        print("inversing matrix")
        AT = A.transpose()
        ATA = AT.dot(A)
        if not Solver._is_invertible(ATA):
            print("matrix is not invertible, using pseudo-inverse instead")
            ATAI = sparse.csr_matrix(np.linalg.pinv(ATA.todense()))
        else:
            ATAI = inv(ATA)
        x = ATAI.dot(AT).dot(B)
        print(datetime.datetime.now())
        return x

    @staticmethod
    def _is_invertible(A):
        """
        Check if matrix A is invertible.

        Parameters:
            A (scipy.sparse.csr_matrix): sparse matrix

        Returns:
            boolean, if A is invertible
        """
        return np.linalg.matrix_rank(A.todense()) == A.shape[0]
