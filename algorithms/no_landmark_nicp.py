import menpo3d
import numpy as np
import scipy.sparse as sp
from menpo.shape import TriMesh
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator

from helper import math_helper


class NonRigidIcp:
    """
    This is an implementation of 'Optimal Step Nonrigid ICP Algorithms for Surface Registration' without landmarks.

    Attributes:
        stiffness_weights (int array or None): stiffness for each iteration
        data_weights (int array or None): data weights for each iteration
		solver (string): 'umfpack' or 'naive'
        eps (float): training precision
        verbose (boolean): whether to print out training info
    """
    def __init__(self, stiffness_weights=None, data_weights=None, solver="umfpack", max_iter=10, eps=1e-3, verbose=True):
        """
        Init non-rigid icp model.

        Parameters:
            stiffness_weights (int array or None): stiffness for each iteration
            data_weights (int array or None): data weights for each iteration
            max_iter (int): max number of iterations for each stiffness
            eps (float): training precision
            verbose (boolean): whether to print out training info
        """
        self.DEFAULT_STIFFNESS_WEIGHTS = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
        self.DEFAULT_DATA_WEIGHTS = [None] * len(self.DEFAULT_STIFFNESS_WEIGHTS)
        self.stiffness_weights = self.DEFAULT_STIFFNESS_WEIGHTS if stiffness_weights is None else stiffness_weights
        self.data_weights = self.DEFAULT_DATA_WEIGHTS if data_weights is None else data_weights
        self.solver = solver
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def non_rigid_icp(self, source, target):
        """
        Non-rigid icp algorithm ignoring landmarks.

        Parameters:
            source (menpo.shape.mesh.base.TriMesh): source mesh to be transformed
            target (menpo.shape.mesh.base.TriMesh): target mesh as the base

        Returns:
            transformed_mesh (menpo.shape.mesh.base.TriMesh): transformed source mesh
            training_info (dict): containing 3 lists of loss/regularized_err/err while training
        """
        n_dims = source.n_dims
        transformed_mesh = source

        M, unique_edge_pairs = self._node_arc_incidence_matrix(source)

        # weight matrix
        G = np.identity(n_dims + 1)

        M_kron_G = sp.kron(M, G)

        # build octree for finding closest points on target.
        target_vtk = trimesh_to_vtk(target)
        closest_points_on_target = VTKClosestPointLocator(target_vtk)

        # log
        training_info = {'loss';[], 'regularized_err':[], 'err':[]}

        for i, (alpha, gamma) in enumerate(zip(self.stiffness_weights, self.data_weights), 1):
            if self.verbose:
                print("Epoch " + str(i) + " with stiffness " + str(alpha))
            transformed_mesh, err_info = self._non_rigid_icp_iter(transformed_mesh, target, closest_points_on_target,
                                                        M_kron_G, alpha, gamma)
            for k in training_info.keys():
                training_info[k] += err_info[k]

        return transformed_mesh, training_info

    def _non_rigid_icp_iter(self, source, target, closest_points_on_target, M_kron_G, alpha, gamma):
        """
        Non-rigid icp for each iteration.

        Parameters:
            source (menpo.shape.mesh.base.TriMesh): original source mesh to be transformed
            target (menpo.shape.mesh.base.TriMesh): target mesh as the base
            closest_points_on_target (menpo3d.vtkutils.VTKClosestPointLocator): octree for finding nearest neighbor
            M_kron_G (scipy.sparse.coo.coo_matrix): matrix M kron matrix G
            alpha (float): stiffness weight
            gamma (float): data weight

        Returns:
            current_instance (menpo.shape.mesh.base.TriMesh): transformed source mesh
            training_info (dict): containing 3 lists of loss/regularized_err/err while training
        """
        # init transformation
        n_dims = source.n_dims
        h_dims = n_dims + 1
        n = source.points.shape[0]
        X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
        v_i = source.points
        edge_tris = source.boundary_tri_index()
        trilist = source.trilist
        target_tri_normals = target.tri_normals()

        # we need to prepare some indices for efficient construction of the D sparse matrix.
        row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(), np.arange(n)))
        x = np.arange(n * h_dims).reshape((n, h_dims))
        col = np.hstack((x[:, :n_dims].ravel(), x[:, n_dims]))
        ones = np.ones(n)
        alpha_M_kron_G = alpha * M_kron_G

        # start iteration
        training_info = {'loss': [], 'regularized_err': [], 'err': []}
        iter_ = 0
        while iter_ < self.max_iter:
            iter_ += 1
            # find nearest neighbour and the normals
            U, tri_indices = closest_points_on_target(v_i)

            data = np.hstack((v_i.ravel(), ones))
            D = sp.coo_matrix((data, (row, col)))

            to_stack_A = [alpha_M_kron_G, D]
            to_stack_B = [np.zeros((alpha_M_kron_G.shape[0], n_dims)), U]

            A = np.array(sp.vstack(to_stack_A).tocsr().todense())
            B = np.array(sp.vstack(to_stack_B).tocsr().todense())

            X = math_helper.Solver.linear_solver(A.T @ A, A.T @ B, self.solver)
            #X = math_helper.solve(np.dot(A.T, A), np.dot(A.T, B), solver=self.solver)

            # deform template
            v_i = np.array(D.dot(X))

            loss = np.linalg.norm(A @ X - B, ord='fro')
            err = np.linalg.norm(X_prev - X, ord='fro')
            regularized_err = err / np.sqrt(np.size(X_prev))

            # log
            training_info['loss'].append(loss)
            training_info['regularized_err'].append(regularized_err)
            training_info['err'].append(err)

            X_prev = X

            if self.verbose:
                info = ' - {} regularized_error: {:.3f} \t {:.3f}'.format(iter_, regularized_err, loss)
                print(info)

            if regularized_err < self.eps:
                break

        current_instance = source.copy()
        current_instance.points = v_i.copy()

        return current_instance, trainging_info

    @staticmethod
    def _node_arc_incidence_matrix(source):
        unique_edge_pairs = source.unique_edge_indices()
        m = unique_edge_pairs.shape[0]

        # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
        row = np.hstack((np.arange(m), np.arange(m)))
        col = unique_edge_pairs.T.ravel()
        data = np.hstack((-1 * np.ones(m), np.ones(m)))
        return sp.coo_matrix((data, (row, col))), unique_edge_pairs
