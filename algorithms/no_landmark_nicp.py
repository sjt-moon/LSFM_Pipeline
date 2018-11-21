import menpo3d
import numpy as np
import scipy.sparse as sp
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
import warnings
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
        training_info = {'loss':[], 'regularized_loss':[]}

        for i, (alpha, gamma) in enumerate(zip(self.stiffness_weights, self.data_weights), 1):
            if self.verbose:
                print("Epoch " + str(i) + " with stiffness " + str(alpha))
            transformed_mesh, err_info = self._non_rigid_icp_iter(transformed_mesh, target, closest_points_on_target,
                                                        M_kron_G, alpha, gamma)
            for k in training_info.keys():
                training_info[k] += err_info[k]

        if self.verbose:
            print(training_info)

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
        #X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
        v_i = source.points

        #edge_tris = source.boundary_tri_index()
        #trilist = source.trilist
        #target_tri_normals = target.tri_normals()

        # we need to prepare some indices for efficient construction of the D sparse matrix.
        row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(), np.arange(n)))
        x = np.arange(n * h_dims).reshape((n, h_dims))
        col = np.hstack((x[:, :n_dims].ravel(), x[:, n_dims]))
        ones = np.ones(n)
        alpha_M_kron_G = alpha * M_kron_G

        # start iteration
        training_info = {'loss': [], 'regularized_loss': []}
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

            # deform template
            v_i = np.array(D.dot(X))

            #delta_x = np.linalg.norm(X_prev - X, ord='fro')
            loss = np.linalg.norm(A @ X - B, ord='fro')
            regularized_loss = loss / len(source.points)
            training_info['loss'].append(loss)
            training_info['regularized_loss'].append(regularized_loss)

            #X_prev = X

            if self.verbose:
                info = ' - {} loss: {:.3f} regularized_loss: {:.3f}  '.format(iter_, loss, regularized_loss)
                print(info)

            if regularized_loss < self.eps:
                break

        current_instance = source.copy()
        current_instance.points = v_i.copy()

        # NO TODO: current_instance.points = index_sort(current_instance.points, target.points, U)

        return current_instance, training_info

    @staticmethod
    def _node_arc_incidence_matrix(source):
        unique_edge_pairs = source.unique_edge_indices()
        m = unique_edge_pairs.shape[0]

        # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
        row = np.hstack((np.arange(m), np.arange(m)))
        col = unique_edge_pairs.T.ravel()
        data = np.hstack((-1 * np.ones(m), np.ones(m)))
        return sp.coo_matrix((data, (row, col))), unique_edge_pairs

    @staticmethod
    def index_sort(source_points, source_trilist, target_points, closest_points):
        """
        Sort source points and trilist to match target.

        Parameters:
            source_points (numpy ndarray): source points
            source_trilist (numpy ndarray): source trilist
            target_points (numpy ndarray): target points
            closest_points (numpy ndarray): the closest target points for each source point

        Returns:
            source_points_sorted (numpy ndarray): sorted source points
            source_trilist_sorted (numpy ndarray): sorted source trilist indices
        """
        # TODO: many to one mapping
        warnings.warn("deprecated", DeprecationWarning)

        source_points_sorted = [None] * len(source_points)
        source_trilist_sorted = [None] * len(source_trilist)

        # closest point on target: target point index
        point2IndexMap = {}
        for i, point in enumerate(target_points):
            point2IndexMap[hash(point)] = i

        # old source point index: new source point index after sorting
        old2newIndexMap = {}
        for i in range(min(len(source_points), len(closest_points))):
            old2newIndexMap[i] = point2IndexMap[hash(closest_points[i])]

        # rearrange source points
        for k, v in old2newIndexMap.items():
            source_points_sorted[v] = source_points[k]

        # rearrange source trilist
        for i in range(len(source_trilist)):
            source_trilist_sorted[i] = [old2newIndexMap[e] for e in source_trilist[i]]

        return source_points_sorted, source_trilist_sorted

    @staticmethod
    def hash(array):
        """
        Hash an array.

        Parameters:
            array (numpy array or list of floats): array to be hashed

        Returns:
            hash (string)
        """
        return ' '.join([str(x) for x in array])
