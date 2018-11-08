import menpo3d
import numpy as np
import scipy.sparse as sp
from menpo.shape import TriMesh
from menpo.transform import Translation, UniformScale
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator

from helper import math_helper


class NonRigidIcp:
    """
    This is an implementation of 'Optimal Step Nonrigid ICP Algorithms for Surface Registration' without landmarks.

    Attributes:
        stiffness_weights (int array or None): stiffness for each iteration
        data_weights (int array or None): data weights for each iteration
        eps (float): training precision
        verbose (boolean): whether to print out training info
    """
    def __init__(self, stiffness_weights=None, data_weights=None, max_iter=10, eps=1e-3, verbose=True):
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
        """
        source, target, scaling_model = self._rescale(source, target)
        n_dims = source.n_dims
        v_i = source.points
        transformed_mesh = source

        M, unique_edge_pairs = self._node_arc_incidence_matrix(source)

        # weight matrix
        G = np.identity(n_dims + 1)

        M_kron_G = sp.kron(M, G)

        # build octree for finding closest points on target.
        target_vtk = trimesh_to_vtk(target)
        closest_points_on_target = VTKClosestPointLocator(target_vtk)

        for i, (alpha, gamma) in enumerate(zip(self.stiffness_weights, self.data_weights), 1):
            if self.verbose:
                print("Epoch " + str(i) + " with stiffness " + str(alpha))
            transformed_mesh = self._non_rigid_icp_iter(v_i, source, target, closest_points_on_target,
                                                        M_kron_G, alpha, gamma)

        return transformed_mesh

    def _non_rigid_icp_iter(self, v_i, source, target, closest_points_on_target, M_kron_G, alpha, gamma):
        """
        Non-rigid icp for each iteration.

        Parameters:
            v_i (numpy.ndarray): current transformed points
            source (menpo.shape.mesh.base.TriMesh): original source mesh to be transformed
            target (menpo.shape.mesh.base.TriMesh): target mesh as the base
            closest_points_on_target (menpo3d.vtkutils.VTKClosestPointLocator): octree for finding nearest neighbor
            M_kron_G (scipy.sparse.coo.coo_matrix): matrix M kron matrix G
            alpha (float): stiffness weight
            gamma (float): data weight

        Returns:
            current_instance (menpo.shape.mesh.base.TriMesh): transformed source mesh
        """
        # init transformation
        n_dims = source.n_dims
        h_dims = n_dims + 1
        n = source.points.shape[0]
        X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
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
        iter_ = 0
        current_instance = source.copy()
        while iter_ < self.max_iter:
            iter_ += 1
            # find nearest neighbour and the normals
            U, tri_indices = closest_points_on_target(v_i)

            # ---- WEIGHTS ----
            # 1.  Edges
            # Are any of the corresponding tris on the edge of the target?
            # Where they are we return a false weight (we *don't* want to
            # include these points in the solve)
            w_i_e = np.in1d(tri_indices, edge_tris, invert=True)

            # 2. Normals
            # Calculate the normals of the current v_i
            v_i_tm = TriMesh(v_i, trilist=trilist)
            v_i_n = v_i_tm.vertex_normals()
            # Extract the corresponding normals from the target
            u_i_n = target_tri_normals[tri_indices]
            # If the dot of the normals is lt 0.9 don't contrib to deformation
            w_i_n = (u_i_n * v_i_n).sum(axis=1) > 0.9

            # Form the overall w_i from the normals, edge case
            # for now disable the edge constraint (it was noisy anyway)
            w_i = w_i_n

            prop_w_i = (n - sum(w_i) * 1.0) / n
            prop_w_i_n = (n - sum(w_i_n) * 1.0) / n
            prop_w_i_e = (n - sum(w_i_e) * 1.0) / n

            if gamma is not None:
                w_i *= gamma

            # Build the sparse diagonal weight matrix
            W = sp.diags(np.array(w_i).astype(np.float)[None, :], [0])

            data = np.hstack((v_i.ravel(), ones))
            D = sp.coo_matrix((data, (row, col)))

            to_stack_A = [alpha_M_kron_G, W.dot(D)]
            to_stack_B = [np.zeros((alpha_M_kron_G.shape[0], n_dims)),
                          U * w_i[:, None]]  # nullify nearest points by w_i

            A = sp.vstack(to_stack_A).tocsr()
            B = sp.vstack(to_stack_B).tocsr()
            X = math_helper.solve(A, B)

            # deform template
            v_i = np.array(D.dot(X))

            err = np.linalg.norm(X_prev - X, ord='fro')
            regularized_err = err / np.sqrt(np.size(X_prev))

            X_prev = X

            current_instance = source.copy()
            current_instance.points = v_i.copy()

            if self.verbose:
                info = ' - {} regularized_error: {:.3f}  ' \
                       'total: {:.0%}  norms: {:.0%}  ' \
                       'edges: {:.0%}'.format(iter_,
                                              regularized_err,
                                              prop_w_i,
                                              prop_w_i_n,
                                              prop_w_i_e)
                print(info)

            if regularized_err < self.eps:
                break

        return current_instance

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
    def _rescale(source, target):
        """
        Rescale source and target meshes.

        Parameters:
            source (menpo.shape.mesh.base.TriMesh): source mesh to be transformed
            target (menpo.shape.mesh.base.TriMesh): target mesh as the base

        Returns:
            source (menpo.shape.mesh.base.TriMesh): rescaled source
            target (menpo.shape.mesh.base.TriMesh): rescaled target mesh
            scaling_model (menpo.transform.homogeneous.similarity.Similarity): scaling model
        """
        tr = Translation(-1 * source.centre())
        sc = UniformScale(1.0 / np.sqrt(sum(source.range() ** 2)), 3)
        prepare = tr.compose_before(sc)

        source = prepare.apply(source)
        target = prepare.apply(target)

        # store how to undo the similarity transform
        scaling_model = prepare.pseudoinverse()

        return source, target, scaling_model
