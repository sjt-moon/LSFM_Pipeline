import menpo3d
import numpy as np
import scipy.sparse as sp
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
import warnings
from helper import math_helper
import time
import datetime


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
    # reset when resume, as no effect on loss
    # total number of mesh files
    _num_of_meshes = None

    # reset as 0 when resume, as no effect on loss
    # count number of mesh files processed by NonRigidIcp
    _mesh_counter = 0

    # count number of iterations processed by NonRigidIcp
    _iter_counter = 0

    # average regularized loss per iteration
    _average_regularized_loss = 0.0

    def __init__(self, stiffness_weights=(50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2), data_weights=None,
                 solver="umfpack", max_iter=10, eps=1e-3, verbose=True):
        """
        Init non-rigid icp model.

        Parameters:
            stiffness_weights (int array or None): stiffness for each iteration
            data_weights (int array or None): data weights for each iteration
            max_iter (int): max number of iterations for each stiffness
            eps (float): training precision
            verbose (boolean): whether to print out training info
        """
        self.stiffness_weights = stiffness_weights
        self.data_weights = data_weights if data_weights is not None else [None] * len(stiffness_weights)
        assert len(self.stiffness_weights) == len(self.data_weights), \
            "number of stiffness weights doesn't match to numbers of data weights."

        assert solver.lower() in {"umfpack", "naive"}, "Unknown solver, only umfpack and naive solvers are supported"
        self.solver = solver.lower()
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        self._expected_remaining_time = None

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
        start_time = time.time()

        # one more mesh file processed
        NonRigidIcp._mesh_counter += 1

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
        training_info = {'loss': [], 'regularized_loss': []}

        for i, (alpha, gamma) in enumerate(zip(self.stiffness_weights, self.data_weights), 1):
            if self.verbose:
                print("Epoch " + str(i) + " with stiffness " + str(alpha))
            transformed_mesh, err_info = self._non_rigid_icp_iter(transformed_mesh, target, closest_points_on_target,
                                                                  M_kron_G, alpha, gamma)
            for k in training_info.keys():
                training_info[k] += err_info[k]

        end_time = time.time()
        mesh_training_time = end_time - start_time
        if NonRigidIcp._num_of_meshes is not None:
            self._expected_remaining_time = str(datetime.timedelta(seconds=mesh_training_time * (NonRigidIcp._num_of_meshes - NonRigidIcp._mesh_counter)))
        else:
            self._expected_remaining_time = str(mesh_training_time) + " x # of mesh files"

        if self.verbose:
            print("average loss: {:.3f}\naverage regularized loss: {:.3f}\nexpected remaining time: {}"
                  .format(np.mean(training_info['loss']),
                          np.mean(training_info['regularized_loss']),
                          self._expected_remaining_time
                          )
                  )

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
        v_i = source.points

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
            NonRigidIcp._iter_counter += 1

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

            loss = np.linalg.norm(A @ X - B, ord='fro')
            regularized_loss = loss / len(source.points)
            training_info['loss'].append(loss)
            training_info['regularized_loss'].append(regularized_loss)

            NonRigidIcp._average_regularized_loss = (NonRigidIcp._iter_counter - 1) * \
                                                    NonRigidIcp._average_regularized_loss / NonRigidIcp._iter_counter

            if self.verbose:
                info = ' - {} loss: {:.3f} regularized_loss: {:.3f}  '.format(iter_, loss, regularized_loss)
                print(info)
            else:
                progress_bar = "["
                if NonRigidIcp._num_of_meshes is not None:
                    progress = int(10.0 * NonRigidIcp._mesh_counter / NonRigidIcp._num_of_meshes)
                    for _ in range(progress-1):
                        progress_bar += "="
                    progress_bar += ">"
                    for _ in range(10 - progress - 1):
                        progress_bar += "."
                    progress_bar += "] " + str(NonRigidIcp._mesh_counter) + "/" + str(NonRigidIcp._num_of_meshes)
                else:
                    progress_bar += str(NonRigidIcp._num_of_meshes) + "]"
                if self._expected_remaining_time is not None:
                    progress_bar += " | remaining time: " + self._expected_remaining_time

                print(("loss @ this iter: {:.3f} | "
                      "loss/iter: {:.3f} | "
                       + progress_bar)
                      .format(regularized_loss,
                              NonRigidIcp._average_regularized_loss
                              ), end="\r", flush=True)

            if regularized_loss < self.eps:
                break

        current_instance = source.copy()
        current_instance.points = v_i.copy()

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

    # deprecated
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

    @classmethod
    def set_num_of_meshes(cls, num_of_meshes):
        cls._num_of_meshes = num_of_meshes

    @classmethod
    def set_iter_counter(cls, iter_counter):
        cls._iter_counter = iter_counter
