from helper import loader
from algorithms import no_landmark_nicp
from os import listdir
from os.path import isfile, join
from menpo.model import PCAModel
import numpy as np
from sklearn.decomposition import PCA
from menpo.shape.mesh.base import TriMesh


class Pipeline:
    """
    LSFM Pipeline.
    """
    def __init__(self, base_model_path, stiffness_weights=None, data_weights=None, max_iter=10, eps=1e-3,
                 max_num_points=100, n_components=0.997, verbose=True):
        """
        LSFM Pipeline.

        Parameters:
            base_model_path (string): target mesh path
            stiffness_weights (int array or None): stiffness for each iteration
            data_weights (int array or None): data weights for each iteration
            max_iter (int): max number of iterations for each stiffness
            eps (float): training precision
            max_num_points (int): max number of points for each point cloud
            n_components (int, float or None):
                if int, n_components == min(n_samples, n_features);
                if 0 < n_components < 1, select the number of components such that the amount of variance
                    that needs to be explained is greater than the percentage specified by n_components;
                if None, all components are retained
            verbose (boolean): whether to print out training info
        """
        # TODO: @abandoned self.target = loader.get_mean_model(base_model_path)
        if verbose:
            print("\nloading target mesh {}\n".format(base_model_path));
        self.target = loader.get_mesh(base_model_path)
        self.max_num_points = max_num_points
        self.n_components = n_components
        self.verbose = verbose
        self.nicp_process = no_landmark_nicp.NonRigidIcp(stiffness_weights=stiffness_weights,
                                                         data_weights=data_weights, max_iter=max_iter,
                                                         eps=eps, verbose=verbose)
        self.mesh_samples = [self.target, ]

    def align(self, input_path):
        """
        Align all the meshes under a certain directory.

        Parameters:
            input_path (string): input directory

        Return:
            aligned meshes
        """
        aligned_meshes = []
        mesh_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for mesh_file in mesh_files:
            if not mesh_file.endswith(".obj"):
                continue
            if self.verbose:
                print("\nloading mesh file {}\n".format(mesh_file))
            source = loader.get_mesh(mesh_file)
            aligned_meshes.append(self.nicp_process.non_rigid_icp(source, self.target))
        if self.verbose:
            print("\n{} meshes aligned to the target\n".format(len(aligned_meshes)))
        return aligned_meshes

    def run(self, input_path):
        """
        Align all the meshes under a certain directory.

        Parameters:
            input_path (string): input directory

        Return:
            LSFM model
        """
        aligned_meshes = self.align(input_path) + self.target
        pca_meshes = self.prune_on_num_points(aligned_meshes)
        return self.pca_prune(pca_meshes)

    def prune_on_num_points(self, aligned_meshes):
        """
        PCA on number of points.

        Parameters:
            aligned_meshes (list of TriMesh): meshes to be reduced the number of points

        Return:
            pruned TriMeshes
        """
        assert len(aligned_meshes) > 0, "@prune_on_num_points, no input meshes"
        N, M = aligned_meshes[0].points.shape[0], len(aligned_meshes)
        tmp = self.max_num_points
        if self.max_num_points > N or self.max_num_points > M:
            print("PCA error, max number of points is too large, use {} points instead".format(min(M, N)))
            self.max_num_points = min(M, N)

        # PCA on number of points for each cloud
        X, Y, Z = [], [], []
        for aligned_mesh in aligned_meshes:
            X.append(aligned_mesh.points[:, 0])
            Y.append(aligned_mesh.points[:, 1])
            Z.append(aligned_mesh.points[:, 2])
        # (M, N) matrices, N points M persons
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        pca = PCA(n_components=self.max_num_points)
        X = pca.fit_transform(X)
        Y = pca.fit_transform(Y)
        Z = pca.fit_transform(Z)

        # generate TriMeshes
        pca_meshes = []
        for x, y, z in zip(X, Y, Z):
            pca_meshes.append(TriMesh(points=np.vstack((x, y, z)).T))

        self.max_num_points = tmp

        return pca_meshes

    def pca_prune(self, meshes):
        """
        PCA on TriMeshes features.

        Parameters:
            meshes (list of TriMesh): meshes to be PCA

        Return:
            PCA model
        """
        pca_model = PCAModel(samples=meshes, verbose=self.verbose)
        n_comps_retained = int(sum(pca_model.eigenvalues_cumulative_ratio() < self.n_components)) if \
            self.n_components >= 1 else self.n_components
        if self.verbose:
            print('\nRetaining {:.2%} of eigenvalues keeps {} components'.format(
                self.n_components, n_comps_retained))
        pca_model.trim_components(self.n_components)
        return pca_model
