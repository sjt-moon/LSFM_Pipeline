from helper import loader, ConfigLoader
from algorithms import no_landmark_nicp
from os import listdir, walk
from os.path import isfile, join
from menpo.model import PCAModel
import numpy as np
from sklearn.decomposition import PCA
from menpo.shape.mesh.base import TriMesh
from functools import reduce
import configparser


class Pipeline:
    """
    LSFM Pipeline.
    """

    def __init__(self, base_model_path, stiffness_weights=None, data_weights=None, solver=None, max_iter=None,
                 eps=None, max_num_points=None, n_components=None, center=None, var=None, verbose=None):
        """
        LSFM Pipeline.

        Parameters:
            base_model_path (string): target mesh path
            stiffness_weights (int array or None): stiffness for each iteration
            data_weights (int array or None): data weights for each iteration
            solver (string): solver for linear equations like Ax = B
            max_iter (int): max number of iterations for each stiffness
            eps (float): training precision
            max_num_points (int): max number of points for each point cloud
            n_components (int, float or None):
                if int, n_components == min(n_samples, n_features);
                if 0 < n_components < 1, select the number of components such that the amount of variance
                    that needs to be explained is greater than the percentage specified by n_components;
                if None, all components are retained
            center (numpy.array of 3 floats): center on 3 dimensions
            var (numpy.array of 3 floats): variance on 3 dimensions
            verbose (boolean): whether to print out training info
        """
        # TODO: @abandoned self.target = loader.get_mean_model(base_model_path)
        # load defaults from config.ini
        config = Pipeline.configuration_check()

        DEFAULT_STIFFNESS_WEIGHTS = ConfigLoader.load_list_numbers(config['DEFAULT']['DEFAULT_STIFFNESS_WEIGHTS'])
        SOLVER = config['DEFAULT']['SOLVER']
        MAX_ITER = int(ConfigLoader.load_number(config['DEFAULT']['MAX_ITER']))
        EPS = ConfigLoader.load_number(config['DEFAULT']['EPS'])
        MAX_NUM_POINTS = ConfigLoader.load_number(config['DEFAULT']['MAX_NUM_POINTS'])
        N_COMPONENTS = ConfigLoader.load_number(config['DEFAULT']['N_COMPONENTS'])
        CENTER = ConfigLoader.load_list_numbers(config['DEFAULT']['CENTER'])
        VAR = ConfigLoader.load_list_numbers(config['DEFAULT']['VAR'])
        VERBOSE = ConfigLoader.load_bool(config['DEFAULT']['VERBOSE'])
        MESH_FILE_EXTENSIONS = ConfigLoader.load_list_strings(config['DEFAULT']['MESH_FILE_EXTENSIONS'])

        self.verbose = verbose if verbose is not None else VERBOSE
        self.center = center if center is not None else CENTER
        self.var = var if var is not None else VAR
        if self.verbose:
            print("\nloading target mesh {}\n".format(base_model_path))
        self.target = loader.get_mesh(base_model_path, self.center, self.var)

        self.stiffness_weights = stiffness_weights if stiffness_weights is not None else DEFAULT_STIFFNESS_WEIGHTS
        self.data_weights = data_weights if data_weights is not None else [None] * len(self.stiffness_weights)
        self.solver = solver if solver is not None else SOLVER
        self.max_iter = max_iter if max_iter is not None else MAX_ITER
        self.eps = eps if eps is not None else EPS
        self.max_num_points = max_num_points if max_num_points is not None else MAX_NUM_POINTS
        self.n_components = n_components if n_components is not None else N_COMPONENTS

        self.nicp_process = no_landmark_nicp.NonRigidIcp(stiffness_weights=self.stiffness_weights,
                                                         data_weights=self.data_weights, solver=self.solver,
                                                         max_iter=self.max_iter, eps=self.eps, verbose=self.verbose)
        self.mesh_samples = [self.target, ]
        self.training_logs = []
        self.mesh_file_extensions = MESH_FILE_EXTENSIONS

    def align(self, input_path):
        """
        Align all the meshes under a certain directory.

        Parameters:
            input_path (string): input directory

        Return:
            aligned_meshes (list of TriMesh): aligned meshes
            trainging_logs (dict of dict): key is mesh file name, value is training logs for that alignment
        """
        aligned_meshes = []
        training_logs = {}
        mesh_files = loader.get_all_mesh_files(input_path, self.mesh_file_extensions, self.verbose)
        for mesh_file in mesh_files:
            if self.verbose:
                print("\nloading mesh file {}\n".format(mesh_file))
            source = loader.get_mesh(mesh_file, self.center, self.var)
            aligned_mesh, training_log = self.nicp_process.non_rigid_icp(source, self.target)
            aligned_meshes.append(aligned_mesh)
            training_logs[mesh_file] = training_log
        if self.verbose:
            print("\n{} meshes aligned to the target".format(len(aligned_meshes)))
            print("average loss: {:.3f}\naverage regularized loss: {:.3f}\n"
                  .format(
                np.mean(reduce(lambda x, y: x + y, map(lambda x: x['loss'], training_logs.values()))),
                np.mean(reduce(lambda x, y: x + y, map(lambda x: x['regularized_loss'], training_logs.values())))
            ))
        return aligned_meshes, training_logs

    def run(self, input_path):
        """
        Align all the meshes under a certain directory.

        Parameters:
            input_path (string): input directory

        Return:
            LSFM model (menpo.model.PCAModel): LSFM model
            trainging_logs (dict of dict): training log while aligning, key is mesh file name,
                value is training logs for that alignment
        """
        aligned_meshes, training_logs = self.align(input_path)
        aligned_meshes += [self.target]
        pca_meshes = self.prune_on_num_points(aligned_meshes)
        return self.pca_prune(pca_meshes), training_logs

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

        if self.verbose:
            print("before trimming on number of points for each mesh, it contains {} points\\mesh\n"
                  "after trimming, it contains {} points\\mesh"
                  .format(N, self.max_num_points))

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
        if self.verbose:
            print("Final PCA Model:\n# of components: {}\n# of points for each mesh (3 dims total): {}"
                  "\neigen value respective ratios: {}\neigen value accumulative ratios: {}"
                  .format(str(pca_model.components.shape[0]),
                          str(pca_model.components.shape[1]),
                          str(pca_model.eigenvalues_ratio()),
                          str(pca_model.eigenvalues_cumulative_ratio())))
        return pca_model

    @staticmethod
    def configuration_check():
        """
        Check whether configuration file exists and is valid.

        Return:
            config (configparser.ConfigParser): config for the pipeline
        """
        assert isfile("config.ini"), "Configuration not found, you should have a configuration file called 'config.ini'"
        config = configparser.ConfigParser()
        config.read("config.ini")

        return config
