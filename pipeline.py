from helper import loader
from algorithms import no_landmark_nicp
from os import listdir
from os.path import isfile, join
from menpo.model import PCAModel


class Pipeline:
    def __init__(self, base_model_path, stiffness_weights=None, data_weights=None, max_iter=10, eps=1e-3, verbose=True):
        self.target = loader.get_mean_model(base_model_path)
        self.verbose = verbose
        self.nicp_process = no_landmark_nicp.NonRigidIcp(stiffness_weights=stiffness_weights,
                                                         data_weights=data_weights, max_iter=max_iter,
                                                         eps=eps, verbose=verbose)
        self.mesh_samples = [self.target, ]

    def align(self, input_path):
        aligned_meshes = []
        mesh_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for mesh_file in mesh_files:
            source = loader.get_mesh(mesh_file)
            aligned_meshes.append(self.nicp_process.non_rigid_icp(source, self.target))
        return aligned_meshes

    def run(self, input_path):
        aligned_meshes = self.align(input_path)
        pca_model = PCAModel(samples=self.mesh_samples + aligned_meshes, verbose=self.verbose)
        return pca_model
