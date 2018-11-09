from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
from menpo.shape import TriMesh
import menpo3d

from helper import rescale


def get_mean_model(model_path):
    """
    Get scaled mean BFM mesh.

    Parameters:
        model_path (string): path to BFM model

    Returns:
        mean_mesh (menpo.shape.mesh.base.TriMesh): mean mesh model
    """
    return rescale.rescale(_get_mean_model(_load_BFM(model_path)))


def get_mesh(mesh_path):
    """
    Get scaled mesh from mesh files like .obj.

    Parameters:
        mesh_path (string): path to mesh file

    Returns:
        mesh object (menpo.shape.mesh.base.TriMesh): mean mesh model
    """
    return rescale.rescale(_get_mesh(mesh_path))


def _get_mesh(mesh_path):
    """
    Get mesh from mesh files like .obj.

    Parameters:
        mesh_path (string): path to mesh file

    Returns:
        mesh object (menpo.shape.mesh.base.TriMesh): mean mesh model
    """
    return menpo3d.io.import_mesh(mesh_path)


def _load_BFM(model_path):
    """
    Load BFM 3DMM model.


    Parameters:
        model_path (string): path to BFM model.

    Returns:
        model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
            'shapeMU': [3*nver, 1]
            'shapePC': [3*nver, 199]
            'shapeEV': [199, 1]
            'expMU': [3*nver, 1]
            'expPC': [3*nver, 29]
            'expEV': [29, 1]
            'texMU': [3*nver, 1]
            'texPC': [3*nver, 199]
            'texEV': [199, 1]
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
            'kpt_ind': [68,] (start from 1)
    """
    C = sio.loadmat(model_path)
    model = C['model']
    model = model[0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order='C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order='C').astype(np.int32) - 1

    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model


def _get_mean_model(model):
    """
    Get mean BFM mesh.

    Parameters:
        model (BFM dict model): BFM model

    Returns:
        mean_mesh (menpo.shape.mesh.base.TriMesh): mean mesh model
    """
    return TriMesh(points=np.reshape(model['shapeMU'], (-1, 3)), trilist=model['tri'])
