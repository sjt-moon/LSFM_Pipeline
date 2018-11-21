from menpo.transform import Translation, UniformScale
import numpy as np


def rescale(mesh, center, var):
    """
    Rescale mesh according to given variances.

    Parameters:
        mesh (menpo.shape.mesh.base.TriMesh): mesh to be rescaled
        center (numpy.array of 3 floats): center on 3 dimensions
        var (numpy.array of 3 floats): variance on 3 dimensions

    Returns:
        mesh (menpo.shape.mesh.base.TriMesh): rescaled source
    """
    tr = Translation(center)
    sc = UniformScale(np.mean(var / mesh.range()), 3)
    prepare = tr.compose_after(sc)

    return prepare.apply(mesh)
