from menpo.transform import Translation, UniformScale
import numpy as np

# variance for dimention (X, Y, Z)
VAR = np.array([85, 300, 220])

# rescaled center
CENTER = np.array([0, 0, 0])


def rescale(mesh, center=CENTER, var=VAR):
    """
    Rescale mesh according to given variances.

    Parameters:
        mesh (menpo.shape.mesh.base.TriMesh): mesh to be rescaled
        center (list of 3 floats): center on 3 dimensions
        var (list of 3 floats): variance on 3 dimensions

    Returns:
        mesh (menpo.shape.mesh.base.TriMesh): rescaled source
    """
    tr = Translation(center)
    sc = UniformScale(sum(var ** 2) / sum(mesh.range() ** 2), 3)
    prepare = tr.compose_after(sc)

    return prepare.apply(mesh)
