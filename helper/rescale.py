from menpo.transform import Translation, UniformScale
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# variance for dimention (X, Y, Z)
VAR = np.array(config['DEFAULT']['VAR'])

# rescaled center
CENTER = np.array(config['DEFAULT']['CENTER'])


def rescale(mesh, center=CENTER, var=VAR):
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
