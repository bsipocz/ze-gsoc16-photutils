from astropy.table import Table
import numpy as np
from sum_psf import _sum_psf
from daogroup import daogroup
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
rcParams['image.cmap'] = 'viridis'
rcParams['image.aspect'] = 1  # to get images with square pixels
rcParams['figure.figsize'] = (20,10)
rcParams['image.interpolation'] = 'none'

# First fit isolated sources, then proceed fitting smaller groups
def nstar(image, groups, shape, fitter, psf_model, **kwargs):
    """
    Fit, as appropriate, a compound or single model to the given `groups` of
    stars.
    
    Parameters
    ----------
    image : numpy.ndarray
        Background-subtracted image.
    groups : list of `~astropy.table.Table`
        Each `~astropy.table.Table` in this list corresponds to a group of
        mutually overlapping starts.
    shape : tuple
        Shape of a rectangular region around the center of an isolated source.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        An instance of an `~astropy.modeling.fitting.Fitter`
        See `~astropy.modeling.fitting` for details about fitters.
    psf_model : `~astropy.modeling.Fittable2DModel` 
        The PSF/PRF analytical model. This model must have centroid and flux
        as parameters.
    kwargs : dict
        Fixed parameters to be passed to `psf_model`.

    Return
    ------
    residual_image : numpy.ndarray
    result_tab : `~astropy.table.Table`
    """

    result_tab = Table()
    models_order = _get_models_order(groups) 
    while len(models_order) > 0:
        curr_order = np.min(models_order)
        n = 0
        N = len(models_order)
        while(n < N):
            if curr_order == len(groups[n]):
                group_psf = _sum_psf(psf_model, groups[n], **kwargs)
                x, y, data = _extract_shape_and_data(shape, groups[n], image)
                fitted_model = fitter(group_psf, x, y, data) 
                image = _subtract_psf(image, x, y, fitted_model)
                models_order.remove(curr_order)
                del groups[n]
                N = N - 1
                patch = _show_region([(np.min(x), np.min(y)),
                                      (np.min(x), np.max(y)),
                                      (np.max(x), np.max(y)),
                                      (np.max(x), np.min(y)),
                                      (np.min(x), np.min(y)),])
                plt.gca().add_patch(patch)
            n = n + 1
    return image


def _extract_shape_and_data(shape, group, image):
    """
    """
    xmin = int(np.around(np.min(group['x_0'])) - shape[0])
    xmax = int(np.around(np.max(group['x_0'])) + shape[0])
    ymin = int(np.around(np.min(group['y_0'])) - shape[1])
    ymax = int(np.around(np.max(group['y_0'])) + shape[1])
    y,x = np.mgrid[ymin:ymax+1, xmin:xmax+1]

    return x, y, image[ymin:ymax+1, xmin:xmax+1]


def _show_region(verts):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=1)
    return patch


def _get_models_order(groups):
    model_order = []
    for i in range(len(groups)):
        model_order.append(len(groups[i]))

    return model_order


def _subtract_psf(image, x, y, fitted_model):
    psf_image = np.zeros(image.shape)
    psf_image[y,x] = fitted_model(x,y)
    return image - psf_image
