from astropy.table import Table, vstack
import numpy as np
from sum_psf import _sum_psf
from daogroup import daogroup
from plotutils import _show_region
from photutils.aperture_core import _prepare_photometry_input
import matplotlib.pyplot as plt

def nstar(image, groups, shape, fitter, psf_model, **psf_kwargs):
    """
    Fit, as appropriate, a compound or single model to the given `groups` of
    stars. Groups are fitted sequentially from the smaller to the bigger. In
    each iteration, `image` is subtracted by the previous fitted group. 
    
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
    psf_kwargs : dict
        Fixed parameters to be passed to `psf_model`.

    Return
    ------
    result_tab : `~astropy.table.Table`
        Astropy table that contains the results of the photometry.
    image : numpy.ndarray
        Residual image.
    """
    
    result_tab = Table([[],[],[],[]], names=('id','x_fit','y_fit','flux_fit'),
                       dtype=('i4','f8','f8','f8'))
    models_order = _get_models_order(groups) 
    while len(models_order) > 0:
        curr_order = np.min(models_order)
        n = 0
        N = len(models_order)
        while(n < N):
            if curr_order == len(groups[n]):
                group_psf = _sum_psf(psf_model, groups[n], **psf_kwargs)
                
                x, y, data = _extract_shape_and_data(shape, groups[n], image)
                fitted_model = fitter(group_psf, x, y, data)
                param_table = _model_params_to_table(fitted_model, groups[n])
                result_tab = vstack([result_tab, param_table])

                image = _subtract_psf(image, x, y, fitted_model)
                models_order.remove(curr_order)
                del groups[n]
                N = N - 1
                ###
                patch = _show_region([(np.min(x), np.min(y)),
                                      (np.min(x), np.max(y)),
                                      (np.max(x), np.max(y)),
                                      (np.max(x), np.min(y)),
                                      (np.min(x), np.min(y)),])
                plt.gca().add_patch(patch)
                ###
            n = n + 1
    return result_tab, image


def _model_params_to_table(fitted_model, group):
    param_tab = Table([[],[],[],[]], names=('id','x_fit','y_fit','flux_fit'),
                      dtype=('i4','f8','f8','f8'))
    if np.size(fitted_model) == 1:
        tmp_table = Table([[group['id'][0]],
                           [getattr(fitted_model,'x_0').value],
                           [getattr(fitted_model, 'y_0').value],
                           [getattr(fitted_model, 'flux').value]],
                           names=('id','x_fit', 'y_fit', 'flux_fit'))
        param_tab = vstack([param_tab, tmp_table])
    else:
        for i in range(np.size(fitted_model)):
            tmp_table = Table([[group['id'][i]],
                               [getattr(fitted_model,'x_0_'+str(i)).value],
                               [getattr(fitted_model, 'y_0_'+str(i)).value],
                               [getattr(fitted_model, 'flux_'+str(i)).value]],
                               names=('id','x_fit', 'y_fit', 'flux_fit'))
            param_tab = vstack([param_tab, tmp_table])

    return param_tab


def _extract_shape_and_data(shape, group, image):
    """
    Parameters
    ----------
    shape : tuple
        Shape of a rectangular region around the center of an isolated source.
    group : `astropy.table.Table`
        Group of stars
    image : numpy.ndarray

    Returns
    -------
    x, y : numpy.mgrid
        All coordinate pairs (x,y) in a rectangular region which encloses all
        sources of the given group
    image : numpy.ndarray
        Pixel value
    """

    xmin = int(np.around(np.min(group['x_0'])) - shape[0])
    xmax = int(np.around(np.max(group['x_0'])) + shape[0])
    ymin = int(np.around(np.min(group['y_0'])) - shape[1])
    ymax = int(np.around(np.max(group['y_0'])) + shape[1])
    y,x = np.mgrid[ymin:ymax+1, xmin:xmax+1]

    return x, y, image[ymin:ymax+1, xmin:xmax+1]


def _get_models_order(groups):
    """
    Parameters
    ----------
    groups : list
        List of groups of mutually overlapping stars.

    Returns
    -------
    model_order : list
        List of the orders (i. e., number of stars) per group.
    """

    model_order = []
    for i in range(len(groups)):
        model_order.append(len(groups[i]))
    return model_order

# No need for this. Should use photutils.psf.subtract_psf.
def _subtract_psf(image, x, y, fitted_model):
    psf_image = np.zeros(image.shape)
    psf_image[y,x] = fitted_model(x,y)
    return image - psf_image
