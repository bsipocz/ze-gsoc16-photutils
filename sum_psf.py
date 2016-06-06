def _sum_psf(psf_model, group, **psf_fixed_params):
    """
    This function computes the sum of PSFs with model given by `psf_model` and
    (x_0, y_0, flux_0) given by `group` as a astropy compound model.

    Parameters
    ----------
    psf_model : `~astropy.modeling.Fittable2DModel`
        The PSF/PRF analytical model. This model must have centroid and flux
        as parameters.
    group : `~astropy.table.Table`
        Table from which the compound PSF/PRF will be generated.
        It must have columns named as `x_0`, `y_0`, and `flux_0`.
    psf_fixed_params : dict
        Fixed parameters to be passed to `psf_model`.
    
    Returns
    -------
    group_psf : CompoundModel 
        CompoundModel as the sum of the PSFs/PRFs models.

    See
    ---
    `~daogroup`
    """
    #TODO This is not close to be safe... E.g. have to check which parameters
    # are fixed
    group_psf = psf_model(**psf_fixed_params, flux=group['flux_0'][0],
                          x_0=group['x_0'][0], y_0=group['y_0'][0])
    for i in range(len(group) - 1):
        group_psf += psf_model(**psf_fixed_params, flux=group['flux_0'][i+1],
                               x_0=group['x_0'][i+1], y_0=group['y_0'][i+1])
    return group_psf
