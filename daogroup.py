import numpy as np
from astropy.table import Column, Table, vstack

def daogroup(starlist, crit_separation=None):
    """
    This is an implementation which follows the DAO GROUP algorithm presented
    by Stetson (1987).

    GROUP divides an entire starlist into sets of distinct, self-contained
    groups of mutually overlapping stars.

    GROUP accepts as input a list of stars and their approximate
    brightenesses relative to a model stellar intensity profile for the frame
    and determines which stars are close enough to be capable of
    adversely influencing each others' profile fits.

    Parameters
    ----------
    starlist : `~astropy.table.Table` or array-like
        List of stars positions.
        If `~astropy.Table`, columns should be named 'x_0' and 'y_0'.
        Additionally, 'flux_0' may also be provided. 
        TODO: If array-like, it should be either (x_0, y_0) or
        (x_0, y_0, flux_0).
        If 'starlist' only contains x_0 and y_0, 'crit_separation' must be
        provided.
    crit_separation : float (optional)
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.
        TODO: If None, 'flux_0' must be provided in 'starlist'.
    
    Returns
    -------
    group_starlist : list of `~astropy.Table`
        Each `~astropy.Table` in the list corresponds to a group of mutually
        overlapping starts.

    Notes
    -----
    Assuming the psf fwhm to be known, 'crit_separation' may be set to
    k*fwhm, for some positive real k.

    See
    ---
    `~daofind`
    """

    group_starlist = []

    if 'id' not in starlist.colnames:
        starlist.add_column(Column(name='id', data=np.arange(len(starlist))))
    
    while len(starlist) is not 0:
        init_group = _find_group(starlist[0], starlist, crit_separation)
        assigned_stars_ids = np.intersect1d(starlist['id'], init_group['id'],
                                            assume_unique=True)
        starlist = _remove_stars(starlist, assigned_stars_ids)
        N = len(init_group)
        n = 1
        while(n < N):    
            tmp_group = _find_group(init_group[n], starlist, crit_separation)
            if len(tmp_group) > 0:
                assigned_stars_ids = np.intersect1d(starlist['id'],
                                                    tmp_group['id'],
                                                    assume_unique=True)
                starlist = _remove_stars(starlist, assigned_stars_ids)
                init_group = vstack([init_group, tmp_group])
                N = len(init_group)
            n = n + 1
        group_starlist.append(init_group)
    return group_starlist

def _find_group(star, starlist, crit_separation):
    """
    Find those stars in `starlist` which are at a distance of
    `crit_separation` from `star`.

    Parameters
    ----------
    star : `~astropy.table.row.Row`
        Star which will be either the head of a cluster or an isolated one.
    
    starlist : `~astropy.table.table.Table`

    Returns
    -------
    `~astropy.table.table.Table` containing those stars which are at a distance
    of `crit_separation` from `star`.
    """
    
    star_distance = np.hypot(star['x_0'] - starlist['x_0'],
                             star['y_0'] - starlist['y_0'])
    distance_criteria = star_distance < crit_separation
    return starlist[distance_criteria]

def _remove_stars(starlist, stars_ids):
    """
    Remove stars whose id is `stars_ids` from `starlist`.

    Parameters
    ----------
    starlist : `~astropy.table.table.Table`

    stars_ids : numpy.ndarray

    Returns
    -------
    `~astropy.table.table.Table`
    """
    
    for i in range(len(stars_ids)):
        starlist.remove_rows(np.where(starlist['id'] == stars_ids[i])[0])
    return starlist
