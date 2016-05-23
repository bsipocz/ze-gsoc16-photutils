import numpy as np
from astropy.table import Column, Table

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
        If array-like, it should be either (x_0, y_0) or (x_0, y_0, flux_0).
        If 'starlist' only contains x_0 and y_0, 'crit_separation' must be
        provided.
    crit_separation : float (optional)
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.
        If None, 'flux_0' must be provided in 'starlist'.
    
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

    init_group = _find_group(starlist[0], starlist, crit_separation)
    group_starlist.append(init_group)
    assigned_stars_ids = np.intersect1d(starlist['id'], init_group['id'],
                                        assume_unique=True)

    # BEGIN MBI (Must Be Improved)
    # what need to be done: get the indices for which id == assigned_stars_ids
    # so that the rows may be removed all at once
    for i in range(len(assigned_stars_ids)):
        starlist.remove_rows(np.where(starlist['id'] ==\
                                      assigned_stars_ids[i])[0])
    # END 
    
    while len(starlist) is not 0: # is there a corner case?? e.g.
                                  # starlist == None?
        current_group = _find_group(starlist[0], starlist, crit_separation)
        group_starlist.append(current_group)
        assigned_stars_ids = np.intersect1d(starlist['id'],
                                            current_group['id'],
                                            assume_unique=True)
        # BEGIN MBI
        # what need to be done: get the indices for which
        # id == assigned_stars_ids
        # so that the rows may be removed all at once
        for i in range(len(assigned_stars_ids)):
            starlist.remove_rows(np.where(starlist['id'] ==\
                                          assigned_stars_ids[i])[0])
        # END 

    return group_starlist

def _find_group(star, starlist, crit_separation):
    """
    Parameters
    ----------
    star : `~astropy.table.row.Row`
        Star which will be either the head of a cluster or an isolated one.
    
    starlist : `~astropy.table.table.Table`

    Returns
    -------
    """
    
    star_distance = np.hypot(star['x_0'] - starlist['x_0'],
                             star['y_0'] - starlist['y_0'])

    distance_criteria = star_distance < crit_separation

    return starlist[distance_criteria]
