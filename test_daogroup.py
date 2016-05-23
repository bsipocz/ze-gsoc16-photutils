import numpy as np
from numpy.testing import assert_array_equal
from astropy.table import Table, vstack
from daogroup import daogroup

def test_daogroup():
    x_0 = np.array([0, np.sqrt(2)/4, np.sqrt(2)/4, -np.sqrt(2)/4,
                    -np.sqrt(2)/4])
    y_0 = np.array([0, np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4,
                    -np.sqrt(2)/4])
    x_1 = x_0 + 2.0

    first_group = Table([x_0, y_0, np.arange(len(x_0))],
                    names=('x_0', 'y_0', 'id'))
    second_group = Table([x_1, y_0, len(x_0) + np.arange(len(x_0))],
                     names=('x_0', 'y_0', 'id'))
    group_starlist = [first_group, second_group]
    starlist = vstack([first_group, second_group])

    assert_array_equal(daogroup(starlist, crit_separation=0.6),
                       group_starlist)
