import numpy as np
from numpy.testing import assert_equal
from astropy.modeling import Model
from astropy.table import Table
from photutils.psf import IntegratedGaussianPRF
from sum_psf import _sum_psf
from daogroup import daogroup

def test_sum_psf():
    x_0 = np.array([0, np.sqrt(2)/4, np.sqrt(2)/4, -np.sqrt(2)/4,
                    -np.sqrt(2)/4])
    y_0 = np.array([0, np.sqrt(2)/4, -np.sqrt(2)/4, np.sqrt(2)/4,
                    -np.sqrt(2)/4])

    starlist = Table([np.arange(len(x_0)), x_0, y_0, np.arange(len(x_0))+10],
                     names=('id', 'x_0', 'y_0', 'flux_0'))
    starlist_psf = _sum_psf(IntegratedGaussianPRF, starlist, sigma=2.0)
    groups = daogroup(starlist, crit_separation=3.0*2.355)
    
    assert_equal(isinstance(starlist_psf, Model), True)
    assert_equal(starlist_psf.flux_0.value, groups[0]['flux_0'][0])
