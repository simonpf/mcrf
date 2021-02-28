"""
Common function used for LIRAS a priori.
"""
import numpy as np
import artssat
from artssat.jacobian import Atanh
from artssat.retrieval.a_priori import (And, TropopauseMask, TemperatureMask,
                                        FreezingLevel, Dilate)

"""
Mask limiting retrieval of ice to between 280 K isotherm and the tropopause.
"""
ice_mask = Dilate(And(TropopauseMask(),
                      FreezingLevel(lower_inclusive=True, invert=False)),
                  1)
#               TemperatureMask(0.0, 273.15, lower_inclusive = True))

"""
Mask limiting retrieval of rain to between surface and 264 K isotherm.
"""
rain_mask = Dilate(FreezingLevel(lower_inclusive = False, invert = True), 1)
#TemperatureMask(273.15, 340.0, upper_inclusive = True)

def n0_a_priori(t):
    """
    Functional relation for of the a priori mean of :math:`N_0^*`
    as described in Cazenave et al. 2019.

    Args:
        t: Array containing the temperature profile.

    Returns:
        A priori for :math:`log(N_0^*)`
    """
    t = t - 273.15
    return np.log10(np.exp(-0.076586 * t + 17.948))

def dm_a_priori(t):
    """
    Functional relation for of the a priori mean of :math:`D_m`
    using the DARDAR :math:`N_0^*` a priori and a fixed water
    content of :math:`10^{-6}` kg m:math:`^{-3}`.

    Args:
        t: Array containing the temperature profile.

    Returns:
        A priori for :math:`D_m`
    """
    n0 = 10**n0_a_priori(t)
    iwc = 1e-6
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm

def rh_a_priori(t):
    """
    Functional form of relative humidity a priori.

    Constant value of 70% up to the 270 K isotherm,
    the linearly decreasing until 20% at 220 K and
    constant above.

    Args:
        t: Array containing the temperature profile.

    Returns:
        A priori for relative humidity in transformed
        space.
    """
    upper_limit = 1.1
    lower_limit = 0.0
    transformation = Atanh()
    transformation.z_max = upper_limit
    transformation.z_min = lower_limit
    x = np.maximum(np.minimum(0.7 - (270 - t) / 100.0, 0.7), 0.2)
    return transformation(x)
