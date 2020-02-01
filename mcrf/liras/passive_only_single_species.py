"""
A-priori assumptions for the passive-only retrieval with only a single
species of frozen hydrometeors.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.
    rain: Hydrometeor species representing precipitating, liquid hydrometeors.
"""
import os
from mcrf.psds import D14NDmIce, D14NDmLiquid, D14NDmSnow
from mcrf.hydrometeors import Hydrometeor
from parts.retrieval.a_priori import *
from parts.scattering.psd import Binned
from parts.jacobian import Atanh, Log10, Identity, Composition

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

# Reduced altitude grid with resolution of 2 km used in the passive
# only retrieval.
z_grid = np.linspace(0, 20e3, 11)

################################################################################
# Ice particles
################################################################################

def n0_a_priori(t):
    """
    Functional relation for of the a priori mean of :math:`N_0^*`
    as described in Cazenave et al. 2019.
    """
    t = t - 272.15
    return np.log10(np.exp(-0.076586 * t + 17.948))

def dm_a_priori(t):
    """
    Functional relation for of the a priori mean of :math:`D_m`
    using the DARDAR :math:`N_0^*` a priori and a fixed water
    content of :math:`10^{-6}` kg m:math:`^{-3}`.
    """
    n0 = 10**n0_a_priori(t)
    iwc = 1e-6
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm


# The default ice shape
ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# Water content
#

ice_covariance = Diagonal(1 * np.ones(z_grid.size),
                          mask=ice_mask,
                          mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)

#
# n0
#

ice_covariance = Diagonal(1, mask=ice_mask, mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=4)
#ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori,
#                                    points_n0,
#                                    ice_mask,
#                                    "altitude",
#                                    provide_retrieval_grid=False,
#                                    transition=1e3)
ice_n0_a_priori = ReducedVerticalGrid(ice_n0_a_priori,
                                      z_grid,
                                      "altitude",
                                      provide_retrieval_grid = False)

#
# dm
#

points_dm = 5
ice_covariance = Diagonal(300e-6**2, mask=ice_mask, mask_value=1e-16)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-6)
#ice_dm_a_priori = MaskedRegularGrid(ice_dm_a_priori,
#                                    points_dm,
#                                    ice_mask,
#                                    "altitude",
#                                    provide_retrieval_grid=False)
ice_dm_a_priori = ReducedVerticalGrid(ice_dm_a_priori,
                                      z_grid,
                                      "altitude",
                                      provide_retrieval_grid=False)

#
# Hydrometeor definition.
#

ice = Hydrometeor("ice", D14NDmIce(), [ice_n0_a_priori, ice_dm_a_priori],
                  ice_shape, ice_shape_meta)
ice.transformations = [
    Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
    Composition(Identity(), PiecewiseLinear(ice_dm_a_priori))
]
ice.limits_low = [4, 1e-8]
ice.radar_only = False

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

#
# water content
#

rain_covariance = Diagonal(4, mask=rain_mask, mask_value=1e-16)
rain_md_a_priori = FixedAPriori("rain_md", -5, rain_covariance)
rain_md_a_priori = ReducedVerticalGrid(rain_md_a_priori, z_grid, "altitude")

#
# n0
#

rain_covariance = Diagonal(1, mask=rain_mask, mask_value=1e-16)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=2)

#
# dm
#

rain_n0_a_priori = ReducedVerticalGrid(rain_n0_a_priori,
                                      z_grid,
                                      quantity = "altitude",
                                      provide_retrieval_grid = False)
#rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori,
#                                     2,
#                                     rain_mask,
#                                     "altitude",
#                                     provide_retrieval_grid=False)

z_grid = np.linspace(0, 20e3, 6)
rain_covariance = Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-16)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-12)
rain_dm_a_priori = ReducedVerticalGrid(rain_dm_a_priori,
                                       z_grid,
                                       quantity = "altitude",
                                       provide_retrieval_grid = False)
#rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori,
#                                     2,
#                                     rain_mask,
#                                     "altitude",
#                                     provide_retrieval_grid=False)

#
# Hydrometeor definition
#

rain = Hydrometeor("rain", D14NDmLiquid(),
                   [rain_n0_a_priori, rain_dm_a_priori], rain_shape,
                   rain_shape_meta)
rain.transformations = [
    Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
    Composition(Identity(), PiecewiseLinear(rain_dm_a_priori))
]
rain.limits_low = [2, 1e-8]
rain.retrieve_second_moment = True
