"""
A-priori assumptions for the passive-only retrieval.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.
    snow: Hydrometeor species representing precipitating, frozen hydrometeors.
    rain: Hydrometeor species representing precipitating, liquid hydrometeors.
    rh_a_priori: A priori provider for humidity retrieval.
    cloud_water_a_priori: A priori provider for cloud water retrieval.
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
    iwc = 1e-5
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm

# The default ice shape
ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# Water content
#

md_z_grid = np.linspace(0, 20e3, 5)
ice_mask = And(TropopauseMask(), TemperatureMask(0.0, 272.15))
snow_mask = And(AltitudeMask(0.0, 18e3), TemperatureMask(0.0, 272.5))
ice_covariance = Diagonal(1 * np.ones(md_z_grid.size))

#
# n0
#

points_n0 = 2
ice_covariance = Diagonal(1, mask=ice_mask, mask_value=1e-12)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=2)
ice_n0_a_priori = ReducedVerticalGrid(ice_n0_a_priori,
                                      z_grid,
                                      "altitude",
                                      provide_retrieval_grid=False)

points_dm = 5
ice_covariance = Diagonal(300e-6**2, mask=ice_mask, mask_value=1e-16)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-6)
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
# Snow particles
################################################################################

# Default shape
snow_shape = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")

snow_mask = And(TropopauseMask(), TemperatureMask(0.0, 280.0))
snow_covariance = Diagonal(4 * np.ones(md_z_grid.size))

#
# n0
#

snow_covariance = Diagonal(0.25, mask=ice_mask, mask_value=1e-12)
snow_n0_a_priori = FixedAPriori("snow_n0",
                                7,
                                snow_covariance,
                                mask=ice_mask,
                                mask_value=0)
snow_n0_a_priori = ReducedVerticalGrid(snow_n0_a_priori,
                                       points_n0,
                                       ice_mask,
                                       "altitude",
                                       provide_retrieval_grid=False)

#
# dm
#

snow_covariance = Diagonal(500e-6**2, mask=ice_mask, mask_value=1e-16)
snow_covariance = SpatialCorrelation(snow_covariance, 4e3, mask=ice_mask)
snow_dm_a_priori = FixedAPriori("snow_dm",
                                1000e-6,
                                snow_covariance,
                                mask=ice_mask,
                                mask_value=1e-6)
snow_dm_a_priori = ReducedVerticalGrid(snow_dm_a_priori,
                                       z_grid,
                                       "altitude",
                                       provide_retrieval_grid=False)

#
# Hydrometeor definition
#

snow = Hydrometeor("snow", D14NDmIce(), [snow_n0_a_priori, snow_dm_a_priori],
                   snow_shape, snow_shape_meta)
snow.transformations = [
    Composition(Log10(), PiecewiseLinear(snow_n0_a_priori)),
    Composition(Identity(), PiecewiseLinear(snow_dm_a_priori))
]
snow.limits_low = [0, 1e-8]
snow.radar_only = True
snow.retrieve_first_moment = False

################################################################################
# Rain particles
################################################################################

# default shape
rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

#
# water content
#

rain_mask = TemperatureMask(273, 340.0)
rain_covariance = Diagonal(4)
rain_md_a_priori = FixedAPriori("rain_dm", -5, rain_covariance)
rain_md_a_priori = ReducedVerticalGrid(rain_md_a_priori, md_z_grid, "altitude")

#
# n0
#

rain_mask = TemperatureMask(273, 340.0)
rain_covariance = Diagonal(1)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=0)
rain_n0_a_priori = ReducedVerticalGrid(rain_n0_a_priori,
                                       z_grid,
                                       "altitude",
                                       provide_retrieval_grid=False)

#
# dm
#

rain_covariance = Diagonal(300e-6**2)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask_value=1e-12)
rain_dm_a_priori = ReducedVerticalGrid(rain_dm_a_priori,
                                       z_grid,
                                       "altitude",
                                       provide_retrieval_grid=False)

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
rain.limits_low = [0, 1e-8]
rain.retrieve_second_moment = True


################################################################################
# Liquid particles
################################################################################

liquid_mask = TemperatureMask(230, 300.0)
liquid_covariance = Diagonal(1**2)
cloud_water_a_priori = FixedAPriori("cloud_water",
                                    -6,
                                    liquid_covariance,
                                    mask=liquid_mask,
                                    mask_value=-20)
cloud_water_a_priori = ReducedVerticalGrid(cloud_water_a_priori,
                                           z_grid,
                                           "altitude",
                                           provide_retrieval_grid=False)

################################################################################
# Humidity
################################################################################

upper_limit = 1.1
lower_limit = 0.0

def a_priori_shape(t):
    transformation = Atanh()
    transformation.z_max = upper_limit
    transformation.z_min = lower_limit
    x = np.maximum(np.minimum(0.7 - (270 - t) / 100.0, 0.7), 0.2)
    return transformation(x)

rh_mask = AltitudeMask(-1, 20e3)
rh_covariance = Diagonal(1.0, mask = rh_mask)
rh_covariance = SpatialCorrelation(rh_covariance, 2e3, mask = rh_mask)
rh_a_priori = FunctionalAPriori("H2O", "temperature", a_priori_shape,
                                rh_covariance,
                                mask = rh_mask,
                                mask_value = -100)
rh_a_priori = ReducedVerticalGrid(rh_a_priori,
                                  z_grid,
                                  quantity = "altitude",
                                  provide_retrieval_grid = False)
rh_a_priori.unit = "rh"
rh_a_priori.transformation = Composition(Atanh(lower_limit, upper_limit),
                                         PiecewiseLinear(rh_a_priori))
