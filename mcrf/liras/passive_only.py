"""
A-priori assumptions for the passive-only retrieval.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.
    snow: Hydrometeor species representing precipitating, frozen hydrometeors.
    rain: Hydrometeor species representing precipitating, liquid hydrometeors.
    h2o_a_priori: A priori provider for humidity retrieval.
    cloud_water_a_priori: A priori provider for cloud water retrieval.
"""
import os
from mcrf.psds import D14NDmIce, D14NDmLiquid, D14NDmSnow
from mcrf.hydrometeors import Hydrometeor
from mcrf.liras.common import (n0_a_priori, dm_a_priori, rh_a_priori,
                               ice_mask, rain_mask)
from artssat.retrieval.a_priori import *
from artssat.scattering.psd import Binned
from artssat.jacobian import Log10, Identity, Composition, Atanh

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

# Reduced altitude grid with resolution of 2 km used in the passive
# only retrieval.
z_grid = np.linspace(0, 20e3, 11)
z_grid_2 = np.linspace(0, 20e3, 6)

################################################################################
# Ice particles
################################################################################

# The default ice shape
ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# Water content
#

md_z_grid = np.linspace(0, 20e3, 5)
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
# Hydrometeor definition
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

liquid_mask = TemperatureMask(240, 300.0)
liquid_covariance = Diagonal(1, mask = liquid_mask)
liquid_covariance = SpatialCorrelation(liquid_covariance, 2e3, mask = liquid_mask)
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

rh_mask = AltitudeMask(-1, 19e3)
rh_covariance = Diagonal(1.0, mask = rh_mask)
rh_covariance = SpatialCorrelation(rh_covariance, 2e3, mask = rh_mask)
h2o_a_priori = FunctionalAPriori("H2O",
                                 "temperature",
                                 rh_a_priori,
                                 rh_covariance,
                                 mask = rh_mask,
                                 mask_value = -100)
h2o_a_priori = ReducedVerticalGrid(h2o_a_priori,
                                   z_grid,
                                   quantity = "altitude",
                                   provide_retrieval_grid = False)
h2o_a_priori.unit = "rh"
h2o_a_priori.transformation = Composition(Atanh(0.0, 1.1),
                                         PiecewiseLinear(h2o_a_priori))
