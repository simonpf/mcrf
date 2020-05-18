import os
from mcrf.psds import D14NDmIce, D14NDmLiquid
from mcrf.hydrometeors import Hydrometeor
from mcrf.liras.common import (n0_a_priori, dm_a_priori, rh_a_priori,
                               ice_mask, rain_mask)
from artssat.retrieval.a_priori import *
from artssat.scattering.psd import Binned
from artssat.jacobian import Atanh, Log10, Identity, Composition

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

# Vertical grid with reduced resolution
z_grid = np.linspace(0, 20e3, 11)

################################################################################
# Ice particles
################################################################################

ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# D_m
#

ice_covariance = Diagonal(200e-6**2, mask=ice_mask, mask_value=1e-24)
ice_covariance = SpatialCorrelation(ice_covariance,
                                    1e3,
                                    mask=ice_mask,
                                    mask_value=1e-24)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-8)

#
# N_0^*
#

ice_covariance = Diagonal(4, mask=ice_mask, mask_value=1e-8)
ice_covariance = SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=4)
ice_n0_a_priori = ReducedVerticalGrid(ice_n0_a_priori,
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
    Identity()
]
ice.limits_low = [4, 1e-10]

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

#
# D_m
#

rain_covariance = Diagonal(300e-6**2, mask=rain_mask, mask_value=1e-16)
rain_covariance = SpatialCorrelation(rain_covariance,
                                     2e3,
                                     mask=rain_mask,
                                     mask_value=1e-16)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-8)

rain_covariance = Diagonal(4, mask=rain_mask, mask_value=1e-12)
rain_covariance = SpatialCorrelation(rain_covariance, 2e3, mask=rain_mask)

#
# N_0^*
#

rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=0)
rain_n0_a_priori = ReducedVerticalGrid(rain_n0_a_priori,
                                       z_grid,
                                       "altitude",
                                       provide_retrieval_grid=False)

rain = Hydrometeor("rain", D14NDmLiquid(),
                   [rain_n0_a_priori, rain_dm_a_priori], rain_shape,
                   rain_shape_meta)
rain.transformations = [
    Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
    Identity(),
]
rain.limits_low = [0, 1e-10]
rain.radar_only = True
