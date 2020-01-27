"""
A-priori assumptions for the passive-only retrieval with only a single
species of frozen hydrometeors.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.

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

settings = {"single_species": True}

################################################################################
# Ice particles
################################################################################


def n0_a_priori(t):
    t = t - 272.15
    return np.log10(np.exp(-0.076586 * t + 17.948))


def dm_a_priori(t):
    n0 = 10**n0_a_priori(t)
    iwc = 1e-6
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm


ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

md_z_grid = np.linspace(0, 20e3, 5)
#md_z_grid = np.array([5e3, 15e3])
ice_mask = And(TropopauseMask(), TemperatureMask(0.0, 273.15))
ice_covariance = Diagonal(1 * np.ones(md_z_grid.size),
                          mask=ice_mask,
                          mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)

# n0
points_n0 = 2
ice_covariance = Diagonal(1, mask=ice_mask, mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=4)
ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori,
                                    points_n0,
                                    ice_mask,
                                    "altitude",
                                    provide_retrieval_grid=False,
                                    transition=1e3)

points_dm = 5
ice_covariance = Diagonal(300e-6**2, mask=ice_mask, mask_value=1e-16)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-6)
ice_dm_a_priori = MaskedRegularGrid(ice_dm_a_priori,
                                    points_dm,
                                    ice_mask,
                                    "altitude",
                                    provide_retrieval_grid=False)

ice = Hydrometeor("ice", D14NDmIce(), [ice_n0_a_priori, ice_dm_a_priori],
                  ice_shape, ice_shape_meta)
ice.transformations = [
    Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
    Composition(Identity(), PiecewiseLinear(ice_dm_a_priori))
]
ice.limits_low = [4, 1e-8]
ice.radar_only = False

################################################################################
# Cloud water
################################################################################

liquid_mask = TemperatureMask(230, 300.0)
liquid_covariance = Diagonal(1**2)
cloud_water_a_priori = FixedAPriori("cloud_water",
                                    -5,
                                    liquid_covariance,
                                    mask=liquid_mask,
                                    mask_value=-18)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori, 5, liquid_mask,
                                         "altitude")

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

# mass density
rain_mask = TemperatureMask(273.15, 340.0)
rain_covariance = Diagonal(4, mask=rain_mask, mask_value=1e-16)

rain_md_a_priori = FixedAPriori("rain_md", -5, rain_covariance)
rain_md_a_priori = ReducedVerticalGrid(rain_md_a_priori, md_z_grid, "altitude")

# n0
rain_covariance = Diagonal(1, mask=rain_mask, mask_value=1e-16)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=2)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori,
                                     2,
                                     rain_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

z_grid = np.linspace(0, 20e3, 6)
rain_covariance = Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-16)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-12)
rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori,
                                     2,
                                     rain_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

rain = Hydrometeor("rain", D14NDmLiquid(),
                   [rain_n0_a_priori, rain_dm_a_priori], rain_shape,
                   rain_shape_meta)
rain.transformations = [
    Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
    Composition(Identity(), PiecewiseLinear(rain_dm_a_priori))
]
rain.limits_low = [2, 1e-8]
rain.retrieve_second_moment = True
