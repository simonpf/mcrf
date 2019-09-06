import os
from mcrf.psds import D14NDmIce, D14NDmLiquid
from mcrf.hydrometeors import Hydrometeor
from parts.retrieval.a_priori import *
from parts.scattering.psd import Binned
from parts.jacobian import Atanh, Log10, Identity, Composition

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

################################################################################
# Ice particles
################################################################################


def n0_a_priori(t):
    t = t - 272.15
    return np.log10(np.exp(-0.076586 * t + 17.948))


def dm_a_priori(t):
    n0 = 10**n0_a_priori(t)
    iwc = 5e-7
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm


ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")
ice_mask = And(TropopauseMask(), TemperatureMask(0.0, 278.0))

ice_covariance = Diagonal(300e-6**2, mask=ice_mask, mask_value=1e-24)
ice_covariance = SpatialCorrelation(ice_covariance,
                                    2e3,
                                    mask=ice_mask,
                                    mask_value=1e-24)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-8)

ice_covariance = Diagonal(4, mask=ice_mask, mask_value=1e-8)
ice_covariance = SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=0)
ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori,
                                    10,
                                    ice_mask,
                                    "altitude",
                                    provide_retrieval_grid=False)

ice = Hydrometeor("ice", D14NDmIce(), [ice_n0_a_priori, ice_dm_a_priori],
                  ice_shape, ice_shape_meta)
ice.transformations = [
    Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
    Identity()
]
ice.limits_low = [0, 1e-10]

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

rain_mask = TemperatureMask(272.15, 340.0)
rain_covariance = Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-24)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-8)
rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori,
                                     10,
                                     rain_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

z_grid = np.linspace(0, 12e3, 7)
rain_covariance = Diagonal(1, mask=rain_mask, mask_value=1e-12)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=0)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori,
                                     4,
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
rain.limits_low = [0, 1e-10]
rain.radar_only = True
