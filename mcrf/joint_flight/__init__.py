"""
mcrf.joint_flight
==================

The joint flight module contains the a priori settings for the joint flight
retrieval.
"""
import os
from mcrf.psds import D14NDmIce, D14NDmSnow, D14NDmLiquid
from mcrf.hydrometeors import Hydrometeor
from parts.retrieval.a_priori import *
from parts.jacobian import Atanh, Log10, Identity, Composition

path = os.environ["JOINT_FLIGHT_PATH"]
scattering_data = os.path.join(path, "data", "scattering_data")

################################################################################
# Ice particles
################################################################################


def n0_a_priori(t):
    r"""
    The assumed a priori for :math:`N_0^*` as a function of t.

    Args:
        t(array): The temperature profile.

    Return:
        Array containing the priori for :math:`N_0^*`.
    """
    t = t - 272.15
    return np.log10(np.exp(-0.076586 * t + 17.948))


def dm_a_priori(t):
    r"""
    Prior for :math:`D_m`

    The priori for :math:`D_m` is computed by assuming a fixed
    mass of :math:`10^{-6}\ \unit{kg\ m^{-3}}` and applying
    the :math:`N_0^*` assumption.

    Args:
        t(array): The temperature profile.

    Return:
        Array containing the priori for :math:`N_0^*`.
    """
    n0 = 10**n0_a_priori(t)
    iwc = 5e-6
    dm = (4.0**4 * iwc / (np.pi * 917.0) / n0)**0.25
    return dm


ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")
ice_mask = And(AltitudeMask(0.0, 12e3), TemperatureMask(0.0, 273.0))

#
# D_m
#

ice_covariance = Diagonal(400e-6**2, mask=ice_mask, mask_value=1e-24)
ice_covariance = SpatialCorrelation(ice_covariance,
                                    1.0e3,
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

ice_covariance = Diagonal(1, mask=ice_mask, mask_value=1e-8)
ice_covariance = SpatialCorrelation(ice_covariance, 1.0e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=4)
z_grid = np.linspace(0, 12e3, 25)
ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori,
                                    11,
                                    ice_mask,
                                    "altitude",
                                    provide_retrieval_grid=False)
ice = Hydrometeor("ice", D14NDmIce(), [ice_n0_a_priori, ice_dm_a_priori],
                  ice_shape, ice_shape_meta)
ice.radar_only = False
ice.retrieve_second_moment = True

ice.transformations = [
    Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
    Identity()
]
ice.limits_low = [0, 1e-10]

################################################################################
# Snow particles
################################################################################

snow_shape = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")

snow_mask = And(AltitudeMask(0.0, 10e3), TemperatureMask(0.0, 280.0))
snow_covariance = Thikhonov(scaling=1.0, mask=snow_mask)
snow_md_a_priori = FixedAPriori("snow_md",
                                np.log10(5 * 1e-6),
                                snow_covariance,
                                mask=snow_mask,
                                mask_value=-12)
snow_dm_a_priori = FixedAPriori("snow_dm",
                                -3,
                                snow_covariance,
                                mask=snow_mask,
                                mask_value=-12)

z_grid = np.linspace(0, 12e3, 13)
snow_n0_a_priori = FixedAPriori("snow_n0", 6, snow_covariance)
snow_n0_a_priori = ReducedVerticalGrid(snow_n0_a_priori, z_grid, "altitude",
                                       Diagonal(4 * np.ones(21)))

snow = Hydrometeor("snow", D14NDmSnow(), [snow_md_a_priori, snow_n0_a_priori],
                   snow_shape, snow_shape_meta)
snow.radar_only = True
snow.retrieve_second_moment = True

################################################################################
# Rain particles
################################################################################

liquid_mask = TemperatureMask(230.0, 300.0)
liquid_covariance = Diagonal(1**2)
liquid_covariance = SpatialCorrelation(liquid_covariance, 2e3)
cloud_water_a_priori = FixedAPriori("cloud_water",
                                    -6,
                                    liquid_covariance,
                                    mask=liquid_mask,
                                    mask_value=-20)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori,
                                         11,
                                         liquid_mask,
                                         "altitude",
                                         provide_retrieval_grid=False)

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

z_grid = np.linspace(0, 12e3, 13)
rain_mask = TemperatureMask(270, 340.0)

rain_covariance = Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-16)
rain_covariance = SpatialCorrelation(rain_covariance,
                                     0.5e3,
                                     mask=rain_mask,
                                     mask_value=1e-16)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-8)

rain_covariance = Diagonal(1, mask=rain_mask, mask_value=1e-12)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=0)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori,
                                     5,
                                     rain_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

rain = Hydrometeor("rain", D14NDmLiquid(),
                   [rain_n0_a_priori, rain_dm_a_priori], rain_shape,
                   rain_shape_meta)
rain.retrieve_second_moment = True

rain.transformations = [
    Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
    Identity()
]
rain.limits_low = [0, 1e-10]

################################################################################
# Humidity
################################################################################


def a_priori_shape(t):
    transformation = Atanh()
    transformation.z_max = 1.1
    transformation.z_min = 0.0
    x = np.maximum(np.minimum(0.5 - (270 - t) / 100.0, 0.5), 0.1)
    return transformation(x)


#rh_covariance = Thikhonov(scaling = 0.2)
rh_covariance = Diagonal(5)


class RelativeHumidityAPriori(APrioriProviderBase):
    def __init__(self, covariance, mask=None, mask_value=1e-12):
        super().__init__("H2O", covariance)
        self.mask = mask
        self.mask_value = mask_value
        self.transformation = Atanh()
        self.transformation.z_max = 1.1
        self.transformation.z_min = 0.0

    def get_xa(self, *args, **kwargs):

        xa = np.minimum(
            self.owner.get_relative_humidity(*args, *kwargs) / 100.0, 0.95)
        xa = np.maximum(xa, 0.1)

        if not self.mask is None:
            mask = np.logical_not(self.mask(self.owner, *args, **kwargs))
            xa[mask] = self.mask_value

        xa = self.transformation(xa)
        return xa


#rh_a_priori = RelativeHumidityAPriori(rh_covariance)
rh_a_priori = FunctionalAPriori("H2O", "temperature", a_priori_shape,
                                rh_covariance)

################################################################################
# Temperature
################################################################################

temperature_covariance = Diagonal(1**2)
temperature_covariance = SpatialCorrelation(temperature_covariance, 0.5e3)
temperature_a_priori = DataProviderAPriori("temperature",
                                           temperature_covariance)


class ObservationError(DataProviderBase):
    """
    """
    def __init__(self,
                 sensors,
                 footprint_error=False,
                 forward_model_error=False):
        """
        Arguments:
            sensors(:code:`list`): List of :code:`parts.sensor.Sensor` objects
                containing the sensors that are used in the retrieval.

            footprint_error(:code:`Bool`): Include footprint error for :code:`lcpr`
                sensor.

            forward_model_error(:code:`Bool`): Include estimated model error for
                all sensors.

        """
        self.sensors = sensors
        self.noise_scaling = dict([(s.name, 1.0) for s in sensors])

    def _get_nedt(self, sensor, i_p):
        try:
            f_name = "get_y_" + sensor.name + "_nedt"
            f = getattr(self.owner, f_name)
            nedt_dp = f(i_p)
        except:
            nedt_dp = 0.0
        return nedt_dp

    def get_observation_error_covariance(self, i_p):
        m = 0

        diag = []

        for s in self.sensors:

            nedt_dp = self._get_nedt(s, i_p)
            c = self.noise_scaling[s.name]
            if isinstance(s, ActiveSensor):
                diag += [(c * s.nedt)**2 + (nedt_dp**2)]

        for s in self.sensors:
            nedt_dp = self._get_nedt(s, i_p)
            c = self.noise_scaling[s.name]
            if isinstance(s, PassiveSensor):
                diag += [(c * s.nedt)**2 + (nedt_dp**2)]

        diag = np.concatenate(diag).ravel()
        covmat = sp.sparse.diags(diag, format="coo")

        return covmat
