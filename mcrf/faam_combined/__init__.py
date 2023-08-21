"""
mcrf.faam_cs
============

This module defines the a priori assumptions and sensor settings
for the FAMM ClouSat underpasses c159 and c161.
"""
import os
from pathlib import Path

import numpy as np
from mcrf.psds import D14NDmIce, D14NDmSnow, D14NDmLiquid
from mcrf.hydrometeors import Hydrometeor
from artssat.retrieval.a_priori import *
from artssat.jacobian import Atanh, Log10, Identity, Composition
from mcrf.liras.common import (n0_a_priori, dm_a_priori, rh_a_priori,
                               ice_mask, rain_mask)
from mcrf.sensors import Ismar, Marss


path = os.environ.get("JOINT_FLIGHT_PATH")
if path is None:
    path = Path(__file__).parent.parent

scattering_data = os.path.join(path, "data", "scattering_data")

# Vertical grid with reduced resolution
z_grid = np.linspace(0, 10e3, 20)

###############################################################################
# Sensor instances
###############################################################################

ismar = Ismar([0, 1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 15, 16])
marss = Marss()

###############################################################################
# Ice particles
###############################################################################


ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# D_m
#

ice_covariance = Diagonal(500e-6**2, mask=ice_mask, mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 2e3, mask=ice_mask)
ice_dm_a_priori = FunctionalAPriori("ice_dm",
                                    "temperature",
                                    dm_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=1e-8)

#
# N_0^*
#

ice_covariance = Diagonal(0.25, mask=ice_mask, mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0",
                                    "temperature",
                                    n0_a_priori,
                                    ice_covariance,
                                    mask=ice_mask,
                                    mask_value=4)
ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori,
                                    20,
                                    ice_mask,
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
# Lower limits for N_0^* and m in transformed space.
ice.limits_low = [2, 1e-8]

###############################################################################
# Snow particles
###############################################################################

snow_shape = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")
snow_mask = And(AltitudeMask(0.0, 9e3), TemperatureMask(0.0, 276.0))

#
# D_m
#

snow_covariance = Diagonal(500e-6**2, mask=snow_mask, mask_value=1e-12)
snow_covariance = SpatialCorrelation(snow_covariance, 2e3, mask=ice_mask)
snow_dm_a_priori = FixedAPriori("snow_dm",
                                1e-3,
                                snow_covariance,
                                mask=snow_mask,
                                mask_value=1e-8)

#
# N_0^*
#

snow_covariance = Diagonal(0.25, mask=snow_mask, mask_value=1e-12)
snow_covariance = SpatialCorrelation(snow_covariance, 4e3, mask=snow_mask)
snow_n0_a_priori = FixedAPriori("snow_n0",
                                7,
                                snow_covariance,
                                mask=snow_mask,
                                mask_value=4)
snow_n0_a_priori = MaskedRegularGrid(snow_n0_a_priori,
                                     10,
                                     ice_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

#
# Hydrometeor definition
#

snow = Hydrometeor("snow", D14NDmIce(), [snow_n0_a_priori, snow_dm_a_priori],
                   snow_shape, snow_shape_meta)
snow.transformations = [
    Log10(),
    Identity()
]
# Lower limits for N_0^* and m in transformed space.
snow.limits_low = [4, 1e-8]

################################################################################
# Rain particles
################################################################################

rain_shape = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

#
# D_m
#

rain_covariance = Diagonal(500e-6**2, mask=rain_mask, mask_value=1e-12)
rain_dm_a_priori = FixedAPriori("rain_dm",
                                500e-6,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=1e-8)

#
# N_0^*
#

rain_covariance = Diagonal(1, mask=rain_mask, mask_value=1e-12)
rain_n0_a_priori = FixedAPriori("rain_n0",
                                7,
                                rain_covariance,
                                mask=rain_mask,
                                mask_value=2)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori,
                                     4,
                                     rain_mask,
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
    Identity()
]
rain.limits_low = [2, 1e-8]

###############################################################################
# Liquid particles
###############################################################################

liquid_mask = TemperatureMask(240.0, 300.0)
liquid_covariance = Diagonal(1**2, mask=liquid_mask)
cloud_water_a_priori = DataProviderAPriori("cloud_water",
                                           liquid_covariance,
                                           transformation=Log10(),
                                           mask=liquid_mask,
                                           mask_value=-20)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori,
                                         8,
                                         liquid_mask,
                                         "altitude",
                                         provide_retrieval_grid=False)

###############################################################################
# Humidity
###############################################################################

rh_mask = AltitudeMask(-1, 12e3)
rh_covariance = Diagonal(1.0)
rh_covariance = SpatialCorrelation(rh_covariance, 1e3)
h2o_a_priori = DataProviderAPriori("H2O", rh_covariance)
#h2o_a_priori = ReducedVerticalGrid(h2o_a_priori,
#                                   z_grid,
#                                   quantity="altitude",
#                                   provide_retrieval_grid=False)
#h2o_a_priori.transformation = Composition(Atanh(0.0, 1.2),
#                                          PiecewiseLinear(h2o_a_priori))
h2o_a_priori.unit = "rh"
h2o_a_priori.transformation = Atanh(0.0, 1.1)
h2o_a_priori.limit_low = -10
h2o_a_priori.limit_high = 10


################################################################################
# Temperature
################################################################################

temperature_covariance = Diagonal(2**2)
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
                if not isinstance(nedt_dp, (np.ndarray, list)):
                    nedt_dp = [nedt_dp] * s.y_vector_length
                diag += [nedt_dp**2]

        for s in self.sensors:
            nedt_dp = self._get_nedt(s, i_p)
            c = self.noise_scaling[s.name]
            if isinstance(s, PassiveSensor):
                if not isinstance(nedt_dp, (np.ndarray, list)):
                    nedt_dp = [nedt_dp] * s.y_vector_length
                diag += [(c * s.nedt)**2 + (nedt_dp**2)]

        diag = np.concatenate(diag).ravel()
        covmat = sp.sparse.diags(diag, format="coo")

        return covmat
