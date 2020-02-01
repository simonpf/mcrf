"""
Contains the hydrometeors for the LIRAS retrieval. The hydrometeors implement
the `mcrf.liras.hydrometeors.Hydrometeor` class and bundle hydrometeor
definition (type) with a priori assumptions. The attributes of this module
define the hydrometoers and a priori assumptions used for the retrievals for
the ESA Wide Swath Cloud Profiling study.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.

    snow: Hydrometeor species representing precipitating, frozen hydrometeors.

    rain: Hydrometeor species representing precipitating, liquid hydrometeors.

    h2o_a_priori: A priori provider for humidity retrieval.

    cloud_water_a_priori: A priori provider for cloud water retrieval.
"""
import os
import numpy as np
from mcrf.psds import D14NDmIce, D14NDmLiquid
from mcrf.hydrometeors import Hydrometeor
from mcrf.liras.common import (n0_a_priori, dm_a_priori, rh_a_priori,
                               ice_mask, rain_mask)
from parts.retrieval.a_priori import *
from parts.scattering.psd import Binned
from parts.jacobian import Atanh, Log10, Identity, Composition

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

################################################################################
# Ice particles
################################################################################

ice_shape = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

#
# D_m
#

ice_covariance = Diagonal(500e-6**2, mask=ice_mask, mask_value=1e-12)
ice_covariance = SpatialCorrelation(ice_covariance, 5e3, mask=ice_mask)
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
ice_covariance = SpatialCorrelation(ice_covariance, 10e3, mask=ice_mask)
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

################################################################################
# Snow particles
################################################################################

snow_shape = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")
snow_mask = And(AltitudeMask(0.0, 19e3), TemperatureMask(0.0, 276.0))

#
# D_m
#

snow_covariance = Diagonal(500e-6**2, mask=snow_mask, mask_value=1e-12)
snow_covariance = SpatialCorrelation(snow_covariance, 5e3, mask=ice_mask)
snow_dm_a_priori = FixedAPriori("snow_dm",
                                1e-3,
                                snow_covariance,
                                mask=snow_mask,
                                mask_value=1e-8)

#
# N_0^*
#

snow_covariance = Diagonal(0.25, mask=snow_mask, mask_value=1e-12)
snow_covariance = SpatialCorrelation(snow_covariance, 10e3, mask=snow_mask)
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
    Composition(Log10(), PiecewiseLinear(snow_n0_a_priori)),
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
rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori,
                                     10,
                                     rain_mask,
                                     "altitude",
                                     provide_retrieval_grid=False)

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
    Composition(Identity(), PiecewiseLinear(rain_dm_a_priori))
]
# Lower limits for N_0^* and m in transformed space.
rain.limits_low = [2, 1e-8]
rain.radar_only = True

################################################################################
# Liquid particles
################################################################################

liquid_mask = TemperatureMask(240.0, 300.0)
liquid_covariance = Diagonal(2**2)
liquid_covariance = SpatialCorrelation(liquid_covariance, 2e3)
cloud_water_a_priori = FixedAPriori("cloud_water",
                                    np.log10(1e-6),
                                    liquid_covariance,
                                    mask=liquid_mask,
                                    mask_value=-20)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori,
                                         11,
                                         liquid_mask,
                                         "altitude",
                                         provide_retrieval_grid=False)

################################################################################
# Humidity
################################################################################

rh_mask = AltitudeMask(-1, 20e3)
rh_covariance = Diagonal(0.5, mask=rh_mask)
rh_covariance = SpatialCorrelation(rh_covariance, 2e3)
h2o_a_priori = FunctionalAPriori("H2O",
                                "temperature",
                                rh_a_priori,
                                rh_covariance,
                                mask=rh_mask,
                                mask_value=-100)
h2o_a_priori = MaskedRegularGrid(h2o_a_priori,
                                21,
                                rh_mask,
                                quantity="altitude",
                                provide_retrieval_grid=False)
h2o_a_priori.unit = "rh"
h2o_a_priori.transformation = Composition(Atanh(0.0, 1.1),
                                         PiecewiseLinear(h2o_a_priori))

################################################################################
# Observation error
################################################################################


class ObservationError(DataProviderBase):
    """
    Observation error covariance matrix provider for the LIRAS study.

    This class works in the same way as the :class:`SensorNoiseAPriori` class
    parts, which means that it takes the :code:`nedt` attributes of the given
    sensors and concatenates these vectors into a combined diagonal matrix.

    In addition to that, however, this class also reads in estimated footprint
    error covariance matrices for LCPR and forward model error covariances
    that are included in the observation error covariance matrix, if the
    corresponding flags are set.

    The class assumes the existence of the following files in the :code:`data`
    subfolder of the LIRAS base directory :code:`LIRAS_PATH`:

        - :code:`covmat.npy`: The estimated covariance matrix of the
          footprint error for the LCPR sensor.

        - :code:`nedt_<sensor.name>_fm.npy`: Vector of estimated
          :math:`NE \Delta T` values for sensors in :code:`sensors`.
    """
    def __init__(self,
                 sensors,
                 footprint_error=False,
                 forward_model_error=False,
                 scene="A"):
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
        self.fpe = footprint_error
        self.fme = forward_model_error

        liras_path = os.environ["LIRAS_PATH"]
        filename = os.path.join(liras_path, "data", "covmat.npy")
        try:
            self.lcpr_covmat = np.load(os.path.join(filename))
        except:
            pass

        self.nedt_fm = {}
        for n in [s.name for s in sensors]:
            filename = "e_" + scene.lower() + "_" + n + ".npy"
            filename = os.path.join(liras_path, "data", filename)
            self.nedt_fm[n] = np.load(filename)

        self.noise_scaling = dict([(s.name, 1.0) for s in sensors])

    def get_observation_error_covariance(self, i_p):
        m = 0

        diag = []

        for s in self.sensors:
            c = self.noise_scaling[s.name]
            if isinstance(s, ActiveSensor):
                if s.name == "lcpr":
                    i_lcpr = sum([v.size for v in diag])
                    j_lcpr = i_lcpr + s.nedt.size
                diag += [s.nedt**2]

                if self.fme and not self.fpe:
                    diag[-1] += self.nedt_fm[s.name]**2
                    diag[-1] *= c

        for s in self.sensors:
            c = self.noise_scaling[s.name]
            if isinstance(s, PassiveSensor):
                diag += [s.nedt**2]

                if self.fme:
                    diag[-1] += self.nedt_fm[s.name]**2
                    diag[-1] *= c

        diag = np.concatenate(diag).ravel()

        covmat = sp.sparse.diags(diag, format="coo")

        if self.fpe:
            covmat = covmat.todense()
            covmat[i_lcpr:j_lcpr, i_lcpr:j_lcpr] += self.lcpr_covmat

        return covmat
