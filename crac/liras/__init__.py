"""
Contains the hydrometeors for the LIRAS retrieval. The hydrometeors implement
the `crac.liras.hydrometeors.Hydrometeor` and bundle hydrometeor definition
(type) with a priori assumption. The attributes of this module define the
hydrometoers and a priori assumptions used for the retrievals for the
ESA Wide Swath Cloud Profiling study.

Attributes:

    ice:  Hydrometeor species representing frozen hydrometeors.

    snow: Hydrometeor species representing precipitating, frozen hydrometeors.

    rain: Hydrometeor species representing precipitating, liquid hydrometeors.

    rh_a_priori: A priori provider for humidity retrieval.

    cloud_water_a_priori: A priori provider for cloud water retrieval.
"""
import os
import numpy as np
from crac.psds                import D14NDmIce, D14NDmLiquid
from crac.hydrometeors        import Hydrometeor
from parts.retrieval.a_priori import *
from parts.scattering.psd     import Binned
from parts.jacobian           import Atanh, Log10, Identity, Composition

liras_path = os.environ["LIRAS_PATH"]
scattering_data = os.path.join(liras_path, "data", "scattering")

################################################################################
# Ice particles
################################################################################

def n0_a_priori(t):
    """
    Defines the a priori mean for the normalized number density (N_0^*) for
    frozen hydrometeors as a function of the temperature t.
    """
    return 10.0 * np.ones(t.shape)

def dm_a_priori(t):
    """
    Defines the a priori mean for the mass-weighted mean diameter (D_m) for
    frozen hydrometeors as a function of the temperature t.
    """
    n0 = 10 ** n0_a_priori(t)
    iwc = 1e-5
    dm = (4.0 ** 4 * iwc / (np.pi * 917.0)  / n0) ** 0.25
    return dm

ice_shape      = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")
ice_mask       = And(AltitudeMask(0.0, 19e3), TemperatureMask(0.0, 276.0))

ice_covariance  = Diagonal(100e-6 ** 2, mask = ice_mask, mask_value = 1e-12)
ice_covariance  = SpatialCorrelation(ice_covariance, 5e3, mask = ice_mask)
ice_dm_a_priori = FunctionalAPriori("ice_dm", "temperature", dm_a_priori, ice_covariance,
                                    mask = ice_mask, mask_value = 1e-8)

ice_covariance  = Diagonal(0.25, mask = ice_mask, mask_value = 1e-12)
ice_covariance  = SpatialCorrelation(ice_covariance, 10e3, mask = ice_mask)
ice_n0_a_priori = FunctionalAPriori("ice_n0", "temperature", n0_a_priori, ice_covariance,
                                    mask = ice_mask, mask_value = 4)
#ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori, 10, ice_mask, "altitude", provide_retrieval_grid = False)

ice = Hydrometeor("ice", D14NDmIce(), [ice_n0_a_priori, ice_dm_a_priori], ice_shape, ice_shape_meta)
ice.transformations = [Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
                       Identity()]
ice.limits_low = [2, 1e-8]

################################################################################
# Snow particles
################################################################################

snow_shape      = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")
#snow_mask       = And(TropopauseMask(), TemperatureMask(0.0, 278.0))
snow_mask        = And(AltitudeMask(0.0, 19e3), TemperatureMask(0.0, 276.0))

snow_covariance = Diagonal(500e-6 ** 2, mask = snow_mask, mask_value = 1e-12)
snow_covariance  = SpatialCorrelation(snow_covariance, 5e3, mask = ice_mask)
snow_dm_a_priori = FixedAPriori("snow_dm", 1e-3, snow_covariance,
                                mask = snow_mask, mask_value = 1e-8)

snow_covariance  = Diagonal(0.25, mask = snow_mask, mask_value = 1e-12)
snow_covariance  = SpatialCorrelation(snow_covariance, 10e3, mask = snow_mask)
snow_n0_a_priori = FixedAPriori("snow_n0", 7, snow_covariance, mask = snow_mask, mask_value = 4)
#snow_n0_a_priori = MaskedRegularGrid(snow_n0_a_priori, 10, ice_mask, "altitude", provide_retrieval_grid = False)

snow = Hydrometeor("snow", D14NDmIce(), [snow_n0_a_priori, snow_dm_a_priori], snow_shape, snow_shape_meta)
snow.transformations = [Composition(Log10(), PiecewiseLinear(snow_n0_a_priori)),
                       Identity()]
snow.limits_low = [4, 1e-8]

################################################################################
# Liquid particles
################################################################################

liquid_mask = TemperatureMask(230, 300.0)
liquid_covariance = Diagonal(1 ** 2)
cloud_water_a_priori = FixedAPriori("cloud_water", -6, liquid_covariance,
                                    mask = liquid_mask, mask_value = -20)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori, 7, liquid_mask,
                                         "altitude", provide_retrieval_grid = False)

################################################################################
# Rain particles
################################################################################

rain_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

rain_mask  = TemperatureMask(273, 340.0)
rain_covariance = Diagonal(500e-6 ** 2, mask = rain_mask, mask_value = 1e-12)
rain_dm_a_priori = FixedAPriori("rain_dm", 500e-6, rain_covariance, mask = rain_mask, mask_value = 1e-8)
rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori, 10, rain_mask, "altitude", provide_retrieval_grid = False)

z_grid = np.linspace(0, 12e3, 7)
rain_covariance = Diagonal(1, mask = rain_mask, mask_value = 1e-12)
rain_n0_a_priori = FixedAPriori("rain_n0", 7, rain_covariance, mask = rain_mask, mask_value = 2)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori, 4, rain_mask, "altitude", provide_retrieval_grid = False)

rain = Hydrometeor("rain", D14NDmLiquid(), [rain_n0_a_priori, rain_dm_a_priori], rain_shape, rain_shape_meta)
rain.transformations = [Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
                        Composition(Identity(), PiecewiseLinear(rain_dm_a_priori))]
rain.limits_low = [2, 1e-8]
rain.radar_only = True

################################################################################
# Humidity
################################################################################

def a_priori_shape(t):
    transformation = Atanh()
    transformation.z_max = 1.1
    transformation.z_min = 0.0
    x = np.maximum(np.minimum(0.7 - (270 - t) / 100.0, 0.7), 0.2)
    return transformation(x)


z_grid = np.linspace(0, 20e3, 11)
rh_covariance = Diagonal(2.0)
rh_covariance = SpatialCorrelation(rh_covariance, 2e3)
rh_a_priori = FunctionalAPriori("H2O", "temperature", a_priori_shape, rh_covariance)
rh_a_priori = ReducedVerticalGrid(rh_a_priori, z_grid, "altitude",
                                  provide_retrieval_grid = False)

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
    def __init__(self, sensors, footprint_error = False, forward_model_error = False):
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
        self.fpe     = footprint_error
        self.fme     = forward_model_error

        liras_path = os.environ["LIRAS_PATH"]
        filename = os.path.join(liras_path, "data", "covmat.npy")
        try:
            self.lcpr_covmat =  np.load(os.path.join(filename))
        except:
            pass

        self.nedt_fm = {}
        for n in [s.name for s in sensors]:
            filename = os.path.join(liras_path, "data", "nedt_" + n + "_fm.npy")
            try:
                self.nedt_fm[n] = np.load(filename)
            except:
                pass

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
                diag += [(c * s.nedt) ** 2 + 0.5 ** 2]

                if self.fme and not self.fpe:
                    diag[-1] += self.nedt_fm[s.name] ** 2

        for s in self.sensors:
            c = self.noise_scaling[s.name]
            if isinstance(s, PassiveSensor):
                diag += [(c * s.nedt) ** 2]

                if self.fme:
                    diag[-1] += self.nedt_fm[s.name] ** 2

        diag = np.concatenate(diag).ravel()

        covmat = sp.sparse.diags(diag, format = "coo")

        if self.fpe:
            covmat = covmat.todense()
            covmat[i_lcpr : j_lcpr, i_lcpr : j_lcpr] += self.lcpr_covmat

        return covmat
