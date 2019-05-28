import os
from crac.psds                import D14NDmIce, D14NDmLiquid, D14NDmSnow
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
    t = t - 272.15
    return np.ones(t.shape) * 10.0
    #return np.log10(np.exp(-0.076586 * t + 17.948))

def dm_a_priori(t):
    n0 = 10 ** n0_a_priori(t)
    iwc = 1e-4
    dm = (4.0 ** 4 * iwc / (np.pi * 917.0)  / n0) ** 0.25
    return dm

ice_shape      = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

md_z_grid = np.linspace(0, 20e3, 5)
#md_z_grid = np.array([5e3, 15e3])
ice_mask       = And(TropopauseMask(), TemperatureMask(0.0, 280.0))
snow_mask      = And(AltitudeMask(0.0, 18e3), TemperatureMask(0.0, 280.0))
ice_covariance = Diagonal(1 * np.ones(md_z_grid.size))

# mass density
ice_md_a_priori = FixedAPriori("ice_md", -5, ice_covariance,
                               mask = ice_mask, mask_value = -12)
ice_md_a_priori = ReducedVerticalGrid(ice_md_a_priori, md_z_grid, "altitude",
                                      ice_covariance)

# n0
points_n0 = 2
ice_covariance = Diagonal(1, mask = ice_mask, mask_value = 1e-12)
#ice_n0_a_priori = FunctionalAPriori("ice_n0", "temperature", n0_a_priori, ice_covariance, mask = ice_mask, mask_value = 2)
ice_n0_a_priori = FixedAPriori("ice_n0", 10, ice_covariance, mask = ice_mask, mask_value = 0)
ice_n0_a_priori = MaskedRegularGrid(ice_n0_a_priori, 2, ice_mask, "altitude", provide_retrieval_grid = False)

points_dm = 8
ice_covariance  = Diagonal(200e-6 ** 2, mask = ice_mask, mask_value = 1e-16)
ice_covariance  = SpatialCorrelation(ice_covariance, 4e3)
ice_dm_a_priori = FunctionalAPriori("ice_dm", "temperature", dm_a_priori, ice_covariance, mask = ice_mask, mask_value = 1e-6)
ice_dm_a_priori = MaskedRegularGrid(ice_dm_a_priori, points_dm, ice_mask, "altitude", provide_retrieval_grid = False)

ice = Hydrometeor("ice",
                  D14NDmIce(),
                  [ice_n0_a_priori, ice_dm_a_priori],
                  ice_shape,
                  ice_shape_meta)
ice.transformations = [Composition(Log10(), PiecewiseLinear(ice_n0_a_priori)),
                       Composition(Identity(), PiecewiseLinear(ice_dm_a_priori))]
ice.limits_low = [0, 1e-8]
ice.radar_only = False

################################################################################
# Snow particles
################################################################################

snow_shape      = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")

snow_mask        = And(TropopauseMask(), TemperatureMask(0.0, 280.0))
snow_mask        = And(AltitudeMask(0.0, 18e3), TemperatureMask(0.0, 280.0))
snow_covariance  = Diagonal(4 * np.ones(md_z_grid.size))

# mass density
snow_md_a_priori = FixedAPriori("snow_md", -5, snow_covariance,
                               mask = snow_mask, mask_value = -12)

# n0
snow_covariance  = Diagonal(1.0, mask = ice_mask, mask_value = 1e-12)
snow_n0_a_priori = FixedAPriori("snow_n0", 7, snow_covariance, mask = ice_mask, mask_value = 0)
snow_n0_a_priori = MaskedRegularGrid(snow_n0_a_priori, 2, ice_mask, "altitude", provide_retrieval_grid = False)

snow_covariance  = Diagonal(500e-6 ** 2, mask = ice_mask, mask_value = 1e-16)
snow_dm_a_priori = FixedAPriori("snow_dm", 1000e-6, snow_covariance, mask = ice_mask, mask_value = 1e-5)
snow_dm_a_priori = MaskedRegularGrid(snow_dm_a_priori, points_dm, ice_mask, "altitude",
                                     provide_retrieval_grid = False)

snow = Hydrometeor("snow",
                   D14NDmIce(),
                   [snow_n0_a_priori, snow_dm_a_priori],
                   snow_shape,
                   snow_shape_meta)
snow.transformations = [Composition(Log10(), PiecewiseLinear(snow_n0_a_priori)),
                        Composition(Identity(), PiecewiseLinear(snow_dm_a_priori))]
snow.limits_low = [0, 1e-8]
snow.radar_only = True
snow.retrieve_first_moment = False


################################################################################
# Liquid particles
################################################################################

liquid_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
liquid_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

points_liquid = 2
liquid_mask       = TemperatureMask(230, 273.0)
liquid_covariance = Diagonal(2)
liquid_md_a_priori = FixedAPriori("liquid_md", -5, liquid_covariance,
				mask = liquid_mask, mask_value = -12)
liquid_md_a_priori = MaskedRegularGrid(liquid_md_a_priori, points_liquid, liquid_mask, "altitude")

z_grid = np.linspace(0, 20e3, 4)
liquid_dm_a_priori = FixedAPriori("liquid_dm", 10, liquid_covariance)
liquid_dm_a_priori = ReducedVerticalGrid(liquid_dm_a_priori, z_grid, "altitude",
                                         Diagonal(2 * np.ones(z_grid.size)))

liquid = Hydrometeor("liquid",
                     D14NDmLiquid(),
                     [liquid_md_a_priori, liquid_dm_a_priori],
                     liquid_shape,
                     liquid_shape_meta)
liquid.retrieve_second_moment = True

liquid.transformations = [Identity(), Identity()]
liquid.limits_low = [1e-12, 1e-12]
cloud_water_a_priori = FixedAPriori("cloud_water", -5, liquid_covariance, mask = liquid_mask,
                                    mask_value = -18)
cloud_water_a_priori = MaskedRegularGrid(cloud_water_a_priori, points_liquid,
                                         liquid_mask, "altitude")

################################################################################
# Rain particles
################################################################################

rain_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

# mass density
rain_mask       = TemperatureMask(273, 340.0)
rain_covariance = Diagonal(4)

rain_md_a_priori = FixedAPriori("rain_md", -5, rain_covariance)
rain_md_a_priori = ReducedVerticalGrid(rain_md_a_priori, md_z_grid, "altitude")

# n0
rain_mask       = TemperatureMask(273, 340.0)
rain_covariance = Diagonal(1)
rain_n0_a_priori = FixedAPriori("rain_n0", 7, rain_covariance, mask = rain_mask, mask_value = 0)
rain_n0_a_priori = MaskedRegularGrid(rain_n0_a_priori, 2, rain_mask, "altitude", provide_retrieval_grid = False)

z_grid = np.linspace(0, 20e3, 6)
rain_covariance = Diagonal(300e-6 ** 2)
rain_dm_a_priori = FixedAPriori("rain_dm", 500e-6, rain_covariance, mask_value = 1e-12)
rain_dm_a_priori = MaskedRegularGrid(rain_dm_a_priori, 2, rain_mask, "altitude", provide_retrieval_grid = False)

rain = Hydrometeor("rain",
                   D14NDmLiquid(),
                   [rain_n0_a_priori, rain_dm_a_priori],
                   rain_shape,
                   rain_shape_meta)
rain.transformations = [Composition(Log10(), PiecewiseLinear(rain_n0_a_priori)),
                        Composition(Identity(), PiecewiseLinear(rain_dm_a_priori))]
rain.limits_low = [0, 1e-8]
rain.retrieve_second_moment = True

################################################################################
# Humidity
################################################################################

def a_priori_shape(t):
    transformation = Atanh()
    transformation.z_max = 1.2
    transformation.z_min = 0.0
    x = np.maximum(np.minimum(0.7 - (270 - t) / 100.0, 0.7), 0.1)
    return transformation(x)


z_grid = np.linspace(0, 20e3, 11)
rh_covariance = Diagonal(1.0)
rh_a_priori = FunctionalAPriori("H2O", "temperature", a_priori_shape,
                                rh_covariance)
rh_a_priori = ReducedVerticalGrid(rh_a_priori, z_grid, "altitude",
                                  provide_retrieval_grid = False)

#rh_a_priori.transformation = Composition(transformation, PiecewiseLinear(rh_a_priori))

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
        self.lcpr_covmat =  np.load(os.path.join(filename))

        self.nedt_fm = {}
        for n in [s.name for s in sensors]:
            filename = os.path.join(liras_path, "data", "nedt_" + n + "_fm.npy")
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
                diag += [(c * s.nedt) ** 2]

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
