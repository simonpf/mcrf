import os
from crac.psds                import D14Ice, D14Liquid
from crac.hydrometeors        import Hydrometeor
from parts.retrieval.a_priori import *
from parts.jacobian           import Atanh

path = os.environ["JOINT_FLIGHT_PATH"]
scattering_data = os.path.join(path, "data", "scattering_data")

################################################################################
# Ice particles
################################################################################

ice_shape      = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

ice_mask       = And(TropopauseMask(), TemperatureMask(0.0, 273.0))
ice_covariance = Thikhonov(scaling = 1.0, mask = ice_mask)
ice_md_a_priori = FixedAPriori("ice_md", -6, ice_covariance,
                               mask = ice_mask, mask_value = -12)

z_grid = np.linspace(0, 12e3, 13)
ice_n0_a_priori = FixedAPriori("ice_n0", 10, ice_covariance,
                               mask = ice_mask, mask_value = 15)
ice_n0_a_priori = ReducedVerticalGrid(ice_n0_a_priori, z_grid, "altitude",
                                      Diagonal(4 * np.ones(13)))

ice = Hydrometeor("ice",
                  D14Ice(),
                  [ice_md_a_priori, ice_n0_a_priori],
                  ice_shape,
                  ice_shape_meta)
ice.radar_only = False
################################################################################
# Snow particles
################################################################################

snow_shape      = os.path.join(scattering_data, "EvansSnowAggregate.xml")
snow_shape_meta = os.path.join(scattering_data, "EvansSnowAggregate.meta.xml")

snow_mask       = And(TropopauseMask(), TemperatureMask(0.0, 275.0))
snow_covariance = Thikhonov(scaling = 1.0, mask = snow_mask)
snow_md_a_priori = FixedAPriori("snow_md", -6, snow_covariance,
                               mask = snow_mask, mask_value = -12)

z_grid = np.linspace(0, 12e3, 13)
snow_n0_a_priori = FixedAPriori("snow_n0", 6, snow_covariance,
                               mask = snow_mask, mask_value = 15)
snow_n0_a_priori = ReducedVerticalGrid(snow_n0_a_priori, z_grid, "altitude",
                                      Diagonal(4 * np.ones(13)))

snow = Hydrometeor("snow",
                   D14Ice(),
                   [snow_md_a_priori, snow_n0_a_priori],
                   snow_shape,
                   snow_shape_meta)
snow.radar_only = True
snow.retrieve_second_moment = False


################################################################################
# Liquid particles
################################################################################

liquid_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
liquid_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

liquid_mask  = TemperatureMask(240, 340.0)
liquid_covariance = Thikhonov(scaling = 1.0, mask = liquid_mask)
liquid_md_a_priori = FixedAPriori("liquid_md", -6, liquid_covariance,
                                  mask = liquid_mask, mask_value = -12)

z_grid = np.linspace(0, 12e3, 7)
liquid_n0_a_priori = FixedAPriori("liquid_n0", 12, liquid_covariance,
                                  mask = liquid_mask, mask_value = 15)
liquid_n0_a_priori = ReducedVerticalGrid(liquid_n0_a_priori, z_grid, "altitude",
                                         Diagonal(4 * np.ones(7)))

liquid = Hydrometeor("liquid",
                     D14Liquid(),
                     [liquid_md_a_priori, liquid_n0_a_priori],
                     liquid_shape,
                     liquid_shape_meta)
liquid.retrieve_second_moment = False

################################################################################
# Rain particles
################################################################################

rain_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
rain_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

rain_mask  = TemperatureMask(273, 340.0)
rain_covariance = Thikhonov(scaling = 1.0, mask = rain_mask)
rain_md_a_priori = FixedAPriori("rain_md", -6, rain_covariance,
                                  mask = rain_mask, mask_value = -12)

z_grid = np.linspace(0, 12e3, 7)
rain_n0_a_priori = FixedAPriori("rain_n0", 5, rain_covariance,
                                  mask = rain_mask, mask_value = 15)
rain_n0_a_priori = ReducedVerticalGrid(rain_n0_a_priori, z_grid, "altitude",
                                         Diagonal(4 * np.ones(7)))

rain = Hydrometeor("rain",
                     D14Liquid(),
                     [rain_md_a_priori, rain_n0_a_priori],
                     rain_shape,
                     rain_shape_meta)
rain.retrieve_second_moment = False

################################################################################
# Humidity
################################################################################

def a_priori_shape(t):
    transformation = Atanh()
    x = np.maximum(np.minimum(0.5 - (270 - t) / 100.0, 0.5), 0.001)
    return transformation(x)


rh_mask = TropopauseMask()
rh_covariance = Thikhonov(scaling = 1.0, mask = rh_mask)
rh_a_priori = FunctionalAPriori("H2O", "temperature", a_priori_shape,
                                rh_covariance)
