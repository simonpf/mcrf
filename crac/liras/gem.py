import os
import glob
import numpy as np

from parts.scattering import ScatteringSpecies
from parts.scattering.psd import MY05
from parts.scattering.psd.binned import Binned


liras_path = os.environ["LIRAS_PATH"]
gem_path = os.path.join(liras_path, "data", "scattering")

# Ice

scat_data      = glob.glob(os.path.join(gem_path, "*Ice*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Ice*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "cloud_ice")
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd_binned.t_min = 0.0
psd_binned.t_max = 280.0
gem_ice        = ScatteringSpecies("gem_ice", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_ice_binned = ScatteringSpecies("ice", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)

scat_data      = glob.glob(os.path.join(gem_path, "*Snow*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Snow*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "snow")
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned.t_min = 0.0
psd_binned.t_max = 280.0
gem_snow       = ScatteringSpecies("gem_snow", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_snow_binned    = ScatteringSpecies("snow", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)

scat_data      = glob.glob(os.path.join(gem_path, "*Hail*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Hail*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "hail")
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned.t_min = 0.0
psd_binned.t_max = 280.0
gem_hail       = ScatteringSpecies("gem_hail", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_hail_binned    = ScatteringSpecies("hail", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)

scat_data      = glob.glob(os.path.join(gem_path, "*Graupel*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Graupel*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "graupel")
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned.t_min = 0.0
psd_binned.t_max = 280.0
gem_graupel    = ScatteringSpecies("gem_graupel", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_graupel_binned = ScatteringSpecies("graupel", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)

# Liquid

scat_data      = glob.glob(os.path.join(gem_path, "*Liquid*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Liquid*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "cloud_water")
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned.t_min = 260.0
psd_binned.t_max = 999.0
gem_liquid     = ScatteringSpecies("gem_cloud_water", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_liquid_binned  = ScatteringSpecies("cloud_water", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)


scat_data      = glob.glob(os.path.join(gem_path, "*Liquid*data_new.xml"))[0]
scat_meta_data = glob.glob(os.path.join(gem_path, "*Liquid*meta_new.xml"))[0]
psd            = MY05(hydrometeor_type = "rain")
psd_binned     = Binned(x = np.logspace(-6, -1, 21), size_parameter = psd.size_parameter)
psd.t_min      = 0.0
psd.t_max      = 999.0
psd_binned.t_min = 260.0
psd_binned.t_max = 999.0
gem_rain       = ScatteringSpecies("gem_rain", psd,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)
gem_rain_binned    = ScatteringSpecies("rain", psd_binned,
                                   scattering_data = scat_data,
                                   scattering_meta_data = scat_meta_data)

gem_hydrometeors        = [gem_ice, gem_snow, gem_hail, gem_graupel, gem_liquid, gem_rain]
gem_hydrometeors_binned = [gem_ice_binned, gem_snow_binned, gem_hail_binned,
                           gem_graupel_binned, gem_liquid_binned, gem_rain_binned]