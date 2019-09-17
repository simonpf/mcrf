from parts.utils.data_providers import NetCDFDataProvider
import os
import numpy as np

import mcrf.liras.setup
import mcrf.liras
from   mcrf.retrieval        import CloudRetrieval
from   mcrf.sensors          import mwi, ici, lcpr
#from   mcrf.liras            import ice, snow, rain, rh_a_priori
from   mcrf.liras  import snow, rh_a_priori, cloud_water_a_priori
from   mcrf.liras.single_species import ice, rain
from   mcrf.liras.model_data import ModelDataProvider

import matplotlib.pyplot as plt

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

filename     = os.path.join(mcrf.liras.liras_path, "data", "forward_simulations_b_noise.nc")

offsets = {"a" : 3000,
           "b" : 2200}

scene = filename.split("_")[-2]
offset = offsets[scene]
observations = NetCDFDataProvider(filename)
observations.add_offset("profile", -offset)

#
# Create the data provider.
#

ip = offset + 735

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  snow_psd   = snow.psd,
                                  scene = scene.upper())


#
# Define hydrometeors and sensors.
#

#hydrometeors = [snow, ice, rain]
hydrometeors = [ice, rain]
sensors      = [lcpr, mwi, ici]

#
# Add a priori providers.
#

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(snow.a_priori[0])
data_provider.add(snow.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(rh_a_priori)
data_provider.add(cloud_water_a_priori)
data_provider.add(mcrf.liras.ObservationError(sensors))
data_provider.add(observations)


#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(ip)

x = np.array([  4, -18.4305,   4, 18.9891, 7.04546, 6.99871, 7.09073, 7.62414, 8.89859, 9.52002, 9.77306,   4,   7, 7.93238, -7.48697e-15, 4.7845, 5.82234, -3.20297e-213, 1e-08, 1e-08, 1e-08, 0.00130508, 0.00131268, 0.00132073, 0.00126157, 0.00113602, 0.000967911, 1e-08, 1e-08, 1e-08, 1e-08, -0.000284216, -0.000464534, -0.000308766, -0.000173336, 7.23606e-05, 0.000312281, 0.00042428, 0.000515418, 0.000582336, 0.000593246, 0.000588204, 0.000486826, 0.000392691, 0.000249942, 0.000142881, 0.000113607, 9.3196e-05, 8.279e-05, 7.439e-05, 6.71764e-05, 6.35955e-05, 5.82613e-05, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.0005, 0.000591103, 0.000535705, 1e-08, 1e-08, 1e-08, 0.00057082, 0.000873984, -8.58573e-05, -7.4156e-05, 0.000680967, 1e-08, -0.517766, -0.826137, -0.823679, -0.473084, 0.0221514, 0.981336, 1.55711, 1.75276, 1.59097, 0.890358, -0.390723, -0.910884, -1.09464, -1.16876, -1.19602, -1.18752, -1.17785, -1.1576, -1.13895, -1.12995, -1.11121,  -6, -6.29586, -6.42502, -6.45757, -6.47558, -6.49403, -6.49045, -6.40652, -20.3196])
