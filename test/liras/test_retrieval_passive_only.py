from parts.utils.data_providers import NetCDFDataProvider
import os

import crac.liras.setup
import crac.liras
from   crac.retrieval          import CloudRetrieval
from   crac.sensors            import mwi, ici, lcpr
from   crac.liras.passive_only import ice, liquid, snow, rain, rh_a_priori, cloud_water_a_priori
#from   crac.liras              import ice, liquid, snow, rain, rh_a_priori, cloud_water_a_priori
from   crac.liras.model_data   import ModelDataProvider

import matplotlib.pyplot as plt

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(crac.liras.liras_path, "data", "forward_simulations_a_noise.nc")

offsets = {"a" : 3000,
           "b" : 2800}

scene = "a"
offset = offsets[scene]
observations = NetCDFDataProvider(filename)
observations.add_offset("profile", -offset)

#
# Create the data provider.
#

ip = offset + 1

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  snow_psd   = snow.psd,
                                  liquid_psd = liquid.psd,
                                  scene = scene.upper())

#
# Define hydrometeors and sensors.
#

#hydrometeors = [ice, snow, liquid, rain]
#hydrometeors = [ice, snow, rain]
hydrometeors = [ice, rain]
sensors      = [mwi, ici]

#
# Add a priori providers.
#

observation_errors = crac.liras.ObservationError(sensors,
                                                 footprint_error = False,
                                                 forward_model_error = False)

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(snow.a_priori[0])
data_provider.add(snow.a_priori[1])
data_provider.add(liquid.a_priori[0])
data_provider.add(liquid.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(rh_a_priori)
data_provider.add(crac.liras.ObservationError(sensors))
data_provider.add(observations)

#
# Run the retrieval.
#
retrieval = CloudRetrieval(hydrometeors, sensors, data_provider, include_cloud_water = False)
retrieval.setup()
retrieval.simulation.retrieval.debug_mode = True
retrieval.run(ip)
