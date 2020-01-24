from parts.utils.data_providers import NetCDFDataProvider
import os

import mcrf.liras.setup
import mcrf.liras
from   mcrf.retrieval          import CloudRetrieval
from   mcrf.sensors            import mwi, ici, lcpr
from   mcrf.liras.passive_only import rh_a_priori, cloud_water_a_priori
from   mcrf.liras.passive_only_single_species import ice, rain
#from   mcrf.liras              import ice, liquid, snow, rain, rh_a_priori, cloud_water_a_priori
from   mcrf.liras.model_data   import ModelDataProvider

import matplotlib.pyplot as plt

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(mcrf.liras.liras_path, "data", "forward_simulations_b_noise.nc")

offsets = {"a" : 3000,
           "b" : 2800}

scene = "b"
offset = offsets[scene]
observations = NetCDFDataProvider(filename)
observations.add_offset("profile", -offset)

#
# Create the data provider.
#

ip = offset + 253 * 3 + 32

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  liquid_psd = rain.psd,
                                  scene = scene.upper())

#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, rain]
sensors      = [mwi, ici]

#
# Add a priori providers.
#

observation_error = mcrf.liras.ObservationError(sensors,
                                                forward_model_error = False,
                                                scene = scene)

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(cloud_water_a_priori)
data_provider.add(rh_a_priori)
data_provider.add(observation_error)
data_provider.add(observations)

#
# Run the retrieval.
#
retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.simulation.retrieval.debug_mode = True
retrieval.run(ip)

ws = retrieval.simulation.workspace
r = retrieval.simulation.retrieval
rqs = r.retrieval_quantities
