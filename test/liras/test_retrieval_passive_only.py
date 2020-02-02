import os
import numpy as np

os.environ["LIRAS_PATH"] = "/home/simonpf/src/joint_flight"
os.environ["ARTS_DATA_PATH"] = "/home/simonpf/src/arts_xml"
os.environ["ARTS_BUILD_PATH"] = "/home/simonpf/build/arts_fast"

import mcrf.liras.setup
import mcrf.liras
from   mcrf.liras              import ObservationError
from   mcrf.retrieval          import CloudRetrieval
from   mcrf.sensors            import mwi, ici, lcpr
from   mcrf.liras.passive_only import h2o_a_priori, cloud_water_a_priori
from   mcrf.liras.passive_only_single_species import ice, rain
from   mcrf.liras.model_data   import ModelDataProvider
from parts.utils.data_providers import NetCDFDataProvider

import matplotlib.pyplot as plt

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(mcrf.liras.liras_path, "data", "forward_simulations_a_noise.nc")
scene = "a"
offsets = {"a" : 3000,
           "b" : 2800}
offset = offsets[scene]
observations = NetCDFDataProvider(filename)
observations.add_offset("profile", -offset)

#
# Create the data provider.
#

ip = offset + 300 #213 * 3 + 32
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

observation_error = ObservationError(sensors,
                                     forward_model_error = False,
                                     scene = scene)

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(cloud_water_a_priori)
data_provider.add(h2o_a_priori)
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
