from parts.utils.data_providers import NetCDFDataProvider
import os

import crac.liras.setup
import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import mwi, ici, lcpr
from   crac.liras            import ice, liquid, snow, rain, rh_a_priori, cloud_water_a_priori
from   crac.liras.model_data import ModelDataProvider

import matplotlib.pyplot as plt

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(crac.liras.liras_path, "data", "reduced_new/retrieval_input_full_ts2.nc")

offsets = {"ts1" : 15,
           "ts2" : 2815,
           "ts3" : 2515}
ts = filename[-6:-3]
offset = offsets[ts]
observations = NetCDFDataProvider(filename)
observations.add_offset("ao", -offset)

#
# Create the data provider.
#

ip = offset + 460

if "ts3" in filename:
    scene = "B"
else:
    scene = "A"

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  snow_psd   = snow.psd,
                                  liquid_psd = liquid.psd,
                                  scene = scene)
#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, snow, rain]
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
data_provider.add(cloud_water_a_priori)
data_provider.add(rh_a_priori)
data_provider.add(crac.liras.ObservationError(sensors))
data_provider.add(observations)

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(ip)

