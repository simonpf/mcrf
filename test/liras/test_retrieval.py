from parts.utils.data_providers import NetCDFDataProvider

import crac.liras.setup
import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import mwi, ici, lcpr
from   crac.liras            import ice, liquid, snow, rain, rh_a_priori
from   crac.liras.model_data import ModelDataProvider

from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(crac.liras.liras_path, "data", "reduced/retrieval_input_full_ts1.nc")

offsets = {"ts1" : 15,
           "ts2" : 2815,
           "ts3" : 2515}
observations = NetCDFDataProvider(filename)
observations.add_offset("ao", -offsets[filename[-6:-3]])

#
# Create the data provider.
#

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

hydrometeors = [ice, snow, liquid, rain]
sensors      = [lcpr, mwi, ici]

#
# Add a priori providers.
#

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

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(15)

