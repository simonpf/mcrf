from parts.utils.data_providers import NetCDFDataProvider
import os

os.environ["JOINT_FLIGHT_PATH"] = "/home/simonpf/data/joint_flight"
import crac.joint_flight.setup

import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import hamp_radar, hamp_passive, ismar
from   crac.joint_flight     import ice, liquid, snow, rain, liquid_md_a_priori, \
    rh_a_priori, ObservationError

from parts.retrieval.a_priori import SensorNoiseAPriori

import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(crac.joint_flight.path, "data", "input.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, snow, rain]
sensors      = [hamp_radar, hamp_passive, ismar]

#
# Add a priori providers.
#

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(snow.a_priori[0])
data_provider.add(snow.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(liquid_md_a_priori)
data_provider.add(rh_a_priori)
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(700)
