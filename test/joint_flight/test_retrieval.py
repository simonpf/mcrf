import os
os.environ["JOINT_FLIGHT_PATH"] = "/home/simonpf/src/joint_flight"
os.environ["ARTS_DATA_PATH"] = "/home/simonpf/src/joint_flight/data"

import mcrf.joint_flight.setup

import mcrf.liras
from   mcrf.retrieval        import CloudRetrieval
from   mcrf.sensors          import hamp_radar, hamp_passive, ismar
from   mcrf.joint_flight     import ice, snow, rain, cloud_water_a_priori, \
    rh_a_priori, ObservationError, temperature_a_priori
from parts.retrieval.a_priori import SensorNoiseAPriori
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

filename     = os.path.join(mcrf.joint_flight.path, "data", "input.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, rain]
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
data_provider.add(cloud_water_a_priori)
data_provider.add(rh_a_priori)
data_provider.add(temperature_a_priori)
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(100)
