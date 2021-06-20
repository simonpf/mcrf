import os
os.environ["JOINT_FLIGHT_PATH"] = "/home/simonpf/src/joint_flight"
from artssat.utils.data_providers import NetCDFDataProvider
import mcrf.joint_flight.setup
import mcrf.liras

import numpy as np
import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
from mpi4py import MPI

import mcrf.liras
from mcrf.retrieval import CloudRetrieval
from mcrf.sensors import hamp_radar, hamp_passive
from mcrf.joint_flight import ice, snow, rain, cloud_water_a_priori, \
    h2o_a_priori, ObservationError, temperature_a_priori, ismar, marss

from artssat.retrieval.a_priori import SensorNoiseAPriori
from mcrf.joint_flight import psd_shapes_high, psd_shapes_low
from mcrf.psds import D14NDmIce

#
# Parse arguments
#

i_start = 220
i_end = i_start + 1
shape = "LargePlateAggregate"


#
# Load observations.
#

filename = os.path.join(mcrf.joint_flight.path, "data", "input_b984.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#
ice_shape = os.path.join(mcrf.joint_flight.path, "data", "scattering_data", shape)
ice.scattering_data = ice_shape
#if config == "low":
#    alpha, log_beta = psd_shapes_low[shape]
#    ice.psd = D14NDmIce(alpha, np.exp(log_beta))
#if config == "high":
#    alpha, log_beta = psd_shapes_high[shape]
#    ice.psd = D14NDmIce(alpha, np.exp(log_beta))

hydrometeors = [ice, rain]
sensors = [hamp_radar, marss, ismar]

#
# Add a priori providers.
#

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.simulation.retrieval.debug_mode = True
retrieval.setup(verbosity=0)

output_dir = os.path.dirname(filename)
name = os.path.basename(filename)
filename = name.replace("input", "output_" + shape)

retrieval.simulation.run_ranges(range(i_start, min(i_end, 1441)))
ws = retrieval.simulation.workspace
