from artssat.utils.data_providers import NetCDFDataProvider
import mcrf.joint_flight.setup

import numpy as np
from mpi4py import MPI

from mcrf.retrieval import CloudRetrieval
from mcrf.sensors import hamp_radar, hamp_passive
from mcrf.joint_flight import (ice, snow, rain, cloud_water_a_priori,
                               h2o_a_priori, ObservationError,
                               temperature_a_priori, ismar)

from artssat.retrieval.a_priori import SensorNoiseAPriori
from mcrf.joint_flight import psd_shapes_high, psd_shapes_low
from mcrf.psds import D14NDmIce

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(description="Run joint flight retrieval.")
parser.add_argument('i_start',
                    type=int,
                    nargs=1,
                    help="Start of range of profiles to retrieve.")
parser.add_argument('i_end',
                    type=int,
                    nargs=1,
                    help="End of range of profiles to retrieve.")
parser.add_argument('shape', metavar='shape', type=str, nargs=1)
#parser.add_argument('config', metavar='config', type=str, nargs=1)
args = parser.parse_args()
i_start = args.i_start[0]
i_end = args.i_end[0]
shape = args.shape[0]
#config = args.config[0]

#
# Load observations.
#

filename = os.path.join(mcrf.joint_flight.path, "data", "combined/input.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#
path = mcrf.joint_flight.path
ice_shape = os.path.join(path, "data", "scattering_data", shape)
ice.scattering_data = ice_shape
#if config == "low":
#    alpha, log_beta = psd_shapes_low[shape]
#    ice.psd = D14NDmIce(alpha, np.exp(log_beta))
#if config == "high":
#    alpha, log_beta = psd_shapes_high[shape]
#    ice.psd = D14NDmIce(alpha, np.exp(log_beta))

hydrometeors = [ice, rain]
sensors = [hamp_radar, hamp_passive, ismar]

#
# Add a priori providers.
#

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(cloud_water_a_priori)
data_provider.add(h2o_a_priori)
data_provider.add(temperature_a_priori)
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup(verbosity=0)

output_dir = os.path.dirname(filename)
name = os.path.basename(filename)
filename = name.replace("input", "output_" + shape)
output_file = os.path.join(output_dir, filename)

retrieval.simulation.initialize_output_file(
    output_file, [("profile", i_end - i_start, i_start)],
    full_retrieval_output=False)
retrieval.simulation.run_ranges(range(i_start, min(i_end, 1441)))
