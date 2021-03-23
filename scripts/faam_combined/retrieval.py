import argparse
import os

from mpi4py import MPI

from artssat.utils.data_providers import NetCDFDataProvider
import mcrf.joint_flight.setup
import mcrf.liras
from mcrf.retrieval import CloudRetrieval
from mcrf.sensors import cloud_sat
from mcrf.faam_combined import ice, snow, rain, cloud_water_a_priori, \
    h2o_a_priori, ObservationError, temperature_a_priori, ismar, marss

from artssat.retrieval.a_priori import SensorNoiseAPriori
from mcrf.psds import D14NDmIce

#
# Parse arguments
#

parser = argparse.ArgumentParser(description="Run FAAM retrieval.")
parser.add_argument('i_start',
                    type=int,
                    nargs=1,
                    help="Start of range of profiles to retrieve.")
parser.add_argument('i_end',
                    type=int,
                    nargs=1,
                    help="End of range of profiles to retrieve.")
parser.add_argument('shape', metavar='shape', type=str, nargs=1)
parser.add_argument('flight', metavar='flight_number', type=str, nargs=1)
#parser.add_argument('config', metavar='config', type=str, nargs=1)
args = parser.parse_args()
i_start = args.i_start[0]
i_end = args.i_end[0]
shape = args.shape[0]
flight = args.flight[0]

#
# Load observations.
#

filename = os.path.join(mcrf.joint_flight.path, "data", f"input_c{flight}.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#

path = mcrf.joint_flight.path
ice_shape = os.path.join(path, "data", "scattering_data", shape + ".xml")
ice.scattering_data = ice_shape

hydrometeors = [ice, rain]
sensors = [cloud_sat, marss, ismar]
#sensors = [cloud_sat]#, marss, ismar]

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
output_file = name.replace("input", "output_" + shape)
output_file = os.path.join(output_dir, output_file)

n_rays = data_provider.file_handle.dimensions["rays"].size
retrieval.simulation.initialize_output_file(
    output_file, [("profile", n_rays, 0)],
    full_retrieval_output=False)
retrieval.simulation.run_ranges(range(0, n_rays))
