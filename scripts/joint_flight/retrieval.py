from parts.utils.data_providers import NetCDFDataProvider
import crac.joint_flight.setup

import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import hamp_radar, hamp_passive, ismar
from   crac.joint_flight     import ice, liquid, snow, rain, liquid_md_a_priori, \
    rh_a_priori, ObservationError

from parts.retrieval.a_priori import SensorNoiseAPriori

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(description = "Run joint flight retrieval.")
parser.add_argument('i_start',
                    type = int,
                    nargs = 1,
                    help = "Start of range of profiles to retrieve.")
parser.add_argument('i_end',
                    type = int,
                    nargs = 1,
                    help = "End of range of profiles to retrieve.")
parser.add_argument('suffix',
                    type = str,
                    nargs = 1,
                    help = "Suffix to append to output filename.")
args = parser.parse_args()
i_start = args.i_start[0]
i_end   = args.i_end[0]
filesuffix = args.suffix[0]

#
# Load observations.
#

filename     = os.path.join(crac.joint_flight.path, "data", "input.nc")
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
data_provider.add(liquid_md_a_priori)
data_provider.add(rh_a_priori)
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup(verbosity = 0)

output_dir = os.path.dirname(filename)
name       = os.path.basename(filename)
output_file = os.path.join(output_dir, name.replace("input", "output" + filesuffix))

retrieval.simulation.initialize_output_file(output_file, [("profile", i_end - i_start, i_start)])
#retrieval.simulation.run_mpi(range(1441))
retrieval.simulation.run_mpi(range(i_start, min(i_end, 1441)))
#retrieval.simulation.run_mpi(range(1080, 1160))
#retrieval.simulation.run_mpi(range(1220, 1350))
