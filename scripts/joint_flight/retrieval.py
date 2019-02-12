from parts.utils.data_providers import NetCDFDataProvider
import crac.joint_flight.setup

import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import hamp_radar, hamp_passive
from   crac.joint_flight     import ice, liquid, snow, rain, rh_a_priori

from parts.retrieval.a_priori import SensorNoiseAPriori

import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(description = "Run joint flight retrieval.")
parser.add_argument('suffix',
                    type = str,
                    nargs = 1,
                    help = "Suffix to append to output filename.")
args = parser.parse_args()
filesuffix = args.suffix[0]

#
# Load observations.
#

filename     = os.path.join(crac.joint_flight.path, "data", "input.nc")
data_provider = NetCDFDataProvider(filename)

#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, snow, liquid, rain]
sensors      = [hamp_radar, hamp_passive]

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
data_provider.add(SensorNoiseAPriori(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()

output_dir = os.path.dirname(filename)
name       = os.path.basename(filename)
output_file = os.path.join(output_dir, name.replace("input", "output_" + filesuffix))

retrieval.simulation.initialize_output_file(output_file)
retrieval.simulation.run_mpi(range(0, 520))
retrieval.simulation.run_mpi(range(780, 940))
retrieval.simulation.run_mpi(range(1080, 1160))
retrieval.simulation.run_mpi(range(1220, 1350))
