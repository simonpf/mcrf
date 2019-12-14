from parts.utils.data_providers import NetCDFDataProvider
import mcrf.joint_flight.setup
import mcrf.liras
import numpy as np

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
from mpi4py import MPI

import mcrf.liras
from mcrf.retrieval import CloudSimulation
from mcrf.sensors import gmi, ici
from examples.data_provider.atmosphere import Tropical
from   mcrf.joint_flight     import ice, snow, rain, cloud_water_a_priori, \
    rh_a_priori, ObservationError, temperature_a_priori

from parts.retrieval.a_priori import SensorNoiseAPriori

#
# Parse arguments
#

import argparse
import os
import glob

parser = argparse.ArgumentParser(prog="ICI/MWI Forward simulations.",
                                 description="Retrieved ice and rain from DARDAR data")
parser.add_argument('filename', metavar='filename', type=str, nargs=1)
parser.add_argument('i_start', metavar='i_start', type=int, nargs=1)
parser.add_argument('i_end', metavar='i_end', type=int, nargs=1)
parser.add_argument('stype', metavar='stype', type=str, nargs=1)

args = parser.parse_args()
filename = args.filename[0]
i_start = args.i_start[0]
i_end = args.i_end[0]
stype = args.stype[0]


class DardarProvider(Tropical):
    def __init__(self, filename, i_start, i_end):
        self.filename = filename
        self.z = np.linspace(0, 20e3, 41)
        self.offset = i_start
        super().__init__(z = self.z)


if stype == "ice":
    hydrometeors = [ice, rain]
elif stype == "rain":
    hydrometeors = [ rain]
elif stype == "clear":
    hydrometeors = []
else:
    raise Exception("Type arguments must be ice, rain or clear.")

data_provider = DardarProvider(filename, i_start, i_end)
data_provider.add(NetCDFDataProvider("retrieval_results.nc", group = -1))

#
# Define hydrometeors and sensors.
#

sensors = [ici, gmi]

#
# Run forward simulation.
#

retrieval = CloudSimulation(hydrometeors, sensors, data_provider)
retrieval.setup(verbosity=0)

output_dir = os.path.dirname(filename)
name = os.path.basename(filename)
output_file = os.path.join(output_dir, "forward_simulation_{}.nc".format(stype))

retrieval.simulation.initialize_output_file(output_file, [("profile", i_end - i_start, i_start)])
retrieval.simulation.run_ranges(range(i_start, i_end, 1))
