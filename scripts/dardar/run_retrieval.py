import os
os.environ["ARTS_DATA_PATH"] = "/home/simonpf/src/joint_flight/data"

import mcrf.liras.setup
import mcrf.liras
from mcrf.retrieval import CloudRetrieval
from mcrf.sensors import mwi, mwi_full, ici, cloud_sat
from mcrf.liras.model_data import ModelDataProvider
from artssat.utils.data_providers import NetCDFDataProvider
from examples.data_provider.atmosphere import Tropical
from pyhdf.SD import SD, SDC
import numpy as np
from   mcrf.liras.single_species import ice, rain

#
# Parse arguments
#

import argparse

parser = argparse.ArgumentParser(prog="DARDAR retrieval",
                                 description="Retrieved ice and rain from DARDAR data")
parser.add_argument('filename', metavar='filename', type=str, nargs=1)
parser.add_argument('i_start', metavar='i_start', type=int, nargs=1)
parser.add_argument('i_end', metavar='i_end', type=int, nargs=1)

args = parser.parse_args()
filename = args.filename[0]
i_start = args.i_start[0]
i_end = args.i_end[0]

class DardarProvider(Tropical):
    def __init__(self, filename, i_start, i_end):
        self.filename = filename
        self.file_handle = SD(self.filename, SDC.READ)
        self.ys = self.file_handle.select("Z")[:][i_start : i_end, ::-1]
        self.rz = self.file_handle.select("height")[:][::-1]
        self.z = np.linspace(0, 20e3, 41)
        self.offset = i_start
        super().__init__(z = self.z)

    def get_y_cloud_sat(self, i):
        bins = cloud_sat.range_bins
        counts, _ = np.histogram(self.rz, bins = bins)
        avgs, _ = np.histogram(self.rz, weights = self.ys[i - i_start, :], bins = bins)
        y = 10 * np.log10(np.maximum(10 ** -2.6, avgs / counts))
        y[counts == 0] = -26.0
        inds = np.where(cloud_sat.range_bins > 1e3)[0][0] - 1
        y[:inds] = y[inds]
        return y


cloud_sat.y_min = -26.0
sensors = [cloud_sat]
hydrometeors = [ice, rain]

data_provider = DardarProvider(filename, i_start, i_end)
data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(mcrf.liras.ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup(verbosity = 0)
output_file = "retrieval_results.nc"
retrieval.simulation.initialize_output_file(output_file,
                                            [("profile", i_end - i_start, i_start)],
                                            full_retrieval_output=False)
retrieval.simulation.run_ranges(range(i_start, i_end))
