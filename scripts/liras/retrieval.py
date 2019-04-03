from parts.utils.data_providers import NetCDFDataProvider

import crac.liras.setup
import crac.liras
from   crac.retrieval        import CloudRetrieval
from   crac.sensors          import mwi, mwi_full, ici, lcpr
from   crac.liras            import ice, liquid, snow, rain, rh_a_priori, \
 cloud_water_a_priori
from   crac.liras.model_data import ModelDataProvider

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(description = "Run LIRAS retrieval.")
parser.add_argument('filename',
                    type = str,
                    nargs = 1,
                    help = "Filename containing observations to retrieve.")
args = parser.parse_args()
filename   = args.filename[0]
liras_path = crac.liras.liras_path

if not os.path.isabs(filename):
    filename = os.path.join(liras_path, filename)

#
# Load observations.
#

print(filename)
ts = filename[-6 : -3]
i_start, i_end, scene = crac.liras.test_scenes[ts]
n  = i_end - i_start
fe = "avg"  in filename
me = "full" in filename

observations = NetCDFDataProvider(filename)
offset = (n - observations.file_handle.dimensions["ao"].size) // 2
observations.add_offset("ao", -(i_start + offset))

#
# Create the data provider.
#

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  snow_psd   = snow.psd,
                                  liquid_psd = liquid.psd,
                                  scene = scene)
#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, snow, rain]
mwi_full.name = "mwi_full"
sensors      = [lcpr, mwi_full, ici]

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
data_provider.add(crac.liras.ObservationError(sensors))
data_provider.add(observations)

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()

output_dir = os.path.dirname(filename)
name       = os.path.basename(filename)
output_file = os.path.join(output_dir, name.replace("input", "output"))

retrieval.simulation.initialize_output_file(output_file, [("profile", n - 2 * offset, i_start + offset)])
retrieval.simulation.run_mpi(range(i_start + offset, i_end - offset))
