from parts.utils.data_providers import NetCDFDataProvider

import mcrf.liras.setup
import mcrf.liras
from   mcrf.retrieval        import CloudRetrieval
from   mcrf.sensors          import mwi, mwi_full, ici, lcpr
from   mcrf.liras            import ice, liquid, rain, snow, rh_a_priori, \
 cloud_water_a_priori
from   mcrf.liras.model_data import ModelDataProvider
from   mcrf.liras.gem import gem_ice, gem_snow

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(prog = "LIRAS retrieval",
                                 description = 'Passive ice cloud retrieval')
parser.add_argument('scene',       metavar = 'scene',       type = str, nargs = 1)
parser.add_argument('start_index', metavar = 'start_index', type = int, nargs = 1)
parser.add_argument('input_file',  metavar = 'input_file', type = str, nargs = 1)
parser.add_argument('output_file', metavar = 'output_file', type = str, nargs = 1)

args = parser.parse_args()

scene        = args.scene[0]
i_start      = args.start_index[0]
input_file   = args.input_file[0]
output_file  = args.output_file[0]

liras_path = mcrf.liras.liras_path

if not os.path.isabs(input_file):
    input_file = os.path.join(liras_path, input_file)

if not os.path.isabs(output_file):
    output_file = os.path.join(liras_path, output_file)

#
# Load observations.
#

observations = NetCDFDataProvider(input_file)
observations.add_offset("profile", -i_start)
n = observations.file_handle.dimensions["profile"].size

#
# Create the data provider.
#

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  liquid_psd = liquid.psd,
                                  scene = scene.upper())
#
# Define hydrometeors and sensors.
#

ice.scattering_data = (gem_ice.scattering_data, gem_ice.scattering_meta_data)
snow.scattering_data = (gem_snow.scattering_data, gem_snow.scattering_meta_data)

hydrometeors = [ice, snow, rain]
sensors = [lcpr, mwi, ici]

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
data_provider.add(mcrf.liras.ObservationError(sensors))
data_provider.add(observations)

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()


retrieval.simulation.initialize_output_file(output_file, [("profile", n, i_start)])
retrieval.simulation.run_ranges(range(i_start, i_start + n))
