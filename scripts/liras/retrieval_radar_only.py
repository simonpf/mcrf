from parts.utils.data_providers import NetCDFDataProvider

import crac.liras.setup
import crac.liras
from   crac.retrieval          import CloudRetrieval
from   crac.sensors            import mwi, mwi_full, ici, lcpr
from   crac.liras              import rh_a_priori, cloud_water_a_priori
from   crac.liras.model_data import ModelDataProvider

#
# Parse arguments
#

import argparse
import os

parser = argparse.ArgumentParser(prog = "LIRAS retrieval",
                                 description = 'Passive ice cloud retrieval')
parser.add_argument('scene',       metavar = 'scene',       type = str, nargs = 1)
parser.add_argument('start_index', metavar = 'start_index', type = int, nargs = 1)
parser.add_argument('ice_shape',   metavar = 'ice_shape', type = str, nargs = 1)
parser.add_argument('snow_shape',   metavar = 'snow_shape', type = str, nargs = 1)
parser.add_argument('input_file',  metavar = 'input_file', type = str, nargs = 1)
parser.add_argument('output_file', metavar = 'output_file', type = str, nargs = 1)

args = parser.parse_args()

scene        = args.scene[0]
i_start      = args.start_index[0]
ice_shape    = args.ice_shape[0]
snow_shape   = args.snow_shape[0]
input_file   = args.input_file[0]
output_file  = args.output_file[0]

liras_path = crac.liras.liras_path

if not os.path.isabs(input_file):
    input_file = os.path.join(liras_path, input_file)

if not os.path.isabs(output_file):
    output_file = os.path.join(liras_path, output_file)

#
# Load observations.
#

observations = NetCDFDataProvider(input_file, parallel = True)
observations.add_offset("profile", -i_start)
n = observations.file_handle.dimensions["profile"].size

if not snow_shape == "None":
    from crac.liras import ice, snow, rain
    ice_shape = os.path.join(liras_path, "data", "scattering", ice_shape)
    ice.scattering_data = ice_shape
    snow_shape = os.path.join(liras_path, "data", "scattering", snow_shape)
    snow.scattering_data = snow_shape
    hydrometeors = [ice, snow, rain]
else:
    from crac.liras.single_species import ice, snow, rain
    ice_shape = os.path.join(liras_path, "data", "scattering", ice_shape)
    ice.scattering_data = ice_shape
    hydrometeors = [ice, rain]


#
# Create the data provider.
#

data_provider = ModelDataProvider(99,
                                  ice_psd    = ice.psd,
                                  scene = scene.upper())
#
# Define hydrometeors and sensors.
#

sensors = [lcpr]

#
# Add a priori providers.
#

observation_errors = crac.liras.ObservationError(sensors, footprint_error = False, forward_model_error = False)
data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(snow.a_priori[0])
data_provider.add(snow.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(observation_errors)
data_provider.add(observations)

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider, include_cloud_water = False)
retrieval.setup()


retrieval.simulation.initialize_output_file(output_file, [("profile", n, i_start)])
retrieval.simulation.run_ranges(range(i_start, i_start + n))
