from parts.utils.data_providers import NetCDFDataProvider
import numpy as np
import os

import argparse
import mcrf.liras.setup
import mcrf.liras
from mcrf.retrieval import CloudSimulation
from mcrf.sensors import mwi, ici, lcpr, mwi_full, hamp_radar
from mcrf.liras import ice, rain
from mcrf.liras.gem import gem_hydrometeors
from mcrf.liras.model_data import ModelDataProvider

parser = argparse.ArgumentParser(
    prog="LIRAS forward simulations",
    description='Forward simulations using GEM model physics.')
parser.add_argument('scene', metavar='scene', type=str, nargs=1)
parser.add_argument('start_index', metavar='start_index', type=int, nargs=1)
parser.add_argument('end_index', metavar='end_index', type=int, nargs=1)
parser.add_argument('ice_shape', metavar='ice_shape', type=str, nargs=1)
parser.add_argument('output_file', metavar='output_file', type=str, nargs=1)
parser.add_argument('sensors',
                    metavar='sensors',
                    type=str,
                    nargs='*',
                    default=["lcpr", "ici", "mwi"])

args = parser.parse_args()
i_start = args.start_index[0]
i_end = args.end_index[0]
ice_shape = args.ice_shape[0]
scene = args.scene[0]
output_file = args.output_file[0]

n = i_end - i_start

#
# Setup the simulation.
#

liras_path = mcrf.liras.liras_path
ice_shape = os.path.join(liras_path, "data", "scattering", ice_shape)
hydrometeors = gem_hydrometeors
include_cloud_water = False
for h in hydrometeors[:4]:
    h.scattering_data = ice_shape

arg_names = ["ice_psd", "snow_psd", "hail_psd", "graupel_psd", "liquid_psd"]
kwargs = dict(zip(arg_names, [h.psd for h in hydrometeors]))
data_provider = ModelDataProvider(99, scene=scene, **kwargs)
sensors = [getattr(mcrf.sensors, n) for n in args.sensors]

simulation = CloudSimulation(hydrometeors,
                             sensors,
                             data_provider,
                             include_cloud_water=include_cloud_water)

simulation.setup(verbosity=0)
dimensions = [("profile", n, i_start)]
simulation.simulation.initialize_output_file(output_file, dimensions)
simulation.simulation.run_ranges(range(i_start, i_end))
