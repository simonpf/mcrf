from parts.utils.data_providers import NetCDFDataProvider
import numpy as np
import os

import argparse
import crac.liras.setup
import crac.liras
from   crac.retrieval        import CloudSimulation
from   crac.sensors          import mwi, ici, lcpr, mwi_full, hamp_radar
from   crac.liras            import ice, rain
from   crac.liras.gem        import gem_hydrometeors
from   crac.liras.model_data import ModelDataProvider

parser = argparse.ArgumentParser(prog = "LIRAS forward simulations",
                                 description = 'Forward simulations using GEM model physics.')
parser.add_argument('scene',       metavar = 'scene',       type = str, nargs = 1)
parser.add_argument('start_index', metavar = 'start_index', type = int, nargs = 1)
parser.add_argument('end_index',   metavar = 'end_index',   type = int, nargs = 1)
parser.add_argument('output_file', metavar = 'output_file', type = str, nargs = 1)
parser.add_argument('sensors',     metavar = 'sensors', type = str, nargs = '*',
                    default = ["lcpr", "ici", "mwi"])

parser.add_argument("--simple", dest = "mode", action = "store_const",
                    const = "simple", default = "full",
                    help = "Use simplified microphysics")

args = parser.parse_args()
i_start = args.start_index[0]
i_end   = args.end_index[0]
scene   = args.scene[0]
output_file = args.output_file[0]

print(i_start, i_end, scene, output_file)

n = i_end - i_start

#
# Setup the simulation.
#

if parser.mode == "simple":
    hydrometeors = [ice, rain]
    include_cloud_water = True
else:
    hydrometeors = gem_hydrometeors
    include_cloud_water = False

arg_names = ["ice_psd", "snow_psd", "hail_psd", "graupel_psd", "liquid_psd"]
kwargs = dict(zip(arg_names, [h.psd for h in hydrometeors]))
data_provider = ModelDataProvider(99, scene = scene, **kwargs)
sensors = [getattr(crac.sensors, n) for n in args.sensors]

simulation    = CloudSimulation(hydrometeors, sensors, data_provider,
                                include_cloud_water)

simulation.setup(verbosity = 0)
dimensions = [("profile", n, i_start)]
simulation.simulation.initialize_output_file(output_file, dimensions)
simulation.simulation.run_ranges(range(i_start, i_end))
