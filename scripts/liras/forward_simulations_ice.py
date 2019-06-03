from parts.utils.data_providers import NetCDFDataProvider
import numpy as np
import os

import argparse
import crac.liras.setup
import crac.liras
from   crac.retrieval        import CloudSimulation
from   crac.sensors          import mwi, ici, lcpr
from   crac.liras            import ice, liquid, snow, rain, rh_a_priori, cloud_water_a_priori
from   crac.liras.gem        import gem_hydrometeors
from   crac.liras.model_data import ModelDataProvider

parser = argparse.ArgumentParser(prog = "LIRAS forward simulations",
                                 description = 'Forward simulations using GEM model physics.')
parser.add_argument('scene',       metavar = 'scene',       type = str, nargs = 1)
parser.add_argument('start_index', metavar = 'start_index', type = int, nargs = 1)
parser.add_argument('end_index',   metavar = 'end_index',   type = int, nargs = 1)
parser.add_argument('output_file', metavar = 'output_file', type = str, nargs = 1)

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

hydrometeors = gem_hydrometeors
kwargs = {"ice_psd"     : hydrometeors[0].psd,
          "snow_psd"    : hydrometeors[1].psd,
          "hail_psd"    : hydrometeors[2].psd,
          "graupel_psd" : hydrometeors[3].psd,
          "liquid_psd"  : hydrometeors[4].psd}
data_provider = ModelDataProvider(99, scene = scene, **kwargs)
sensors       = [lcpr, mwi, ici]
simulation    = CloudSimulation(hydrometeors[:1], sensors, data_provider)

simulation.setup(verbosity = 0)
dimensions = [("profile", n, i_start)]
simulation.simulation.initialize_output_file(output_file, dimensions, "w")
simulation.simulation.run_ranges(range(i_start, i_end))
