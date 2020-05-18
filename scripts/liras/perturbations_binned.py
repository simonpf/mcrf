from artssat.utils.data_providers import NetCDFDataProvider
import numpy as np
import os

import argparse
import mcrf.liras.setup
import mcrf.liras
from   mcrf.retrieval        import CloudSimulation
from   mcrf.sensors          import mwi, ici, lcpr
from   mcrf.liras            import ice, liquid, snow, rain, h2o_a_priori, cloud_water_a_priori
from   mcrf.liras.gem        import gem_hydrometeors, gem_hydrometeors_binned
from   mcrf.liras.model_data import ModelDataProvider

parser = argparse.ArgumentParser(prog = "LIRAS forward simulations",
                                 description = 'Run binned perturbation calculation.')
parser.add_argument('start_index', metavar = 'start_index', type = int, nargs = '1')
parser.add_argument('end_index',   metavar = 'end_index',   type = int, nargs = '1')
parser.add_argument('scene',       metavar = 'scene',       type = str, nargs = '1')
parser.add_argument('output_file', metavar = 'output_file', type = str, nargs = '1')

args = parser.parse_args()
i_start = parser.start_index
i_end   = parser.end_index
scene   = parse.scene
output_file = parser.output_file

n = i_end, i_start

#
# Setup the simulation.
#

hydrometeors = gem_hydrometeors_binned
kwargs = {"ice_psd"     : hydrometeors[0].psd,
          "snow_psd"    : hydrometeors[1].psd,
          "hail_psd"    : hydrometeors[2].psd,
          "graupel_psd" : hydrometeors[3].psd,
          "liquid_psd"  : hydrometeors[4].psd}
data_provider = ModelDataProvider(99, scene = scene, **kwargs)
sensors      = [lcpr, mwi, ici]
simulation = CloudSimulation(hydrometeors, sensors, data_provider)
simulation.setup()

simulation.initialize_output_file(output_file,
                                  [("profile", (n, i_start)),
                                   ("bin", (21, 0))])
simulation.run_ranges(range(i_start, i_end), range(21))
