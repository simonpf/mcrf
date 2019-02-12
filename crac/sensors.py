"""
The :code:`crac.sensors` module contains Python classes that
represent the sensors that are used within the study. These
are:
    - The Ice Cloud Imager (ICI)
    - The Microwave Imager (MWI)
    - The LIRAS Cloud Profiling Radar (LCPR)

Attributes:

    ici: An instance of the ICI class that includes all of
       the available ICI channels.

    mwi: An instance of the MWI class that includes all of
        the available MWI channels.

    lcpr: The LIRAS cloud profiling radar.

"""
import numpy as np
import scipy as sp
import os
from netCDF4 import Dataset
from parts.sensor import PassiveSensor, ActiveSensor

################################################################################
# Ice cloud imager (ICI).
################################################################################

class ICI(PassiveSensor):
    """
    The Ice Cloud Imager (ICI) sensor.

    Attributes:

        channels(:code:`list`): List of channels that are available
            from ICI

        nedt(:code:`list`): Noise equivalent temperature differences for the
            channels in :code:`channels`.
    """
    channels = np.array([1.749100000000000e+11,
                         1.799100000000000e+11,
                         1.813100000000000e+11,
                         2.407000000000000e+11,
                         3.156500000000000e+11,
                         3.216500000000000e+11,
                         3.236500000000000e+11,
                         4.408000000000000e+11,
                         4.450000000000000e+11,
                         4.466000000000000e+11,
                         6.598000000000000e+11])

    nedt = np.array([0.8, 0.8, 0.8,       # 183 GHz
                     0.7 * np.sqrt(0.5),  # 243 GHz
                     1.2, 1.3, 1.5,       # 325 GHz
                     1.4, 1.6, 2.0,       # 448 GHz
                     1.6 * np.sqrt(0.5)]) # 664 GHz

    def __init__(self,
                 name = "ici",
                 channel_indices = None,
                 stokes_dimension = 1):
        """
        This creates an instance of the ICI sensor to be used within a
        :code:`parts` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor used within the parts
                simulation.

            channel_indices(:code:`list`): List of channel indices to be used
                in the simulation/retrieval.

            stokes_dimension(:code:`int`): The stokes dimension to use for
                the retrievals.
        """
        if channel_indices is None:
            channels  = ICI.channels
            self.nedt = ICI.nedt
        else:
            channels  = ICI.channels[channel_indices]
            self.nedt = self.nedt[channel_indices]
        super().__init__(name, channels, stokes_dimension = stokes_dimension)

################################################################################
# Microwave imager (MWI).
################################################################################

class MWI(PassiveSensor):
    """
    The Microwave Imager (MWI) sensor.

    Attributes:

        channels(:code:`list`): The list of the channels available from the
            MWI sensor.

        nedt(:code:`list`): The noise equivalent temperature differences for
            the channels in :code:`channels`.
    """
    channels = np.array([18.7e9,
                         23.8e9,
                         31.4e9,
                         50.3e9,
                         52.6e9,
                         53.24e9,
                         53.75e9,
                         89.0e9,
                         115.5503e9,
                         116.6503e9,
                         117.3503e9,
                         117.5503e9,
                         164.75e9,
                         176.31e9,
                         177.21e9,
                         178.41e9,
                         179.91e9,
                         182.01e9])

    nedt = np.array([0.8 * np.sqrt(0.5), #18 GHz
                     0.7 * np.sqrt(0.5), #24 GHz
                     0.9 * np.sqrt(0.5), #31 GHz
                     1.1 * np.sqrt(0.5), #50 GHz
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5), #89 GHz
                     1.3, #118 GHz
                     1.3,
                     1.3,
                     1.3,
                     1.2, #165 GHz
                     1.3, #183 GHz
                     1.2,
                     1.2,
                     1.2,
                     1.3])

    def __init__(self,
                 name = "mwi",
                 channel_indices = None,
                 stokes_dimension = 1):
        """
        Create an MWI instance to be used within a :code:`parts` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor to be used within the
                parts simulation.

            channel_indices(:code:`list`): List of channel indices to be used
                for the simulation.

            stokes_dimension(:code:`int`): The Stokes dimension to be used for
                the simulation.
        """
        if channel_indices is None:
            channels  = MWI.channels
            self.nedt = MWI.nedt
        else:
            channels  = MWI.channels[channel_indices]
            self.nedt = MWI.nedt[channel_indices]

        self.channels = channels
        super().__init__(name, channels, stokes_dimension = stokes_dimension)

################################################################################
# Hamp Passive
################################################################################

class HampPassive(PassiveSensor):

    channels = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40,
                         50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00,
                         90,
                         118.75 - 8.5, 118.75 - 4.2, 118.75 - 2.3, 118.75 - 1.4,
                         118.75 + 1.4, 118.75 + 2.3, 118.75 + 4.2, 118.75 + 8.5,
                         183.31 - 12.5, 183.31 - 7.5, 183.31 - 5.0, 183.31 - 3.5,
                         183.31 - 2.5, 183.31 - 1.5, 183.31 - 0.6,
                         183.31 + 0.6, 183.31 + 1.5, 183.31 + 2.5, 183.31 + 3.5,
                         183.31 + 5.0, 183.31 + 7.5, 183.31 + 12.5]) * 1e9

    _nedt = np.array([0.1] * 7 + [0.2] * 7 + [0.25] + [0.6] * 4 + [0.6] * 7)

    def __init__(self, stokes_dimension = 1):
        super().__init__(name = "hamp_passive",
                         f_grid = HampPassive.channels,
                         stokes_dimension = stokes_dimension)
        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position     = np.array([12500.0])

        self.sensor_response_f    = self.f_grid[:-11]
        self.sensor_response_pol  = self.f_grid[:-11]
        self.sensor_response_dlos = self.f_grid[:-11, np.newaxis]

        data = 15 * [1.0] + [0.5] * 22
        i    = list(range(15)) + 2 * list(range(15, 19)) + 2 * list(range(19, 26))
        j    = list(range(37))
        self.sensor_response = sp.sparse.coo_matrix((data, (i, j)))
        self.sensor_f_grid   = self.f_grid[:-11]


    @property
    def nedt(self):
        return HampPassive._nedt

################################################################################
# Liras cloud profiling radar (LCPR).
################################################################################

class LCPR(ActiveSensor):
    channels = np.array([94.0e9])

    def __init__(self,
                 name = "lcpr",
                 range_bins = np.arange(0.0, 20e3, 500.0),
                 stokes_dimension = 1):
        super().__init__(name,
                         f_grid = np.array([94e9]),
                         stokes_dimension = stokes_dimension,
                         range_bins = range_bins)
        self.nedt = 0.5 * np.ones(range_bins.size - 1)
        self.instrument_pol       = [1]
        self.instrument_pol_array = [[1]]
        self.extinction_scaling   = 1.0
        self.y_min = -30.0

################################################################################
# HAMP
################################################################################

class HampRadar(ActiveSensor):

    def __init__(self, stokes_dimension = 1):
        import crac.joint_flight
        path = crac.joint_flight.path

        ds = Dataset(os.path.join(path, "data", "input.nc"))
        z  = ds.variables["altitude"][0, :]

        range_bins = np.zeros(z.size + 1)
        range_bins[1:-1] = 0.5 * (z[1:] + z[:-1])
        range_bins[0]  = 2 * range_bins[1] - range_bins[2]
        range_bins[-1] = 2 * z[-1] - z[-2]
        ds.close()

        super().__init__(name = "hamp_radar",
                         f_grid = [35.564e9],
                         range_bins = range_bins,
                         stokes_dimension = stokes_dimension)

        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position      = np.array([12500.0])
        self.y_min = -30.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)

################################################################################
# RASTA
################################################################################

class RastaRadar(ActiveSensor):

    def __init__(self, stokes_dimension = 1):

        super().__init__(name = "rasta",
                         f_grid = [95e9],
                         stokes_dimension = stokes_dimension)

        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position      = np.array([12500.0])
        self.y_min = -16.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)


#
# ICI
#

ici = ICI(stokes_dimension = 1)
ici.sensor_line_of_sight = np.array([[135.0]])
ici.sensor_position = np.array([[600e3]])
ici.nedt = ICI.nedt

#
# MWI
#

mwi = MWI(channel_indices = list(range(7, 18)), stokes_dimension = 1)
mwi.sensor_line_of_sight = np.array([[135.0]])
mwi.sensor_position = np.array([[600e3]])

mwi_full = MWI(stokes_dimension = 1)
mwi_full.sensor_line_of_sight = np.array([[135.0]])
mwi_full.sensor_position = np.array([[600e3]])

#
# HAMP Passive
#

hamp_passive = HampPassive()

#
# LCPR
#

lcpr = LCPR(stokes_dimension = 1)
lcpr.sensor_line_of_sight = np.array([[135.0]])
lcpr.sensor_position = np.array([[600e3]])

#
# HAMP RADAR
#

hamp_radar   = HampRadar()
