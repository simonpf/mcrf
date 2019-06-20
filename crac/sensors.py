"""
The :code:`crac.sensors` module contains Python classes that
represent the sensors that are used within the study. These
are:
    - The Ice Cloud Imager (ICI)
    - The Microwave Imager (MWI)
    - The LIRAS Cloud Profiling Radar (LCPR)
    - The HAMP RADAR and radiometer
    - the ISMAR radiometer

Attributes:

    ici: An instance of the ICI class that includes all of
       the available ICI channels.

    ismar: Sensor instance representing the ISMAR demonstrator for ICI.

    hamp_radar: Sensor instance representing the HAMP radar.

    hamp_passive: Sensor representing the passive radiometers of HAMP

    mwi: An instance of the MWI class that includes only the channels
         with frquencies larger than 89 GHz. This is the sensor that
         has been used withing the ESA Wide Swath cloud profiling study.

    mwi_full: Instance of MWI class with all channels of the MWI sensor.

    lcpr: The LIRAS cloud profiling radar.

"""
import numpy as np
import scipy as sp
import os
from netCDF4 import Dataset
from parts.sensor import PassiveSensor, ActiveSensor, PassiveSensor
from parts.sensor.utils import sensor_properties

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
    center_frequencies = np.array([183.31, 243, 325.15, 448.0, 664.0]) * 1e9
    sidebands = [np.array([2.0, 3.4, 7.0]) * 1e9,
                 np.array([2.5]) * 1e9,
                 np.array([9.5, 3.5, 1.5]) * 1e9,
                 np.array([7.2, 3.0, 1.4]) * 1e9,
                 np.array([4.2]) * 1e9]
    nedt = np.array([0.8, 0.8, 0.8,       # 183 GHz
                     0.7 * np.sqrt(0.5),  # 243 GHz
                     1.2, 1.3, 1.5,       # 325 GHz
                     1.4, 1.6, 2.0,       # 448 GHz
                     1.6 * np.sqrt(0.5)]) # 664 GHz

    def __init__(self,
                 name = "ici",
                 band_indices = None,
                 stokes_dimension = 1):
        """
        This creates an instance of the ICI sensor to be used within a
        :code:`parts` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor used within the parts
                simulation.

            band_indicees(:code:`list`): Indices of the frequency bands to be used.

            stokes_dimension(:code:`int`): The stokes dimension to use for
                the retrievals.
        """
        if not band_indices is None:
            channel_indices = []
            i = 0
            for bi in band_indices:
                for o in ICI.sidebands[bi]:
                    channel_indices += [i]
                    i += 1

            center_frequencies = [ICI.center_frequencies[i] for i in band_indices]
            offsets = [ICI.sidebands[i] for i in band_indices]
            self.nedt = ICI.nedt[channel_indices]
        else:
            center_frequencies = ICI.center_frequencies
            offsets = ICI.sidebands
            self.nedt = ICI.nedt

        channels, sensor_response = sensor_properties(center_frequencies, offsets,
                                                      order = "negative")
        super().__init__(name, channels, stokes_dimension = stokes_dimension)
        self.sensor_line_of_sight = np.array([[132.0]])
        self.sensor_position = np.array([[600e3]])

        m = sensor_response.shape[0]

        self.sensor_response_f    = self.f_grid[:m]
        self.sensor_response_pol  = self.f_grid[:m]
        self.sensor_response_dlos = self.f_grid[:m, np.newaxis]
        self.sensor_response = sensor_response
        self.sensor_f_grid   = self.f_grid[:m]



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
    center_frequencies = np.array([18.7, 23.8, 31.4, 50.3, 52.61, 53.24, 53.75, 89.0,
                                   118.75, 165.5, 183.31]) * 1e9
    sidebands = [np.array([0.0])] * 8 + \
                [np.array([1.2, 1.4, 2.1, 3.2]) * 1e9,
                 np.array([0.75]) * 1e9,
                 np.array([2.0, 3.4, 4.9, 6.1, 7.0]) * 1e9]

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
                 band_indices = None,
                 stokes_dimension = 1):
        """
        This creates an instance of the MWI sensor to be used within a
        :code:`parts` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor used within the parts
                simulation.

            band_indicees(:code:`list`): Indices of the frequency bands to be used.

            stokes_dimension(:code:`int`): The stokes dimension to use for
                the retrievals.
        """
        if not band_indices is None:
            channel_indices = []
            i = 0
            for bi in band_indices:
                for o in MWI.sidebands[bi]:
                    channel_indices += [i]
                    i += 1

            center_frequencies = [MWI.center_frequencies[i] for i in band_indices]
            offsets = [MWI.sidebands[i] for i in band_indices]
            self.nedt = MWI.nedt[channel_indices]
        else:
            center_frequencies = MWI.center_frequencies
            offsets = MWI.sidebands
            self.nedt = MWI.nedt

        channels, sensor_response = sensor_properties(center_frequencies, offsets,
                                                      order = "positive")
        super().__init__(name, channels, stokes_dimension = stokes_dimension)
        self.sensor_line_of_sight = np.array([[132.0]])
        self.sensor_position = np.array([[600e3]])

        m = sensor_response.shape[0]

        self.sensor_response_f    = self.f_grid[:m]
        self.sensor_response_pol  = self.f_grid[:m]
        self.sensor_response_dlos = self.f_grid[:m, np.newaxis]
        self.sensor_response = sensor_response
        self.sensor_f_grid   = self.f_grid[:m]

################################################################################
# Hamp Passive
################################################################################

class HampPassive(PassiveSensor):

    center_frequencies = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40,
                                   50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00,
                                   90, 118.75, 183.31]) * 1e9
    sidebands = [np.array([0.0]) * 1e9] * 15 \
                + [np.array([1.4, 2.3, 4.2, 8.5]) * 1e9] \
                + [np.array([0.6,  1.5,  2.5, 3.5,  5.0,  7.5, 12.5]) * 1e9]
    channels, sensor_reponse = sensor_properties(center_frequencies,
                                                 sidebands,
                                                 order = "positive")

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
        self.sensor_response = HampPassive.sensor_response
        self.sensor_f_grid   = self.f_grid[:-11]

    @property
    def nedt(self):
        return 0.1 * np.ones(self.sensor_response_f.size)

################################################################################
# ISMAR
################################################################################


class Ismar(PassiveSensor):

    center_frequencies = np.array([118.75, 243.2, 325.15, 664.0]) * 1e9
    offsets = [np.array([1.1, 1.5, 2.1, 3.0, 5.0]) * 1e9,
               np.array([2.5]) * 1e9,
               np.array([1.5, 3.5, 9.5]) * 1e9,
               np.array([4.2]) * 1e9]

    channels, sensor_response = sensor_properties(center_frequencies,
                                                  offsets,
                                                  order = "positive")

    _nedt = np.array(10 * [2.0])

    def __init__(self, stokes_dimension = 1):
        super().__init__(name = "ismar",
                         f_grid = Ismar.channels,
                         stokes_dimension = stokes_dimension)
        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position     = np.array([9300.0])
        self.iy_unit             = "RJBT"

        self.sensor_response_f    = self.f_grid[:10]
        self.sensor_response_pol  = self.f_grid[:10]
        self.sensor_response_dlos = self.f_grid[:10, np.newaxis]

        self.sensor_response = Ismar.sensor_response
        self.sensor_f_grid   = self.f_grid[:10]


    @property
    def nedt(self):
        return 0.1 * np.ones(self.sensor_response_f.size)

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
        range_bins = np.linspace(0.0, 10e3, 51) + 100.0
        super().__init__(name = "hamp_radar",
                         f_grid = [35.564e9],
                         range_bins = range_bins,
                         stokes_dimension = stokes_dimension)

        self.sensor_line_of_sight = np.array([180.0])
        self.sensor_position      = np.array([12500.0])
        self.instrument_pol       = [1]
        self.instrument_pol_array = [[1]]
        self.extinction_scaling   = 1.0
        self.y_min = -30.0

    @property
    def nedt(self):
        return 0.5 * np.ones(self.range_bins.size - 1)

################################################################################
# RASTA
################################################################################

class RastaRadar(ActiveSensor):

    def __init__(self, stokes_dimension = 1):

        range_bins = np.linspace(0.0, 10e3, 51) + 100.0
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

#
# MWI
#

mwi = MWI(band_indices = [7, 8, 9, 10], stokes_dimension = 1)
mwi.sensor_line_of_sight = np.array([[132.0]])
mwi.sensor_position = np.array([[600e3]])

mwi_full = MWI(stokes_dimension = 1)
mwi_full.sensor_line_of_sight = np.array([[132.0]])
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

ismar = Ismar()
