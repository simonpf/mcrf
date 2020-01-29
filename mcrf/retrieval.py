"""
mcrf.retrieval

Provides classes for performing forward simulations and cloud retrieval
calculations.
"""
import numpy as np
from parts.atmosphere import Atmosphere1D
from parts.sensor import ActiveSensor, PassiveSensor
from parts.atmosphere.surface import Tessem
from parts.retrieval.a_priori import DataProviderAPriori, PiecewiseLinear
from parts.retrieval import RetrievalRun
from parts.atmosphere.absorption import O2, N2, H2O, CloudWater, RelativeHumidity, VMR
from parts.atmosphere.catalogs import Aer, Perrin
from parts.utils.data_providers import NetCDFDataProvider
from parts.scattering.solvers import Disort, RT4
from parts.simulation import ArtsSimulation
from parts.jacobian import Log10, Atanh, Composition, Identity

################################################################################
# Cloud retrieval
################################################################################


class CloudRetrieval:
    """
    Class for performing cloud retrievals.

    Attributes:

        simulation(parts.ArtsSimulation): parts ArtsSimulation object that is used
            to perform retrieval caculations.

        h2o(parts.atmosphere.AtmosphericQuantity): The AtmosphericQuantity instance
            that represent water vapor in the ARTS simulation.

        cw(parts.atmosphere.AtmosphericQuantity): The AtmosphericQuantity instance
            that represent cloud liquid in the ARTS simulation.

        sensors(parts.sensor.Sensor): The sensors used in the retrieval.

        data_provider: The data provider used to perform the retrieval.
    """
    def _setup_retrieval(self):
        """
        Setup the parts simulation used to perform the retrieval.
        """

        for q in self.hydrometeors:

            limit_low_1, limit_low_2 = q.limits_low
            if hasattr(q, "limits_high"):
                limit_high_1, limit_high_2 = q.limits_high
            else:
                limit_high_1, limit_high_2 = np.inf, np.inf

            md = q.moments[0]
            self.simulation.retrieval.add(md)
            md.transformation = q.transformations[0]
            md.retrieval.limit_low = q.limits_low[0]
            md.retrieval.limit_high = limit_high_1

            n0 = q.moments[1]
            self.simulation.retrieval.add(n0)
            n0.transformation = q.transformations[1]
            n0.retrieval.limit_low = q.limits_low[1]
            n0.retrieval.limit_high = limit_high_2

        h2o = self.simulation.atmosphere.absorbers[-1]

        h2o_a = [p for p in self.data_provider.subproviders \
                 if getattr(p, "name", "") == "H2O"]
        if len(h2o_a) > 0:
            h2o_a = h2o_a[0]
            self.simulation.retrieval.add(h2o)
            atanh = Atanh(0.0, 1.1)
            h2o.transformation = h2o_a.transformation
            h2o.retrieval.unit = RelativeHumidity()
            self.h2o = h2o
        else:
            self.h2o = None

        if self.include_cloud_water:
            cw_a = [p for p in self.data_provider.subproviders \
                    if getattr(p, "name", "") == "cloud_water"][0]
            cw = self.simulation.atmosphere.absorbers[-2]
            self.simulation.retrieval.add(cw)
            pl = PiecewiseLinear(cw_a)
            cw.transformation = Composition(Log10(), pl)
            cw.retrieval.limit_high = -3
            self.cw = cw
        else:
            self.cw = None

        t_a = [p for p in self.data_provider.subproviders \
                 if getattr(p, "name", "") == "temperature"]
        if len(t_a) > 0:
            t = self.simulation.atmosphere.temperature
            self.temperature = t
            self.simulation.retrieval.add(self.temperature)
        else:
            self.temperature = None

    def __init__(self, hydrometeors, sensors, data_provider):

        cw_a = [p for p in data_provider.subproviders \
                if getattr(p, "name", "") == "cloud_water"]
        self.include_cloud_water = len(cw_a) > 0

        self.hydrometeors = hydrometeors
        absorbers = [
            O2(model="TRE05", from_catalog=False),
            N2(model="SelfContStandardType", from_catalog=False),
            H2O(model=["SelfContCKDMT320", "ForeignContCKDMT320"],
                from_catalog=True,
                lineshape="VP",
                normalization="VVH",
                cutoff=750e9)
        ]
        absorbers = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(model="ELL07", from_catalog=False))
        scatterers = hydrometeors
        surface = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=Disort())
        self.sensors = sensors

        self.data_provider = data_provider
        self.simulation.data_provider = self.data_provider

        self._setup_retrieval()

        self.radar_only = all(
            [isinstance(s, ActiveSensor) for s in self.sensors])

        def radar_only(rr):

            rr.settings["max_iter"] = 20
            rr.settings["stop_dx"] = 1e-6
            rr.settings["lm_ga_settings"] = np.array(
                [1000.0, 3.0, 2.0, 10e3, 1.0, 1.0])

            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]
            #rr.retrieval_quantities = [h.moments[1] for h in self.hydrometeors]

        def all_quantities(rr):

            rr.settings["max_iter"] = 20
            rr.settings["stop_dx"] = 1e-6
            rr.settings["lm_ga_settings"] = np.array(
                [1000.0, 3.0, 2.0, 10e3, 1.0, 1.0])

            #if all([isinstance(s, PassiveSensor) for s in rr.sensors]):
            #    rr.settings["lm_ga_settings"] = np.array(
            #        [0.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            #else:
            #    rr.settings["lm_ga_settings"] = np.array(
            #        [10.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]

            if not self.h2o is None:
                rr.retrieval_quantities += [self.h2o]
            if not self.cw is None:
                rr.retrieval_quantities += [self.cw]
            if not self.temperature is None:
                rr.retrieval_quantities += [self.temperature]

        if all([isinstance(s, ActiveSensor) for s in self.sensors]):
            self.simulation.retrieval.callbacks = [("Radar only", radar_only)]
        elif any([isinstance(s, ActiveSensor) for s in self.sensors]):
            self.simulation.retrieval.callbacks = [("All quantities",
                                                    all_quantities)]
        else:
            self.simulation.retrieval.callbacks = [("All quantities",
                                                    all_quantities)]

    def setup(self, verbosity=1):
        """
        Run parts setup of simulation instance. This function needs to be executed
        before the retrieval can be calculated.

        Arguments:

            verbosity: ARTS workspace verbosity. 0 for silent.
        """

        self.simulation.setup(verbosity=verbosity)

    def run(self, i):
        """
        Run retrieval with simulation argument i.

        Arguments:
            i: The simulation argument that is passed to the run method of the
               ArtsSimulation object.
        """
        self.index = i
        return self.simulation.run(i)


################################################################################
# Cloud simulation
################################################################################


class CloudSimulation:
    """
    Class for performing forward simulation on GEM model data.
    """
    def __init__(self,
                 hydrometeors,
                 sensors,
                 data_provider,
                 include_cloud_water=False):
        """
        Arguments:

            hydrometeors(list): List of the hydrometeors to use in the simulation.

            sensors(list): List of sensors for which to simulate observations.

            data_provider: Data provider object providing the simulation data.

            include_cloud_water(bool): Whether or not to include cloud water
        """
        self.include_cloud_water = include_cloud_water

        self.hydrometeors = hydrometeors
        absorbers = [
            O2(model="TRE05", from_catalog=False),
            N2(model="SelfContStandardType", from_catalog=False),
            H2O(model=["SelfContCKDMT320", "ForeignContCKDMT320"],
                lineshape="VP",
                normalization="VVH",
                cutoff=750e9)
        ]
        absorbers = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(model="ELL07", from_catalog=False))
        scatterers = hydrometeors
        surface = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=Disort())
        self.sensors = sensors

        self.data_provider = data_provider
        self.simulation.data_provider = self.data_provider

    def setup(self, verbosity=1):
        """
        Run setup method of ArtsSimulation.
        """
        self.simulation.setup(verbosity=verbosity)

    def run(self, *args, **kwargs):
        return self.simulation.run(*args, **kwargs)
