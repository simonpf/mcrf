from parts.atmosphere            import Atmosphere1D
from parts.sensor                import ActiveSensor
from parts.atmosphere.surface    import Tessem
from parts.retrieval.a_priori    import DataProviderAPriori
from parts.atmosphere.absorption import O2, N2, H2O, RelativeHumidity
from parts.utils.data_providers  import NetCDFDataProvider
from joint_flight.hydrometeors   import ice, rain, cloud_water
from parts.scattering.solvers    import Disort
from parts.simulation            import ArtsSimulation
from joint_flight.apriori        import *
from parts.jacobian              import Log10, Atanh

class CloudRetrieval:

    def _setup_retrieval(self):

        ice, rain, cw = self.simulation.atmosphere.scatterers

        for q in self.hydrometeors:

            limit_low_1, limit_low_2 = q.limits_low

            md = q.moments[0]
            self.simulation.retrieval.add(md)
            md.transformation = q.transformations[0]
            md.limit_low      = q.limits_low[0]

            n0 = q.moments[1]
            self.simulation.retrieval.add(n0)
            n0.transformation = q.transformation[1]
            n0.limit_low      = q.linmits_low[1]

        h2o = self.simulation.atmosphere.absorbers[-1]
        self.simulation.retrieval.add(h2o)
        h2o.retrieval.unit      = RelativeHumidity()
        h2o.transformation      = Atanh()
        h2o.transformation.z_max = 1.1
        self.h2o = h2o

        settings = self.simulation.retrieval.settings
        settings["max_iter"] = 10
        settings["stop_dx"]  = 0.5
        settings["lm_ga_settings"] = np.array([100.0, 3.0, 2.0, 1e5, 1.0, 1.0])


    def __init__(self,
                 hydrometeors,
                 sensors,
                 data_provider):

        self.hydrometeors = hydrometeors
        absorbers  = [O2(), N2(), H2O()]
        scatterers = hydrometeors
        surface    = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors = sensors,
                                         scattering_solver = Disort())
        self.sensors = sensors

        self.data_provider            = data_provider
        self.simulation.data_provider = self.data_provider

        self._setup_retrieval()
        self._setup_a_priori()

        def radar_only(rr):
            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors \
                                       if h.radar_only]

        def only_first_moments(rr):
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [self.h2o]

        def all_quantities(rr):
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors \
                                       if h.retrieve_second_moment]
            rr.retrieval_quantities += [self.h2o]

        self.simulation.retrieval.callbacks = [("Radar only", radar_only),
                                               ("First moments", only_first_moments),
                                               ("All quantities", all_quantities)]


    def setup(self):
        self.simulation.setup()

    def run(self, i):
        return self.simulation.run(i)
