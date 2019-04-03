import numpy as np

from parts.atmosphere            import Atmosphere1D
from parts.sensor                import ActiveSensor, PassiveSensor
from parts.atmosphere.surface    import Tessem
from parts.retrieval.a_priori    import DataProviderAPriori
from parts.atmosphere.absorption import O2, N2, H2O, CloudWater, RelativeHumidity
from parts.utils.data_providers  import NetCDFDataProvider
from parts.scattering.solvers    import Disort
from parts.simulation            import ArtsSimulation
from parts.jacobian              import Log10, Atanh

class CloudRetrieval:

    def _setup_retrieval(self):

        for q in self.hydrometeors:

            limit_low_1, limit_low_2 = q.limits_low

            md = q.moments[0]
            self.simulation.retrieval.add(md)
            md.transformation = q.transformations[0]
            md.limit_low      = q.limits_low[0]

            n0 = q.moments[1]
            self.simulation.retrieval.add(n0)
            n0.transformation = q.transformations[1]
            n0.limit_low      = q.limits_low[1]

        h2o = self.simulation.atmosphere.absorbers[-1]
        self.simulation.retrieval.add(h2o)
        h2o.retrieval.unit      = RelativeHumidity()
        h2o.transformation      = Atanh()
        h2o.transformation.z_min = 0.0
        h2o.transformation.z_max = 1.05
        self.h2o = h2o

        if self.include_cloud_water:
            cw = self.simulation.atmosphere.absorbers[-2]
            self.simulation.retrieval.add(cw)
            cw.transformation = Log10()
            cw.retrieval.limit_high = -3
            self.cw = cw

        #t = self.simulation.atmosphere.temperature
        #self.t = t
        #self.simulation.retrieval.add(t)

        settings = self.simulation.retrieval.settings
        settings["max_iter"] = 10
        settings["stop_dx"]  = 0.1
        settings["lm_ga_settings"] = np.array([100.0, 3.0, 2.0, 1e5, 1.0, 0.0])


    def __init__(self,
                 hydrometeors,
                 sensors,
                 data_provider,
                 include_cloud_water = True):

        self.include_cloud_water = include_cloud_water

        self.hydrometeors = hydrometeors
        absorbers  = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater())
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

        def radar_only(rr):
            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors \
                                       if h.radar_only]

        def only_second_moments(rr):
            #rr.sensors = [s for s in rr.sensors if isinstance(s, PassiveSensor)]
            rr.retrieval_quantities = [h.moments[1] for h in self.hydrometeors \
                                       if h.retrieve_second_moment]
            rr.retrieval_quantities += [self.cw]
            rr.retrieval_quantities += [self.h2o]

        def passive_only(rr):
            rr.sensors = [s for s in rr.sensors if isinstance(s, PassiveSensor)]
            rr.retrieval_quantities = [h.moments[1] for h in self.hydrometeors \
                                       if h.retrieve_second_moment]
            rr.retrieval_quantities += [self.cw]
            rr.retrieval_quantities += [self.h2o]

        def only_first_moments(rr):
            rr.settings["lm_ga_settings"] = np.array([0.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [self.cw]
            rr.retrieval_quantities += [self.h2o]

            #rr.retrieval_quantities += [self.t]

        def all_quantities(rr):
            rr.settings["lm_ga_settings"] = np.array([0.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [h.moments[1] for h in self.hydrometeors \
                                       if h.retrieve_second_moment]
            rr.retrieval_quantities += [self.cw]
            rr.retrieval_quantities += [self.h2o]

        self.simulation.retrieval.callbacks = [("Radar only", radar_only),
                                               ("First moments", only_first_moments),
                                               ("All quantities", all_quantities)]
        self.simulation.retrieval.callbacks = [("Second moments", only_second_moments),
                                               ("First moments", only_first_moments),
                                               ("All quantities", all_quantities)]
        self.simulation.retrieval.callbacks = [("First moments", only_first_moments),
                                               ("Passive only ", passive_only),
                                               ("All quantities", all_quantities)]
        #self.simulation.retrieval.callbacks = [("Second moments", only_second_moments),
        #                                       ("First moments", only_first_moments)]
        #                                       #("All quantities", all_quantities)]
        #self.simulation.retrieval.callbacks = [("First moments", only_first_moments)]
        #self.simulation.retrieval.callbacks = [("First moments", only_first_moments),
        #                                       ("All quantities", all_quantities)]
        #self.simulation.retrieval.callbacks = [("All quantities", all_quantities)]



    def setup(self, verbosity = 1):
        self.simulation.setup(verbosity = verbosity)

    def run(self, i):
        self.index = i
        return self.simulation.run(i)

    def plot_results(self, axs = None):
        import matplotlib.pyplot as plt

        if axs == None:
            f, axs = plt.subplots(1, 3, figsize = (8, 8))

        try:
            results = self.simulation.retrieval.results
        except:
            raise Exception("Retrieval must be run before the results can be plotted.")

        z = self.data_provider.get_altitude(self.index) / 1e3

        names = [q.name for q in self.simulation.retrieval.retrieval_quantities]

        colors = {"ice_md" : "royalblue",
                  "ice_dm" : "royalblue",
                  "ice_n0" : "royalblue",
                  "snow_md" : "darkcyan",
                  "snow_dm" : "darkcyan",
                  "snow_n0" : "darkcyan",
                  "liquid_md" : "darkorchid",
                  "liquid_n0" : "darkorchid",
                  "rain_md" : "firebrick",
                  "rain_dm" : "firebrick",
                  "rain_n0" : "firebrick",
                  "H2O" : "lightsalmon"}


        #
        # First moments
        #

        ax = axs[0]

        for h in self.hydrometeors:
            rq = h.moments[0]

            try:
                f_get = "get_" + rq.name
                f = getattr(self.data_provider, f_get)
                args   = self.simulation.args
                kwargs = self.simulation.kwargs
                truth = f(*args, **kwargs)
                ax.plot(truth, z, c = colors[rq.name], alpha = 0.5)
            except:
                pass

            x = results[-1].get_result(rq, interpolate = True)
            if x is None:
                x = results[-1].get_xa(rq, interpolate = True)
            x = rq.transformation.invert(x)
            ax.plot(x, z, c = colors[rq.name])


        rq = self.cw
        try:
            f_get = "get_cloud_water"
            f = getattr(self.data_provider, f_get)
            args   = self.simulation.args
            kwargs = self.simulation.kwargs
            truth = f(*args, **kwargs)
            ax.plot(truth, z, c = colors[rq.name], alpha = 0.5)
        except:
            pass

        x = results[-1].get_result(rq, interpolate = True)
        if x is None:
            x = results[-1].get_xa(rq, interpolate = True)
        x = rq.transformation.invert(x)
        ax.plot(x, z, c = colors["liquid_md"])

        ax.set_xscale("log")
        ax.set_xlim([1e-6, 1e-3])
        ax.set_ylim([0, 20])
        ax.set_xlabel("Mass density [$kg / m^{3}$]")
        ax.set_ylabel("Altitude [km]")

        #
        # Second moments
        #

        ax = axs[1]

        for h in self.hydrometeors:
            rq = h.moments[1]

            try:
                f_get = "get_" + rq.name
                f = getattr(self.data_provider, f_get)
                args   = self.simulation.args
                kwargs = self.simulation.kwargs
                truth = f(*args, **kwargs)
                ax.plot(truth, z, c = colors[rq.name], alpha = 0.5)
            except:
                pass

            x = results[-1].get_result(rq, interpolate = True)
            if x is None:
                x = results[-1].get_xa(rq, interpolate = True)
            x = rq.transformation.invert(x)
            ax.plot(x, z, c = colors[rq.name])

        ax.set_xscale("log")
        ax.set_xlim([1e4, 1e13])
        ax.set_ylim([0, 20])
        ax.set_xlabel("$N_0^*$ [$m^{-4}$]")

        #
        # Second moments
        #

        ax = axs[2]

        rq = self.h2o

        try:
            f_get = "get_relative_humidity"
            f = getattr(self.data_provider, f_get)
            args   = self.simulation.args
            kwargs = self.simulation.kwargs
            truth = f(*args, **kwargs)
            ax.plot(100 * truth, z, c = colors[rq.name], alpha = 0.5)
        except:
            pass

        x = results[-1].get_result(rq, interpolate = True)
        if x is None:
            x = results[-1].get_xa(rq, interpolate = True)
        x = rq.transformation.invert(x)
        ax.plot(100 * x, z, c = colors[rq.name])

        ax.set_xlim([0, 100])
        ax.set_ylim([0, 20])
        ax.set_xlabel("RH [%]")

    def plot_fit(self, axs = None):
        import matplotlib.pyplot as plt

        try:
            results = self.simulation.retrieval.results
        except:
            raise Exception("Retrieval must be run before the results can be plotted.")

        r = results[-1]
        n_sensors = len(r.sensors)
        if axs == None:
            f, axs = plt.subplots(1, n_sensors, figsize = (n_sensors * 4, 8))

        ai = 0

        for s in r.sensors:
            ax = axs[ai]
            if isinstance(s, ActiveSensor):
                z = s.range_bins
                z = 0.5 * (z[1:] + z[:-1])

                i, j = r.sensor_indices[s.name]
                y = np.copy(r.y[i : j])
                yf = np.copy(r.yf[i : j])

                ax.plot(yf, z, label = "fit")
                ax.plot(y, z, label = "observation")
                ai += 1

        for s in r.sensors:
            ax = axs[ai]
            if isinstance(s, PassiveSensor):
                i, j = r.sensor_indices[s.name]
                y = np.copy(r.y[i : j])
                yf = np.copy(r.yf[i : j])

                ax.plot(yf, label = "fit")
                ax.plot(y, label = "observation")
                ai += 1

def plot_fit(self, axs = None):
    import matplotlib.pyplot as plt

    try:
        results = self.simulation.retrieval.results
    except:
        raise Exception("Retrieval must be run before the results can be plotted.")

    r = results[-1]
    n_sensors = len(r.sensors)
    if axs == None:
        f, axs = plt.subplots(1, n_sensors, figsize = (n_sensors * 4, 8))

    ai = 0

    for s in r.sensors:
        ax = axs[ai]
        if isinstance(s, ActiveSensor):
            z = s.range_bins
            z = 0.5 * (z[1:] + z[:-1])

            i, j = r.sensor_indices[s.name]
            y = np.copy(r.y[i : j])
            yf = np.copy(r.yf[i : j])

            ax.plot(yf, z, label = "fit")
            ax.plot(y, z, label = "observation")
            ai += 1

    for s in r.sensors:
        ax = axs[ai]
        if isinstance(s, PassiveSensor):
            i, j = r.sensor_indices[s.name]
            y = np.copy(r.y[i : j])
            yf = np.copy(r.yf[i : j])

            ax.plot(yf, label = "fit")
            ax.plot(y, label = "observation")
            ai += 1

class CloudSimulation:

    def __init__(self,
                 hydrometeors,
                 sensors,
                 data_provider,
                 include_cloud_water = True):

        self.include_cloud_water = include_cloud_water

        self.hydrometeors = hydrometeors
        absorbers  = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater())
        scatterers = hydrometeors
        surface    = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors = sensors,
                                         scattering_solver = Disort())
        self.sensors = sensors

        self.data_provider            = data_provider
        self.simulation.data_provider = self.data_provider

    def setup(self, verbosity = 1):
        self.simulation.setup(verbosity = verbosity)

    def run(self, *args, **kwargs):
        return self.simulation.run(*args, **kwargs)
