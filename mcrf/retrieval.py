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
from parts.atmosphere.catalogs import Aer
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
            self.simulation.retrieval.add(h2o)
            h2o_a = h2o_a[0]
            atanh = Atanh(0.0, 1.1)
            h2o.transformation = atanh
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
                lineshape="Voigt_Kuntz6",
                normalization="VVH",
                cutoff=750e9)
        ]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(from_catalog=False))
        scatterers = hydrometeors
        surface = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        atmosphere.catalog = Aer("h2o_lines.xml.gz")
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=RT4())
        self.sensors = sensors

        self.data_provider = data_provider
        self.simulation.data_provider = self.data_provider

        self._setup_retrieval()

        self.radar_only = all(
            [isinstance(s, ActiveSensor) for s in self.sensors])

        def radar_only(rr):
            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.settings["lm_ga_settings"] = np.array(
                [100.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.settings["stop_dx"] = 0.1
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]

        def all_quantities(rr):
            if all([isinstance(s, PassiveSensor) for s in rr.sensors]):
                rr.settings["lm_ga_settings"] = np.array(
                    [10.0, 3.0, 2.0, 1e5, 1.0, 1.0])
                rr.settings["max_iter"] = 10
                rr.settings["stop_dx"] = 0.1
            else:
                rr.settings["lm_ga_settings"] = np.array(
                    [10.0, 3.0, 2.0, 1e5, 1.0, 1.0])
                rr.settings["max_iter"] = 10
                rr.settings["stop_dx"] = 0.1
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
            self.simulation.retrieval.callbacks = [("Radar only", radar_only),
                                                   ("All quantities",
                                                    all_quantities)]
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


class HybridRetrieval(CloudRetrieval):
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
    def __init__(self, hydrometeors, sensors, data_provider):
        CloudRetrieval.__init__(self, hydrometeors, sensors, data_provider)

    def setup(self, verbosity=0):
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
        self.ensemble_size = 10

        #
        # Run radar only retrieval.
        #

        def radar_only(rr):
            rr.sensors = [s for s in rr.sensors if isinstance(s, ActiveSensor)]
            rr.settings["lm_ga_settings"] = np.array(
                [100.0, 3.0, 2.0, 1e5, 1.0, 1.0])
            rr.settings["max_iter"] = 10
            rr.retrieval_quantities = [h.moments[0] for h in self.hydrometeors]
            rr.retrieval_quantities += [
                h.moments[1] for h in self.hydrometeors
            ]

        self.simulation.retrieval.callbacks = [("Radar only", radar_only)]
        self.simulation.run(i)

        x0 = np.copy(self.simulation.workspace.x.value)

        self.x0 = x0
        covmat_sx = self.simulation.workspace.covmat_sx.value.to_dense()
        covmat_se = self.simulation.workspace.covmat_se.value.to_dense()
        K = self.simulation.workspace.jacobian.value
        self.covmat = np.linalg.inv(K.T @ covmat_se @ K +
                                    np.linalg.inv(covmat_sx))

        rqs = [h.moments[0] for h in self.hydrometeors]
        rqs += [h.moments[1] for h in self.hydrometeors]
        if not self.h2o is None:
            rqs += [self.h2o]
        if not self.cw is None:
            rqs += [self.cw]
        self.rqs = rqs

        def apply_limits(rq, x):
            if not rq.retrieval.limit_high == None:
                x = np.minimum(x, rq.retrieval.limit_high)
            if not rq.retrieval.limit_low == None:
                x = np.maximum(x, rq.retrieval.limit_low)
            return x

        masks = []
        for rq in rqs:
            mask_name = "get_" + rq.name + "_mask"
            try:
                f = getattr(self.data_provider, mask_name)
                masks += [f(self.index)]
            except:
                xa_name = "get_" + rq.name + "_xa"
                f = getattr(self.data_provider, xa_name)
                xa = f(self.index)
                masks += [np.ones(xa.size)]
        self.mask = np.concatenate(masks)

        simulation = self.simulation
        pr = simulation.retrieval.results[-1]
        retrieval = RetrievalRun("Hybrid",
                                 simulation,
                                 simulation.retrieval._y,
                                 simulation.retrieval.settings,
                                 simulation.retrieval.sensor_indices,
                                 rqs,
                                 previous_run=pr)
        retrieval.setup_a_priori(i)

        for rq in rqs:
            print(rq.name, rq.transformation)
            retrieval.simulation.jacobian.add(rq)

        def simulate_passive(simulation):
            simulation.workspace.jacobian_do = 0
            sensors = [s for s in self.sensors if isinstance(s, PassiveSensor)]
            simulation._run_forward_simulation(sensors)
            return np.copy(simulation.workspace.y.value)

        def simulate_active(simulation):
            simulation.workspace.jacobian_do = 1
            sensors = [s for s in self.sensors if isinstance(s, ActiveSensor)]
            simulation._run_forward_simulation(sensors)
            simulation.workspace.jacobianAdjustAndTransform()
            return (np.copy(simulation.workspace.y.value),
                    np.copy(simulation.workspace.jacobian.value))

        m_p = sum([
            s.y_vector_length for s in self.sensors
            if isinstance(s, PassiveSensor)
        ])
        m_a = sum([
            s.y_vector_length for s in self.sensors
            if isinstance(s, ActiveSensor)
        ])
        n = simulation.workspace.xa.value.size
        self.xs = np.zeros((n, self.ensemble_size))
        self.ys_p = np.zeros((m_p, self.ensemble_size))
        self.ys_a = np.zeros((m_a, self.ensemble_size))
        self.K_a = np.zeros((m_a, n, self.ensemble_size))

        #
        # Initialize ensemble.
        #
        simulation.workspace.jacobian_do = 0
        for i in range(self.ensemble_size):
            print("simulating member {}.".format(i))
            pr.x = np.random.multivariate_normal(x0, self.covmat)

            x_parts = []
            for rq in rqs:
                x = pr.get_result(rq)
                if x is None:
                    xa = retrieval.get_xa(rq, interpolate=False)
                    f_cov = getattr(self.data_provider,
                                    "get_" + rq.name + "_covariance")
                    cov = f_cov(self.index)
                    x = np.random.multivariate_normal(xa, cov)

                x = apply_limits(rq, x)
                x_parts += [x]
                rq.set_from_x(simulation.workspace, x)

            self.xs[:, i] = np.concatenate(x_parts)
            self.ys_p[:, i] = simulate_passive(self.simulation)
            y, K = simulate_active(self.simulation)
            self.ys_a[:, i] = y
            self.K_a[:, :, i] = K

        covmat_se = self.data_provider.get_observation_error_covariance(
            self.index).todense()
        covmat_se_inv = np.linalg.inv(covmat_se)
        x_mean = np.mean(self.xs, axis=-1)
        y_p_mean = np.mean(self.ys_p, axis=-1)
        y_a_mean = np.mean(self.ys_a, axis=-1)

        self.ys = np.zeros((m_a + m_p, self.ensemble_size))
        for i in range(self.ensemble_size):
            self.ys[:m_a, i] = self.ys_a[:, i]
            self.ys[m_a:, i] = self.ys_p[:, i]
        y_mean = np.mean(self.ys, axis=-1)

        N = self.ensemble_size

        cov_yy_p = 1.0 / (N - 1) * (self.ys_p - y_p_mean.reshape(-1, 1)) @ (
            self.ys_p - y_p_mean.reshape(-1, 1)).T
        cov_yy_a = 1.0 / (N - 1) * (self.ys_a - y_a_mean.reshape(-1, 1)) @ (
            self.ys_a - y_a_mean.reshape(-1, 1)).T
        cov_xy = 1.0 / (N - 1) * (self.xs - x_mean.reshape(-1, 1)) @ (
            self.ys - y_mean.reshape(-1, 1)).T
        cov_yy = 1.0 / (N - 1) * (self.ys - y_mean.reshape(-1, 1)) @ (
            self.ys - y_mean.reshape(-1, 1)).T

        dys = retrieval.y.reshape(-1, 1) - self.ys

        dxs = np.linalg.solve(cov_yy + covmat_se, dys)
        dxs = self.mask.reshape(-1, 1) * cov_xy @ dxs

        self.xs += dxs

        self.debug = {"xs": [], "ys": []}

        def ensemble_step():
            for i in range(self.ensemble_size):
                success = False
                while not success:
                    try:
                        print("simulating member {}.".format(i))
                        retrieval.x = self.xs[:, i]
                        x_parts = []
                        for rq in rqs:
                            x = retrieval.get_result(rq)
                            x = apply_limits(rq, x)
                            x_parts += [x]
                            rq.set_from_x(simulation.workspace, x)

                        self.xs[:, i] = np.concatenate(x_parts)
                        self.ys_p[:, i] = simulate_passive(self.simulation)
                        y, K = simulate_active(self.simulation)
                        self.ys_a[:, i] = y
                        self.K_a[:, :, i] = K

                        success = True
                    except Exception as e:
                        print(e)
                        success = False
                        coeffs = np.random.uniform(0,
                                                   1,
                                                   size=self.ensemble_size - 1)
                        coeffs /= coeffs.sum()
                        x_new = np.zeros(self.xs.shape[0])
                        for j in range(self.ensemble_size):
                            if j < i:
                                x_new += coeffs[j] * self.xs[:, j]
                            if j > i:
                                x_new += coeffs[j - 1] * self.xs[:, j]
                        print("new x:", x_new)
                        self.xs[:, i] = x_new

            x_mean = np.mean(self.xs, axis=-1)
            y_p_mean = np.mean(self.ys_p, axis=-1)
            y_a_mean = np.mean(self.ys_a, axis=-1)

            x_mean = np.mean(self.xs, axis=-1)
            y_p_mean = np.mean(self.ys_p, axis=-1)
            y_a_mean = np.mean(self.ys_a, axis=-1)

            self.ys = np.zeros((m_a + m_p, self.ensemble_size))
            for i in range(self.ensemble_size):
                self.ys[:m_a, i] = self.ys_a[:, i]
                self.ys[m_a:, i] = self.ys_p[:, i]

            self.ys += np.random.multivariate_normal(np.zeros(m_a + m_p),
                                                     covmat_se,
                                                     size=self.ensemble_size)
            y_mean = np.mean(self.ys, axis=-1)

            N = self.ensemble_size

            cov_xy = 1.0 / (N - 1) * (self.xs - x_mean.reshape(-1, 1)) @ (
                self.ys - y_mean.reshape(-1, 1)).T
            cov_yy = 1.0 / (N - 1) * (self.ys - y_mean.reshape(-1, 1)) @ (
                self.ys - y_mean.reshape(-1, 1)).T
            dys = retrieval.y.reshape(-1, 1) - self.ys
            dxs = np.linalg.solve(cov_yy + covmat_se, dys)
            dxs = self.mask.reshape(-1, 1) * cov_xy @ dxs

            self.xs += dxs

        for i in range(10):
            print("Step {}:".format(i))
            dys = self.ys - retrieval.y.reshape(-1, 1)
            for j in range(self.ensemble_size):
                print("\tMember {0}: {1}".format(
                    j, np.sum(dys[:, j] * covmat_se @ dys[:, j])))

            ensemble_step()

        #self.xs = xs
        #self.ys = ys


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
        absorbers = [O2(), N2(), H2O()]
        if self.include_cloud_water:
            absorbers.insert(2, CloudWater(model="Ell07"))
        scatterers = hydrometeors
        surface = Tessem()
        atmosphere = Atmosphere1D(absorbers, scatterers, surface)
        self.simulation = ArtsSimulation(atmosphere,
                                         sensors=sensors,
                                         scattering_solver=RT4())
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
