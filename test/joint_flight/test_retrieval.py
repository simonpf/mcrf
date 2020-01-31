import os
os.environ["JOINT_FLIGHT_PATH"] = "/home/simonpf/src/joint_flight"
os.environ["ARTS_DATA_PATH"] = "/home/simonpf/src/joint_flight/data"

import mcrf.joint_flight.setup

import numpy as np
import mcrf.liras
from   mcrf.psds import D14NDmIce
from   mcrf.retrieval        import CloudRetrieval
from   mcrf.sensors          import hamp_radar, hamp_passive, ismar
from   mcrf.joint_flight     import ice, snow, rain, cloud_water_a_priori, \
    h2o_a_priori, ObservationError, temperature_a_priori, psd_shapes_low, psd_shapes_high
from parts.retrieval.a_priori import SensorNoiseAPriori
from parts.utils.data_providers import NetCDFDataProvider

import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

#
# Load observations.
#

filename     = os.path.join(mcrf.joint_flight.path, "data", "combined", "input.nc")
data_provider = NetCDFDataProvider(filename)

ice_shape = "FluffyMcSnowPhase"
ice.scattering_data = "/home/simonpf/src/joint_flight/data/scattering/{}.xml".format(ice_shape)

if ice_shape in psd_shapes_low:
    alpha, log_beta = psd_shapes_high[ice_shape]
    ice.psd = D14NDmIce(alpha, np.exp(log_beta))

#
# Define hydrometeors and sensors.
#

hydrometeors = [ice, rain]
sensors      = [hamp_radar, hamp_passive, ismar]

#
# Add a priori providers.
#

data_provider.add(ice.a_priori[0])
data_provider.add(ice.a_priori[1])
data_provider.add(snow.a_priori[0])
data_provider.add(snow.a_priori[1])
data_provider.add(rain.a_priori[0])
data_provider.add(rain.a_priori[1])
data_provider.add(cloud_water_a_priori)
data_provider.add(h2o_a_priori)
#data_provider.add(temperature_a_priori)
data_provider.add(ObservationError(sensors))

#
# Run the retrieval.
#

retrieval = CloudRetrieval(hydrometeors, sensors, data_provider)
retrieval.setup()
retrieval.run(658)

def plot_misfit():
    ws = retrieval.simulation.workspace
    y = ws.y.value
    yf = ws.yf.value
    y_hamp = y[:59]
    yf_hamp = yf[:59]
    plt.plot(y_hamp)
    plt.plot(yf_hamp)

    plt.figure()
    y_hamp = y[59:]
    yf_hamp = yf[59:]
    plt.plot(y_hamp)
    plt.plot(yf_hamp)
    plt.figure()

def plot_masses():
    from joint_flight.utils import iwc, rwc
    ws = retrieval.simulation.workspace
    ws.x2artsAtmAndSurf()

    ice_n0 = ws.particle_bulkprop_field.value[0]
    ice_dm = ws.particle_bulkprop_field.value[1]
    rain_n0 = ws.particle_bulkprop_field.value[2]
    rain_dm = ws.particle_bulkprop_field.value[3]
    z = ws.z_field.value.ravel()

    md_ice = iwc(ice_n0, ice_dm)
    md_rain = rwc(rain_n0, rain_dm)

    plt.plot(md_ice.ravel(), z)
    plt.plot(md_rain.ravel(), z)
    plt.plot(md_ice.ravel() + md_rain.ravel(), z)
    plt.xscale("log")
    plt.xlim([1e-7, 1e-3])

    return (ice_n0, ice_dm, rain_n0, rain_dm)

def plot_jacobianz(channel = -3):
    ws = retrieval.simulation.workspace
    j = ws.jacobian.value
    rqs = retrieval.simulation.retrieval.retrieval_quantities

    z = ws.z_field.value.ravel()

    A  = rqs[0].transformation.transformations[1].A
    m, n = A.shape
    j_n0 = np.dot(j[channel, :m], A)
    plt.figure()
    plt.plot(j_n0, z)

    i_start = m
    plt.figure()
    A  = rqs[2].transformation.transformations[1].A
    m, n = A.shape
    j_n0_r = np.dot(j[channel, i_start : i_start + m], A)
    plt.plot(j_n0_r, z)

    plt.figure()
    i_start += m
    j_dm = j[channel, i_start : i_start + 61]
    plt.plot(j_dm, z)

    #plt.figure()
    #y_hamp = y[59:]
    #yf_hamp = yf[59:]
    #plt.plot(y_hamp)
    #plt.plot(yf_hamp)
    #plt.figure()


    r
