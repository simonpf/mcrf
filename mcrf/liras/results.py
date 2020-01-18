"""LIRAS retrieval results

This module contains commodity functions that aid the handling of retrieval
results.
"""
import os
import glob
import numpy as np

from mcrf.liras import liras_path
from mcrf.psds import D14NDmIce, D14NDmSnow, D14NDmLiquid
from mcrf.liras.model_data import ModelDataProvider
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors   import LogNorm, Normalize
from matplotlib.cm import magma

def get_results(scene = "a",
                config = "",
                type_suffix = "",
                variables = ["ice_dm", "ice_n0"]):
    """
    This function return a dictionary of retrieval results
    for all particle types that the retrieval has been
    performed for.

    Args:

        scene(str): The scene name (a or b)

        config: Which retrieval configuration ("ice", "snow", "ice_snow", ...)

        type_suffix = Type of the retrieval ("", "po", "ro")

        variables: List of variables to extract from file.

    Returns:

        A dict of dict holding the extracted variables for all the
        particle types found in $LIRAS_PATH/data
    """
    path = os.path.join(liras_path, "data")
    results = {}
    if config == "":
        pattern = os.path.join(path,
                               "retrieval_" + scene + "_" + "*"
                               + type_suffix + ".nc")
    else:
        pattern = os.path.join(path,
                               "retrieval_" + config + "_" + scene + "_" + "*"
                               + type_suffix + ".nc")
    files = glob.glob(pattern)
    if type_suffix == "":
        files = [f for f in files if not f[-5:-3] in ["ro", "po"]]
    else:
        files = [f for f in files if f.split("_")[-1] == type_suffix + ".nc"]

    for f in files:
        splits = os.path.basename(f).split("_")

        i = splits.index(scene)
        if type_suffix == "":
            sl = slice(i + 1, None)
        else:
            sl = slice(i + 1 , -1)

        if config == "":
            habit = "_".join(splits[sl]).split(".")[0]
        else:
            habit = "_".join(splits[sl]).split(".")[0]
        results[habit] = {}
        file = Dataset(f, mode = "r")
        for v in variables:
            try:
                k = list(file.groups.keys())[-1]
                results[habit][v] = file.groups[k][v][:]
            except:
                results[habit][v] = None
    return results

def get_reference_data(scene = "a",
                       i_start = 3000,
                       i_end = 3800):
    """
    Get reference data from cloud scene.

    Arguments:

        scene: From which scene to take the values ("a", "b")

        i_start: Start index of the scene

        i_end: End index of the scene
    """
    n       = i_end - i_start

    ice_psd    = D14NDmIce()
    snow_psd   = D14NDmSnow()
    liquid_psd = D14NDmLiquid()
    data_provider = ModelDataProvider(99,
                                      scene = scene.upper(),
                                      ice_psd = ice_psd,
                                      snow_psd = snow_psd,
                                      hail_psd = snow_psd,
                                      graupel_psd = snow_psd,
                                      liquid_psd = liquid_psd)
    z = data_provider.get_altitude(i_start)

    iwc, swc, gwc, hwc, lwc, rwc = np.zeros((6, n, z.size))
    iwc_nd, swc_nd, gwc_nd, hwc_nd, lwc_nd, rwc_nd = np.zeros((6, n, z.size))
    iwc_dm, swc_dm, gwc_dm, hwc_dm, lwc_dm, rwc_dm = np.zeros((6, n, z.size))
    iwc_n0, swc_n0, gwc_n0, hwc_n0, lwc_n0, rwc_n0 = np.zeros((6, n, z.size))
    lats, lons = np.zeros((2, n))
    h2o = np.zeros((n, z.size))
    temperature = np.zeros((n, z.size))

    for i in range(i_start, i_end):
        j = i - i_start
        iwc[j, :] = data_provider.get_gem_ice_mass_density(i)
        swc[j, :] = data_provider.get_gem_snow_mass_density(i)
        hwc[j, :] = data_provider.get_gem_hail_mass_density(i)
        gwc[j, :] = data_provider.get_gem_graupel_mass_density(i)
        rwc[j, :] = data_provider.get_gem_rain_mass_density(i)
        lwc[j, :] = data_provider.get_gem_liquid_mass_density(i)
        iwc_nd[j, :] = data_provider.get_gem_ice_number_density(i)
        swc_nd[j, :] = data_provider.get_gem_snow_number_density(i)
        hwc_nd[j, :] = data_provider.get_gem_hail_number_density(i)
        gwc_nd[j, :] = data_provider.get_gem_graupel_number_density(i)
        rwc_nd[j, :] = data_provider.get_gem_rain_number_density(i)
        lwc_nd[j, :] = data_provider.get_gem_liquid_number_density(i)
        iwc_dm[j, :] = data_provider.get_ice_dm(i)
        swc_dm[j, :] = data_provider.get_snow_dm(i)
        hwc_dm[j, :] = data_provider.get_hail_dm(i)
        gwc_dm[j, :] = data_provider.get_graupel_dm(i)
        rwc_dm[j, :] = data_provider.get_rain_dm(i)
        lwc_dm[j, :] = data_provider.get_cloud_water_dm(i)
        iwc_n0[j, :] = data_provider.get_ice_n0(i)
        swc_n0[j, :] = data_provider.get_snow_n0(i)
        hwc_n0[j, :] = data_provider.get_hail_n0(i)
        gwc_n0[j, :] = data_provider.get_graupel_n0(i)
        rwc_n0[j, :] = data_provider.get_rain_n0(i)
        lwc_n0[j, :] = data_provider.get_cloud_water_n0(i)
        h2o[j, :] = data_provider.get_relative_humidity(i)
        lats[j] = data_provider.get_latitude(i)
        lons[j] = data_provider.get_longitude(i)
        temperature[j, :] = data_provider.get_temperature(i)

    return {
        "iwc" : iwc,
        "iwc_nd" : iwc_nd,
        "iwc_dm" : iwc_dm,
        "iwc_n0" : iwc_n0,
        "swc" : swc,
        "swc_nd" : swc_nd,
        "swc_dm" : swc_dm,
        "swc_n0" : swc_n0,
        "hwc" : hwc,
        "hwc_nd" : hwc_nd,
        "hwc_dm" : hwc_dm,
        "hwc_n0" : hwc_n0,
        "gwc" : gwc,
        "gwc_nd" : gwc_nd,
        "gwc_dm" : gwc_dm,
        "gwc_n0" : gwc_n0,
        "rwc" : rwc,
        "rwc_nd" : rwc_nd,
        "rwc_dm" : rwc_dm,
        "rwc_n0" : rwc_n0,
        "lwc" : lwc,
        "lwc_nd" : lwc_nd,
        "lwc_dm" : lwc_dm,
        "lwc_n0" : lwc_n0,
        "h2o" : h2o,
        "lat" : lats,
        "lon" : lons,
        "z" : z,
        "temperature" : temperature
    }

def plot_results(lats, z, qs, name, norm,
                 costs = None,
                 cmap = magma,
                 include_columns = False,
                 column_label = None,
                 titles = None):

    n_plots = len(qs)

    if not costs is None:
        height_ratios = [1.0] + [1.0] * n_plots + [0.1]
        n_panels = n_plots + 2
    else:
        height_ratios = [1.0] * n_plots + [0.1]
        n_panels = n_plots + 1

    if include_columns:
        n_panels += 1
        height_ratios.insert(1, 1.0)

    f = plt.gcf()
    if f is None:
        f  = plt.figure(figsize = (10, 15))

    gs = GridSpec(n_panels, 1, height_ratios = height_ratios)
    #cmap.set_bad("grey", 1.0)

    x = lats
    y = z / 1e3
    i = 0

    #
    # OEM cost
    #
    prefix = ["({})".format(l) for l in ["a", "b", "c", "d", "e", "f", "g", "h"]]

    if not costs is None:
        ax = plt.subplot(gs[0])
        inds = []
        for i, c in enumerate(costs):
            inds += [c > 10.0]
            if not titles is None:
                ax.plot(x, c, c = "C{0}".format(i), label = titles[i + 1])
            ax.set_title(prefix[0] + " Final OEM cost", fontsize = 14, loc = "left")

        #ax.axhline(y = 10, c = "k")
        ax.legend(loc = "center", ncol = 3, fontsize = 14, bbox_to_anchor = [0.5, 1.2])

        ax.set_ylabel(r"$\chi^2_y$", fontsize = 14)
        ax.set_xticks([])
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([1e-3, 1e3])
        ax.set_yscale("log")
        i = 1

    #
    # Integrated quantities
    #

    if include_columns:
        ax = plt.subplot(gs[i])
        qr = np.trapz(qs[0], x = z)
        #ax.plot(x, qr, label = "Reference", c = "k", ls = "--")

        for j, q in enumerate(qs[1:]):
            q = np.copy(q)
            inds = np.broadcast_to(np.reshape(z > 20e3, (1, -1)), q.shape)
            q[inds] = 0.0
            qi = np.trapz(q, x = z)
            #ax.plot(x, qi, label = titles[j + 1])
            ax.plot(x, 10.0 * np.log10(qi / qr), label = titles[j + 1], zorder = 2)
        ax.set_ylabel(column_label, fontsize = 14)
        ax.set_title(prefix[i] + " Column-integrated results",
                     fontsize = 14, loc = "left")
        ax.set_xticks([])
        ax.set_ylim([-5, 5])
        #ax.set_yscale("log")
        ax.set_xlim([x[0], x[-1]])
        i = i + 1


    for j in range(n_plots):
        ax = plt.subplot(gs[i + j])
        q = qs[j]
        img = ax.pcolormesh(x, y, q.T, norm = norm, cmap = cmap)
        ax.set_ylim([0, 20])

        ax.set_ylabel("Altitude [km]", fontsize = 14)
        if j < n_plots - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Latitude [$^\circ$]", fontsize = 14)
        if not titles is None:
            ax.set_title(prefix[j + i] + " " + titles[j], loc = "left", fontsize = 14)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

    #
    # Colorbar
    #

    ax = plt.subplot(gs[-1])
    cbar = plt.colorbar(img, cax = ax, orientation = "horizontal",
                 label = name,
                extend = "both")
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label(name, fontsize = 14)

    plt.tight_layout()
    return f

def plot_results_2(lats, z, qs, name, norm, costs = None, cmap = magma, titles = None):

    n_plots = len(qs)

    if not costs is None:
        height_ratios = [0.5] + [1.0] * (n_plots // 2) + [0.1]
        n_panels = (n_plots // 2) + 2
    else:
        height_ratios = [1.0] * (n_plots // 2) + [0.1]
        n_panels = (n_plots // 2) + 1

    f  = plt.figure(figsize = (10, 10))
    gs = GridSpec(n_panels, 2, height_ratios = height_ratios)
    cmap.set_bad("grey", 1.0)

    x = lats
    y = z / 1e3
    i = 0

    #
    # OEM cost
    #

    if not costs is None:
        ax = plt.subplot(gs[0])
        inds = []
        for i, c in enumerate(costs):
            inds += [c > 10.0]
            if not titles is None:
                ax.plot(x, c, c = "C{0}".format(i), label = titles[i])

        ax.axhline(y = 10, c = "k")
        ax.legend(loc = "upper left", ncol = 3)

        ax.set_ylim([1e-1, 1e3])
        ax.set_ylabel("$\chi^2_y$ []")
        ax.set_xticks([])
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([-10, 20])
        i = 1

    for j in range(n_plots):
        ax = plt.subplot(gs[(i + j) // 2, (i + j) % 2])
        q = qs[j]
        img = ax.pcolormesh(x, y, q.T, norm = norm, cmap = cmap)
        ax.set_ylim([0, 20])

        ax.set_ylabel("Altitude [$km$]")
        if j < n_plots - 1:
            ax.set_xticks([])
        if not titles is None:
            ax.set_title(titles[j], loc = "left")
            ax.set_xlabel("Latitude [$^\circ$]")

    #
    # Colorbar
    #

    ax = plt.subplot(gs[-1, :])
    plt.colorbar(img, cax = ax, orientation = "horizontal",
                 label = name,
                extend = "both")

    plt.tight_layout()
    return f

def mfe_by_mass(q, q_r, bins = np.logspace(-6, -2, 21)):
    n = len(bins) - 1
    es = np.zeros(n)
    dqf = np.abs(q - q_r) / q_r
    for i in range(n):
        l = bins[i]
        r = bins[i + 1]
        inds = np.logical_and(q_r > l, q_r <= r)
        dqf_dist = dqf[inds]
        es[i] = np.median(dqf_dist)
    return es

def mfe_by_height(q, q_r, z, bins = np.linspace(5, 15, 6), q_min = 1e-6):
    n = len(bins) - 1
    es = np.zeros(n)
    dqf = np.abs(q - q_r) / q_r
    for i in range(n):
        l = bins[i]
        r = bins[i + 1]
        inds = np.logical_and(z > l, z <= r).reshape(1, -1)
        inds = np.broadcast_to(inds, dqf.shape)
        inds = np.logical_and(inds, q_r > q_min)
        dqf_dist = dqf[inds]
        es[i] = np.nanmedian(dqf_dist)
    return es
