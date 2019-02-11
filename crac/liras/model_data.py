"""
The :code:`model_data` module contains code for accessing the model data
used for the radiative transfer simulations.

The data from scenes A and B is available in different forms.

The data for scene B is contained in 201 separate transects contained
in two ARTS XML files each.

The data from scene A is contained in a .mat file.
"""

import numpy    as np
import scipy    as sp
import scipy.io as sio
import typhon
import glob
import os

from h5py import File
from typhon.arts.xml import load

liras_path = os.environ["LIRAS_PATH"]

################################################################################
# Model data API
################################################################################

gem_psd_parameters = {"LWC" : [1.0, 1.0, 524.0, 3.0],
                      "RWC" : [0.0, 1.0, 524.0, 3.0],
                      "IWC" : [0.0, 1.0, 440.0, 3.0],
                      "SWC" : [0.0, 1.0,  52.4, 3.0],
                      "GWC" : [0.0, 1.0, 209.0, 3.0],
                      "HWC" : [0.0, 1.0, 471.0, 3.0]}

class ModelData():
    """
    The :class:`ModelDataNew` class provides access to the GEM model data
    contained in *.mat files.

    Attributes:

        file(:code:): File handle to the :code:`.mat` containing the model
            data

    """
    path = "/home/simonpf/projects/LIRAS/EWS_simulations_LIRAS/atmdata/" \
           + "EWS_data_SceneAandB_FullArcs_1km.39320.Idx1_Inf"
    cache_size = 3

    def __init__(self, path = "/home/simonpf/Dendrite/UserAreas/Simon/ipa_ICI_ScA_Arc1km.mat"):
        """
        Create :class:`ModelDataA` instance and lookup available files.
        """
        self.file = File(path, "r")

        self.i_t = 0
        self.i_p = 0

    def get_number_of_profiles(self):
        return self.file["height_thermodynamic"].shape[-1]

    def get_pressure_grid(self, i_t = None, i_p = None):
        """
        Get the pressure grid from a given profile and transect.

        Parameters:

            i_t(int): The transect index to get the pressure from.

            i_p(int): The profile index to get the pressure from.

        Return:

            :code:`numpy.ndarray` containing the pressure grid
            of the given profile.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        return self.file["pressure_thermodynamic"][:, i_t, i_p][::-1]

    def get_temperature_profile(self, i_t = None, i_p = None):
        """
        Get the temperature profile from a given transect.

        Parameters:

            i_t(int): The transect index to get the temperature from.

            i_p(int): The profile index to get the temperature from.

        Return:

            :code:`numpy.ndarray` containing the temperature profile
            from the given transect.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        return self.file["temperature"][:, i_t, i_p][::-1]

    def get_altitude_profile(self, i_t = None, i_p = None):
        """
        Get the altitude profile from given transect.

        Parameters:

            i_t(int): The transect index to get the altitude from.

            i_p(int): The profile index.

        Return:

            :code:`numpy.ndarray` containing the altitude profile
            with given index and from given transect.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        return self.file["height_thermodynamic"][:, i_t, i_p][::-1]

    def get_scatterer(self, species, moment, i_t = None, i_p = None):

        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        name_lookup = {"IWC" : "ice",
                       "SWC" : "snow",
                       "GWC" : "graupel",
                       "HWC" : "hail",
                       "LWC" : "cloud",
                       "RWC" : "rain"}

        moment_lookup = {"mass_density" : "water_content",
                        "number_density" : "number_concentration"}

        name = moment_lookup[moment] + "_" + name_lookup[species]

        return self.file[name][:, i_t, i_p][::-1]

    def get_absorber(self, species, i_t = None, i_p = None):

        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        if species == "H2O":
            return self.file["vmr_h2o"][:, i_t, i_p][::-1]

        if species == "O3":
            return np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.])
        if species == "O2":
            return np.array([0.20564152, 0.20545728, 0.20550998, 0.2055904 , 0.2057184 ,
                          0.20587689, 0.20604537, 0.20616054, 0.20617701, 0.20614173,
                          0.2061263 , 0.20621029, 0.20638656, 0.20657266, 0.20674402,
                          0.20693583, 0.20716823, 0.20739526, 0.20762796, 0.20789085,
                          0.20811935, 0.20831731, 0.20847917, 0.20857159, 0.20869443,
                          0.20900947, 0.20931807, 0.2093402 , 0.20933995, 0.20942354,
                          0.20946913, 0.20949371, 0.20949676, 0.20949709, 0.20949767,
                          0.20949834, 0.20949857, 0.2094986 , 0.20949871, 0.20949889,
                          0.20949903, 0.20949907, 0.20949905, 0.20949903, 0.20949903,
                          0.20949901, 0.209499  , 0.20949899, 0.20949899, 0.20949899,
                          0.20949898, 0.20949897, 0.20949894, 0.20949885, 0.20949874,
                          0.20949875, 0.2094988 , 0.20949881])
        if species == "N2":
            return np.array([0.76641956, 0.76573291, 0.76592934, 0.76622904, 0.76670611,
                             0.76729678, 0.7679247 , 0.76835394, 0.76841531, 0.76828385,
                             0.76822632, 0.76853936, 0.76919632, 0.76988989, 0.77052854,
                             0.77124341, 0.77210957, 0.7729557 , 0.77382296, 0.77480273,
                             0.77565435, 0.77639217, 0.77699539, 0.77733986, 0.77779767,
                             0.77897181, 0.78012196, 0.78020442, 0.78020349, 0.78051505,
                             0.78068496, 0.78077658, 0.78078792, 0.78078915, 0.78079132,
                             0.78079381, 0.78079467, 0.7807948 , 0.78079518, 0.78079585,
                             0.78079639, 0.78079653, 0.78079646, 0.7807964 , 0.78079638,
                             0.78079633, 0.78079627, 0.78079625, 0.78079625, 0.78079623,
                             0.7807962 , 0.78079616, 0.78079605, 0.78079571, 0.78079532,
                             0.78079532, 0.78079552, 0.78079558])

    @property
    def profiles(self):
        """
        Iterator over profiles in the model data. This returns the model
        data instance with the transect and profile indices set, so that
        data can be extracte by calling any of the :code:`get_*` functions
        without additional arguments.`
        """
        s = self.file["water_content_ice"].shape
        for i_t in range(s[0]):
            self.i_t = i_t
            for i_p in range(s[1]):
                self.i_p = i_p
                yield self

    def get_transect_view(self, i_t = None, species = ["IWC"]):

        if i_t is None:
            i_t = self.i_t

        name_lookup = {"IWC" : "water_content_ice",
                       "SWC" : "water_content_snow",
                       "HWC" : "water_content_hail",
                       "GWC" : "water_content_graupel",
                       "LWC" : "water_content_cloud",
                       "RWC" : "water_content_rain",
                       "Temperature" : "temperature"}

        s = self.file["water_content_ice"].shape
        x = np.zeros((s[0], s[2]))

        for s in species:
            name = name_lookup[s]
            x += self.file[name][:, i_t, :][::-1, :]

        return x


class ModelDataOld():
    """
    The :class:`ModelDataA` class provides acces to the GEM model data.
    The model data is given in multiple files, each containing the
    profiles from one transect though the model scene.

    The class uses a simple file cache so that consecutively
    accessesing the same file should be fast.

    Attributes:

        cache(:code:`dict`): The :code:`dict` used to cache opened data
        files.

        cache_size(:code:`int`): The size of the file cache to use.

        files(:code:`list`): The list of transect files found in the given
        folder.

        i_t(:code:`int`): Transect index used for iterating the data files.

        i_p(:code:`int`): Profile index used for iterating the data files.

        path(:code:`str`): The path containing the model data.

    """
    path = "/home/simonpf/projects/LIRAS/EWS_simulations_LIRAS/atmdata/" \
           + "EWS_data_SceneAandB_FullArcs_1km.39320.Idx1_Inf"
    cache_size = 3

    def __init__(self):
        """
        Create :class:`ModelDataA` instance and lookup available files.
        """
        self.files = glob.glob(os.path.join(self.path, "*atm_data*.xml"))
        #self.files.sort()
        self.cache = {}

        self.i_t = 0
        self.i_p = 0

    def get_number_of_profiles(self):
        return len(self.files) * len(self.get_fields(0))

    def get_fields(self, i_t = None):
        """
        Get the :code:`GriddedField` data from a given transect. This
        function is cached so consecutive accessing of the same file
        will be fast.

        Parameters:

            i_t(int): The index of the transect.

        Returns:

            :code:`GriddedField` instance containing the data of the
            transect with index :code:`i_t`.
        """
        if i_t is None:
            i_t = self.i_t

        if not i_t in self.cache:
            if len(self.cache) > ModelDataB.cache_size:
                self.cache.popitem()
            print("Loading: ", self.files[i_t])
            self.cache[i_t] = load(self.files[i_t])
        return self.cache[i_t]

    def get_pressure_grid(self, i_t = None, i_p = None):
        """
        Get the pressure grid from a given profile and transect.

        Parameters:

            i_t(int): The transect index to get the pressure from.

            i_p(int): The profile index to get the pressure from.

        Return:

            :code:`numpy.ndarray` containing the pressure grid
            of the given profile.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)
        return f[i_p].grids[1]

    def get_temperature_profile(self, i_t = None, i_p = None):
        """
        Get the temperature profile from a given transect.

        Parameters:

            i_t(int): The transect index to get the temperature from.

            i_p(int): The profile index to get the temperature from.

        Return:

            :code:`numpy.ndarray` containing the temperature profile
            from the given transect.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)
        return f[i_p][0, :, 0, 0]

    def get_altitude_profile(self, i_t = None, i_p = None):
        """
        Get the altitude profile from given transect.

        Parameters:

            i_t(int): The transect index to get the altitude from.

            i_p(int): The profile index.

        Return:

            :code:`numpy.ndarray` containing the altitude profile
            with given index and from given transect.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)
        return f[i_p][1, :, 0, 0]

    def get_scatterer(self, species, moment, i_t = None, i_p = None):

        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]
        try:
            i = f.grids[0].index("scat_species-" + species + "-" + moment)
        except:
            raise Exception("Could not find species {0} and moment {1} "
                            "in the data.".format(species, moment))
        return f[i, :, 0, 0]

    def get_absorber(self, species, i_t = None, i_p = None):

        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]
        try:
            i = f.grids[0].index("abs_species-" + species)
        except:
            raise Exception("Could not find species {0}in the data."
                            .format(species, moment))
        return f[i, :, 0, 0]

    def get_ice_number_density(self, i_t = None, i_p = None):
        """
        Get the number density profile of frozen hydrometeors. This
        sums the number densities of the ice, graupel, hail and
        snow hydrometeors in the profile.

        Parameters:

            i_t(int): The index of the transect to take the profile from.

            i_p(int): The index of the profile within the transect.

        Returns:

            The total number density profile of frozen hydrometeors.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]

        iwc = None
        species = ["IWC", "GWC", "HWC", "SWC"]
        for s in species:
            name = "scat_species-" + s + "-number_density"
            i = [i for i,n  in enumerate(f.grids[0]) if name in n]
            if iwc is None:
                iwc = f[i, :, 0, 0]
            else:
                iwc += f[i, :, 0, 0]

        return iwc

    def get_ice_mass_density(self, i_t = None, i_p = None):
        """
        Get the mass density profile of frozen hydrometeors. This
        sums the number densities of the ice, graupel, hail and
        snow hydrometeors in the profile.

        Parameters:

            i_t(int): The index of the transect to take the profile from.

            i_p(int): The index of the profile within the transect.

        Returns:

            The total number density profile of frozen hydrometeors.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]

        iwc = None
        species = ["IWC", "GWC", "HWC", "SWC"]
        for s in species:
            name = "scat_species-" + s + "-mass_density"
            i = [i for i,n  in enumerate(f.grids[0]) if name in n]
            if iwc is None:
                iwc = f[i, :, 0, 0]
            else:
                iwc += f[i, :, 0, 0]

        return iwc

    def get_ice_psd(self, i_t = None):
        """
        Get the ice PSD data of frozen hydrometeors.

        """

    def get_liquid_number_density(self, i_t = None, i_p = None):
        """
        Get the number density profile of liquid hydrometeors. This
        sums the number densities of the rain and liquid water in the
        profile.

        Parameters:

            i_t(int): The index of the transect to take the profile from.

            i_p(int): The index of the profile within the transect.

        Returns:

            The total number density profile of liquid hydrometeors.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]

        lwc = None
        species = ["RWC", "CWC"]
        for s in species:
            name = "scat_species-" + s + "-number_density"
            i = [i for i,n  in enumerate(f.grids[0]) if name in n]
            if lwc is None:
                lwc = f[i, :, 0, 0]
            else:
                lwc += f[i, :, 0, 0]

        return lwc

    def get_liquid_mass_density(self, i_t = None, i_p = None):
        """
        Get the mass density profile of liquid hydrometeors. This
        sums the number densities of the rain and liquid hydrometeors
        in the profile.

        Parameters:

            i_t(int): The index of the transect to take the profile from.

            i_p(int): The index of the profile within the transect.

        Returns:

            The total number density profile of liquid hydrometeors.

        """
        if i_t is None:
            i_t = self.i_t
        if i_p is None:
            i_p = self.i_p

        f = self.get_fields(i_t)[i_p]

        lwc = None
        species = ["RWC", "LWC"]
        for s in species:
            name = "scat_species-" + s + "-mass_density"
            i = [i for i,n  in enumerate(f.grids[0]) if name in n]
            if lwc is None:
                lwc = f[i, :, 0, 0]
            else: lwc += f[i, :, 0, 0]
        return lwc

    @property
    def profiles(self):
        """
        Iterator over profiles in the model data. This returns the model
        data instance with the transect and profile indices set, so that
        data can be extracte by calling any of the :code:`get_*` functions
        without additional arguments.`
        """
        for i_t in range(len(self.files)):
            self.i_t = i_t
            fs = self.get_fields(i_t)
            for i_p in range(len(fs)):
                self.i_p = i_p
                yield self

    def get_transect_view(self, i_t = None, species = ["IWC"]):

        if i_t is None:
            i_t = self.i_t

        fs = self.get_fields(i_t)
        n = len(self.get_fields(i_t))
        m = fs[0].grids[1].size

        x = np.zeros((m, n))

        if species == "Temperature":
            for i, f in enumerate(fs):
                name = "T"
                j = f.grids[0].index(name)
                x[:, i] += f[j, :, 0, 0]
            return x

        for i, f in enumerate(fs):
            for s in species:
                name = "scat_species-" + s + "-mass_density"
                j = f.grids[0].index(name)
                x[:, i] += f[j, :, 0, 0]
        return x

################################################################################
# ModelDataProvider
################################################################################

class ModelDataProvider(DataProviderBase):
    """
    Data provider for the model scene data implementing the parts
    data provider interface.

    Attributes:

    i_t: Index of the transect from which to provide the data.

    ice_psd: A PSD object used to describe the PSD used to represent
             frozen hydrometeors.

    liquid_psd: A PSD object used to describe the PSD used to represent
                liquid hydrometeors.
    """
    def __init__(self, i_t,
                 ice_psd    = None,
                 snow_psd   = None,
                 liquid_psd = None,
                 scene = "B"):

        super().__init__()

        self.i_t = i_t


        if scene == "A":
            self.m = ModelData(os.path.join(liras_path, "data", "scene_a.mat"))
        else:
            self.m = ModelData(os.path.join(liras_path, "data", "scene_b.mat"))

        self.ice_psd  = ice_psd
        self.liquid_psd = liquid_psd

        self.perturbations = {}

        if snow_psd is None:
            ice_species = ["IWC", "SWC", "HWC", "GWC"]
        else:
            ice_species = ["IWC"]


        def make_getter(name, psd, species, i):
            def getter(self, i_p):
                return self.get_psd_moment(name, psd, species, i, i_p)
            return getter

        if not ice_psd is None:

            for i, mn in enumerate(ice_psd.moment_names):
                name = "get_ice_" + mn
                f = make_getter(name, self.ice_psd, ice_species, i)
                self.__dict__[name] = f.__get__(self)

            for i, mn in enumerate(ice_psd.moment_names):
                name = "get_snow_" + mn
                f = make_getter(name, self.ice_psd, ["SWC", "HWC", "GWC"], i)
                self.__dict__[name] = f.__get__(self)

        if not liquid_psd is None:

            for i, mn in enumerate(liquid_psd.moment_names):
                name = "get_liquid_" + mn
                f = make_getter(name, self.liquid_psd, ["RWC", "LWC"], i)
                self.__dict__[name] = f.__get__(self)

            for i, mn in enumerate(liquid_psd.moment_names):
                name = "get_rain_" + mn
                f = make_getter(name, self.liquid_psd, ["RWC"], i)
                self.__dict__[name] = f.__get__(self)

            for i, mn in enumerate(liquid_psd.moment_names):
                name = "get_cloud_water_" + mn
                f = make_getter(name, self.liquid_psd, ["LWC"], i)
                self.__dict__[name] = f.__get__(self)

    def get_pressure(self, i_p):
        p = self.m.get_pressure_grid(self.i_t, i_p)
        return p

    def get_temperature(self, i_p):
        t = self.m.get_temperature_profile(self.i_t, i_p)
        return t

    def get_altitude(self, i_p):
        z = self.m.get_altitude_profile(self.i_t, i_p)
        return z

    def get_psd_moment(self, name, psd, species, n, i_p):
        m  = None
        md = None
        nd = None

        for s in species:

            if md is None and nd is None:
                md = np.copy(self.m.get_scatterer(s, "mass_density", self.i_t, i_p,))
                nd = np.copy(self.m.get_scatterer(s, "number_density", self.i_t, i_p))
            else:
                md += self.m.get_scatterer(s, "mass_density", self.i_t, i_p,)
                nd += self.m.get_scatterer(s, "number_density", self.i_t, i_p)

        my05 = MY05(*gem_psd_parameters[s],
                    mass_density = md,
                    number_density = nd)
        psd.convert_from(my05)

        if hasattr(psd, "cutoff_low"):
            cutoff = psd.cutoff_low[n]
        else:
            cutoff = 1e-12

        x = psd.moments[n]
        x[np.logical_not(np.isfinite(x))] = cutoff
        x = np.maximum(x, cutoff)

        if name in self.perturbations:
            p = self.perturbations[name]
            if p["type"] == "multiplicative":
                x*= p["dx"]
            else:
                x += p["dx"]
        return x

    def add_perturbation(self, name, dx, t = "multiplicative"):
        self.perturbations[name] = {"type" : t,
                                    "dx" : dx}

    def get_H2O(self, i_p):
        return self.m.get_absorber("H2O", i_p = i_p)

    def get_relative_humidity(self, i_p):
        from typhon.physics.atmosphere import vmr2relative_humidity
        t   = self.get_temperature(i_p).ravel()
        p   = self.get_pressure(i_p).ravel()
        vmr = self.get_H2O(i_p).ravel()
        q   = vmr2relative_humidity(vmr, p, t)
        return q

    def get_O2(self, i_p):
        return self.m.get_absorber("O2", i_p)

    def get_O3(self, i_p):
        return self.m.get_absorber("O3", i_p)

    def get_N2(self, i_p):
        return self.m.get_absorber("N2", i_p)

    def get_surface_temperature(self, i_p):
        return self.m.get_temperature_profile(self.i_t, i_p)[0]

    def get_surface_salinity(self, i_p):
        return 0.034

    def get_surface_wind_speed(self, i_p):
        return 0.0

    def get_cloud_water(self, i_p):
        t = self.get_temperature(i_p)
        cw = np.copy(self.m.get_scatterer("LWC", "mass_density", self.i_t, i_p))
        cw[t <= 215] = 0.0
        return cw

    #
    # GEM hydrometeors
    #

    def get_gem_ice_mass_density(self, i_p):
        return self.m.get_scatterer("IWC", "mass_density", self.i_t, i_p)

    def get_gem_ice_number_density(self, i_p):
        return self.m.get_scatterer("IWC", "number_density", self.i_t, i_p)

    def get_gem_snow_mass_density(self, i_p):
        return self.m.get_scatterer("SWC", "mass_density", self.i_t, i_p)

    def get_gem_snow_number_density(self, i_p):
        return self.m.get_scatterer("SWC", "number_density", self.i_t, i_p)

    def get_gem_hail_mass_density(self, i_p):
        return self.m.get_scatterer("HWC", "mass_density", self.i_t, i_p)

    def get_gem_hail_number_density(self, i_p):
        return self.m.get_scatterer("HWC", "number_density", self.i_t, i_p)

    def get_gem_graupel_mass_density(self, i_p):
        return self.m.get_scatterer("GWC", "mass_density", self.i_t, i_p)

    def get_gem_graupel_number_density(self, i_p):
        return self.m.get_scatterer("GWC", "number_density", self.i_t, i_p)

    def get_gem_rain_mass_density(self, i_p):
        return self.m.get_scatterer("RWC", "mass_density", self.i_t, i_p)

    def get_gem_rain_number_density(self, i_p):
        return self.m.get_scatterer("RWC", "number_density", self.i_t, i_p)

    def get_gem_liquid_mass_density(self, i_p):
        return self.m.get_scatterer("LWC", "mass_density", self.i_t, i_p)

    def get_gem_liquid_number_density(self, i_p):
        return self.m.get_scatterer("LWC", "number_density", self.i_t, i_p)
