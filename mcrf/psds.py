"""
This module provides classes defining the particle size distribbutions used
for the cloud retrievals.
"""
from parts.scattering.psd.d14 import D14MN, D14N, D14

################################################################################
# Mass and N_0^*
################################################################################

class D14Ice(D14MN):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and normalized number density (N_0^*).
    The shape is the same as the one used for the DARDAR v3 retrievals.
    """
    def __init__(self):
        super().__init__(-0.26, 1.75, 917.0)
        self.name = "d14n"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

class D14Snow(D14MN):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and normalized number density (N_0^*).
    The shape parameters were obtained by fit to cloud scenes from the ICON
    model.
    """
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14n"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

class D14Liquid(D14MN):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and normalized number density (N_0^*).
    The shape parameters were obtained by fit to cloud scenes from the ICON
    model.
    """
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d14n"
        self.t_min = 270.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

################################################################################
# Mass and D_m 
################################################################################

class D14DmIce(D14):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and mass weighted mean diameter (D_m).
    The shape is the same as the one used for the DARDAR v3 retrievals.
    """
    def __init__(self):
        super().__init__(-0.26, 1.75, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "dm"]

class D14DmSnow(D14):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and mass weighted mean diameter (D_m).
    The shape is shape has been obtained by fit to ICON model scenes.
    """
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "dm"]

class D14DmLiquid(D14):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and mass weighted mean diameter (D_m).
    The shape is shape has been obtained by fit to ICON model scenes.
    """
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d1n"
        self.t_min = 270.0

    @property
    def moment_names(self):
        return ["md", "dm"]

################################################################################
# N_0^* and D_m
################################################################################

class D14NDmIce(D14N):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using normalized number density (N_0^*) and mass weighted
    mean diameter (D_m). The shape is the same as the one used for the
    DARDAR v3 retrievals.
    """
    def __init__(self):
        super().__init__(-0.26, 1.75, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "dm"]

class D14NDmSnow(D14N):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using normalized number density (N_0^*) and mass weighted
    mean diameter (D_m). The shape parameters have been obtain by fit to
    distributions within a scene of the ICON model.
    """
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "dm"]

class D14NDmLiquid(D14N):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using normalized number density (N_0^*) and mass weighted
    mean diameter (D_m). The shape parameters have been obtain by fit to
    distributions within a scene of the ICON model.
    """
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d14"
        self.t_min = 270.0

    @property
    def moment_names(self):
        return ["n0", "dm"]
