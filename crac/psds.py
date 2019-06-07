from parts.scattering.psd.d14 import D14MN, D14N, D14

################################################################################
# Mass and N_0^*
################################################################################

class D14Ice(D14MN):
    def __init__(self):
        super().__init__(-0.26, 4.0, 917.0)
        self.name = "d14n"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

class D14Snow(D14MN):
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14n"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

class D14Liquid(D14MN):
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d14n"
        self.t_min = 240.0

    @property
    def moment_names(self):
        return ["n0", "md"][::-1]

################################################################################
# Mass and D_m 
################################################################################

class D14DmIce(D14):
    def __init__(self):
        super().__init__(-0.26, 4.0, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "dm"]

class D14DmSnow(D14):
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "dm"]

class D14DmLiquid(D14):
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d1n"
        self.t_min = 240.0

    @property
    def moment_names(self):
        return ["md", "dm"]

################################################################################
# N_0^* and D_m
################################################################################

class D14NDmIce(D14N):
    def __init__(self):
        super().__init__(-0.26, 4.0, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "dm"]

class D14NDmSnow(D14N):
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["n0", "dm"]

class D14NDmLiquid(D14N):
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d1n"
        self.t_min = 240.0

    @property
    def moment_names(self):
        return ["n0", "dm"]
