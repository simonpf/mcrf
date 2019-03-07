from parts.scattering.psd.d14 import D14MN

class D14Ice(D14MN):
    def __init__(self):
        super().__init__(-1.0, 3.0, 917.0)
        self.name = "d14mn"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "n0"]

class D14Snow(D14MN):
    def __init__(self):
        super().__init__(2.654, 0.750, 917.0)
        self.name = "d14mn"
        self.t_max = 280.0

    @property
    def moment_names(self):
        return ["md", "n0"]

class D14Liquid(D14MN):
    def __init__(self):
        super().__init__(2.0, 1.0, 1000.0)
        self.name  = "d14mn"
        self.t_min = 240.0

    @property
    def moment_names(self):
        return ["md", "n0"]
