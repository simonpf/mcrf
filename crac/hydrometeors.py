import parts
from parts.jacobian import Log10

class Hydrometeor(parts.scattering.ScatteringSpecies):
    def __init__(self,
                 name,
                 psd,
                 a_priori,
                 scattering_data,
                 scattering_meta_data):
        super().__init__(name, psd, scattering_data, scattering_meta_data)
        self.a_priori = a_priori
        self.transformations = [Log10(), Log10()]
        self.limits_low      = [1e-12, 2]
        self.radar_only             = True
        self.retrieve_second_moment = True
