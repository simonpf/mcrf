class Hydrometeor(parts.scattering.ScatteringSpecies):
    def __init__(self,
                 name,
                 psd,
                 a_priori,
                 scattering_data,
                 scattering_meta_data):
        super().__init__(name, psd, scattering_data, scattering_meta_data)
        self.transformations = [Log10(), Log10()]
        self.limits_low      = [1e-12, 2]
