"""
Provide Hydrometeor class that extends the parts ScatteringSpecies class
with some additional attributes used by the :class:CloudRetrieval class.
"""
import artssat
from artssat.jacobian import Log10

class Hydrometeor(artssat.scattering.ScatteringSpecies):
    """
    Specialization of the artssat.scattering.ScatteringSpecies class that
    adds serveral attributes that are used to customize the behavior of
    the CloudRetrieval class.

    Attributes:

        a_priori: A priori data provider for this hydrometeor species.

        transformations: List containing two transformations to apply to the
            two moments of the hydrometeor.

        scattering_data: Path of the file containing the scattering data to
            use for this hydrometeor.

        scattering_data_meta: Path of the file containing the meta data for
            this hydrometeor.

    """
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
        self.retrieve_first_moment = True
        self.retrieve_second_moment = True
