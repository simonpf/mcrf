import os
from crac.psd import D14Ice, D14Liquid
from parts.retrieval.a_priori import *

liras_path = os.environ("LIRAS_PATH")
scattering_data = os.path.join(liras_path, "data", "scattering")

################################################################################
# Ice particles
################################################################################

ice_shape      = os.path.join(scattering_data, "8-ColumnAggregate.xml")
ice_shape_meta = os.path.join(scattering_data, "8-ColumnAggregate.meta.xml")

ice_mask       = And(TropopauseMask(), TemperatureMask(0.0, 273.0))
ice_covariance = Thikhonov(scaling = 1.0, mask = ice_mask)
ice_md_a_priori = FixedAPriori("ice_md", 1e-6, ice_covariance)
ice_n0_a_priori = FixedAPriori("ice_n0", 1e10, ice_covariance)

ice = Hydrometeor("ice",
                  D14Ice(),
                  [ice_md_a_priori, ice_n0_a_priori],
                  ice_shape,
                  ice_shape_meta)

################################################################################
# Liquid particles
################################################################################

liquid_shape      = os.path.join(scattering_data, "LiquidSphere.xml")
liquid_shape_meta = os.path.join(scattering_data, "LiquidSphere.meta.xml")

liquid_mask  = TemperatureMask(240, 340.0)
liquid_covariance = Thikhonov(scaling = 1.0, mask = liquid_mask)
liquid_md_a_priori = FixedAPriori("liquid_md", 1e-6, liquid_covariance)
liquid_n0_a_priori = FixedAPriori("liquid_n0", 1e5, ice_covariance)

liquid = Hydrometeor("liquid",
                     D14Liquid(),
                     [liquid_md_a_priori, liquid_n0_a_priori],
                     liquid_shape,
                     liquid_shape_meta)
