'''
# Components

All the photonic Components defined in PhotonTorch

Each [Component](component.Component) is generally defined by several key properties:

  * num_ports: The number of ports of the components.
  * C: The connection matrix for the component (usually all zero for base components)
  * S: The scattering matrix fro the component
  * sources_at: Where there are sources in the component (usually all zero for base components)
  * detectors_at: Where there are detectors in the component (usually all zero for base components)
  * delays: delays introduced by the nodes of the component.

'''


## Components

# Component
from .component import Component

# Terms
from .terms import Term
from .terms import Source
from .terms import Detector

# Mirrors
from .mirrors import Mirror

# SOAs
from .soas import LinearSoa

# MMIs
from .mmis import Mmi21
from .mmis import Mmi33

# Waveguides
from .waveguides import Waveguide
from .waveguides import Connection

# Grating Couplers
from .gratingcouplers import GratingCoupler

# Directional couplers
from .directionalcouplers import DirectionalCoupler
from .directionalcouplers import RealisticDirectionalCoupler
from .directionalcouplers import DirectionalCouplerWithLength
