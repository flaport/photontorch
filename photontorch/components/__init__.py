"""
# Photontorch base component

The base component is a parent class meant for subclassing. It should not be used
directly.

Each Component is generally defined by several key attributes defining the behavior
of the component in a network.

    `num_ports`: The number of ports of the components.

    `S`: The scattering matrix of the component.

    `C`: The connection matrix for the component (usually all zero for base components)

    `sources_at`: The location of the sources in the component (usually all zero for
        base components)

    `detectors_at`: The location of the detectors in the component (usually all zero
        for base components)

    `actions_at`: The location of the active nodes in the component (usually all zero
        for passive components)

    `delays`: delays introduced by the nodes of the component.

"""


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
from .soas import Soa
from .soas import BaseSoa
from .soas import LinearSoa
from .soas import AgrawalSoa

# MMIs
from .mmis import Mmi
from .mmis import PhaseArray

# MZIs
from .mzis import Mzi

# Waveguides
from .waveguides import Waveguide
from .waveguides import Connection

# Grating Couplers
from .gratingcouplers import GratingCoupler

# Directional couplers
from .directionalcouplers import DirectionalCoupler
from .directionalcouplers import RealisticDirectionalCoupler
from .directionalcouplers import DirectionalCouplerWithLength
