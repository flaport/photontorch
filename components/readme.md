[comment]: # (This is and automatically generated readme file)
[comment]: # (To edit this file, edit the docstring in the __init__.py file)
[comment]: # (And run the documentation: python -m photontorch.documentation)

# Components

All the photonic Components defined in PhotonTorch

Each [Component](component.Component) is generally defined by several key properties:

  * num_ports: The number of ports of the components.
  * C: The connection matrix for the component (usually all zero for base components)
  * S: The scattering matrix fro the component
  * sources_at: Where there are sources in the component (usually all zero for base components)
  * detectors_at: Where there are detectors in the component (usually all zero for base components)
  * delays: delays introduced by the nodes of the component.

