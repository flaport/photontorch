[comment]: # (This is and automatically generated readme file)
[comment]: # (To edit this file, edit the docstring in the __init__.py file)
[comment]: # (And run the documentation: python -m photontorch.documentation)

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

