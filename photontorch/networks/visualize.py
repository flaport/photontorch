""" Special Network Plotting Function """

#############
## Imports ##
#############

# Torch
import torch

# Other
import numpy as np

# Relative
from ..components.terms import Detector


##########
## Plot ##
##########

# The plot function to plot the detected power of a network
def plot(network, detected, **kwargs):
    """ Plot detected power versus time or wavelength

    Args:
        detected (np.ndarray|Tensor): detected power. Allowed shapes:
            * (#timesteps,)
            * (#timesteps, #detectors)
            * (#timesteps, #detectors, #batches)
            * (#timesteps, #wavelengths)
            * (#timesteps, #wavelengths, #detectors)
            * (#timesteps, #wavelengths, #detectors, #batches)
            * (#wavelengths,)
            * (#wavelengths, #detectors)
            * (#wavelengths, #detectors, #batches)
            the plot function should be smart enough to figure out what to plot.
        **kwargs: keyword arguments given to plt.plot

    Note:
        if #timesteps = #wavelengths, the plotting function will choose #timesteps
        as the first dimension

    """

    import matplotlib.pyplot as plt

    # First we define a helper function
    def plotfunc(x, y, labels, **kwargs):
        """ Helper function """
        plots = plt.plot(x, y, **kwargs)
        if labels is not None:
            for p, l in zip(plots, labels):
                p.set_label(l)
        if labels is not None and len(labels) > 1:
            # Shrink current axis by 10%
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.85, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return plots

    # Handle y
    y = detected
    if torch.is_tensor(y):
        y = y.detach().cpu()

    if len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1:
        raise ValueError("cannot plot for a single timestep and a single wavelength")

    y = np.squeeze(np.array(y, "float32"))

    # Handle x
    time_mode = wl_mode = False
    if network.env.num_t == y.shape[0]:
        time_mode = True
        x = network.env.t
    elif network.env.num_wl == y.shape[0]:
        wl_mode = True
        x = network.env.wl
    if not (time_mode or wl_mode):
        raise ValueError("First dimension should be #timesteps or #wavelengths")

    # Handle prefixes
    f = (int(np.log10(max(x)) + 0.5) // 3) * 3 - 3
    prefix = {
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: r"$\mu$",
        -9: "n",
        -12: "p",
        -15: "f",
    }[f]
    x = x * 10 ** (-f)

    # Handle labels
    plt.ylabel("Intensity [a.u.]")
    plt.xlabel("Time [%ss]" % prefix if time_mode else "Wavelength [%sm]" % prefix)

    # standard labels:
    detectors = [
        name for name, comp in network.components.items() if isinstance(comp, Detector)
    ]
    wavelengths = ["%inm" % wl for wl in 1e9 * network.env.wl]

    # Plot
    if y.ndim == 1:
        return plotfunc(x, y, None, **kwargs)

    if wl_mode:
        if y.ndim == 2:
            if y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ["batch %i" % i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            y = y.transpose(0, 2, 1)
            labels = [
                "%i | %s" % (i, det) for i in range(y.shape[1]) for det in detectors
            ]
            return plotfunc(
                x, y.reshape(network.env.num_wl, -1), labels, **kwargs
            )
        else:
            raise RuntimeError(
                "When plotting in wavelength mode, the max dim of y should be < 4"
            )

    if time_mode:
        if y.ndim == 2:
            if y.shape[1] == network.env.num_wl:
                labels = wavelengths
            elif y.shape[1] == network.num_detectors:
                labels = detectors
            else:
                labels = ["batch %i" % i for i in range(y.shape[1])]
            return plotfunc(x, y, labels, **kwargs)
        elif y.ndim == 3:
            if (
                y.shape[1] == network.env.num_wl
                and y.shape[2] == network.num_detectors
            ):
                labels = [
                    "%s | %s" % (wl, det) for wl in wavelengths for det in detectors
                ]
            elif (
                y.shape[1] == network.env.num_wl
                and y.shape[2] != network.num_detectors
            ):
                y = y.transpose(0, 2, 1)
                labels = [
                    "%i | %s" % (b, wl) for b in range(y.shape[1]) for wl in wavelengths
                ]
            elif y.shape[1] == network.num_detectors:
                y = y.transpose(0, 2, 1)
                labels = [
                    "%i | %s" % (b, det) for b in range(y.shape[1]) for det in detectors
                ]
            return plotfunc(
                x, y.reshape(network.env.num_t, -1), labels, **kwargs
            )
        elif y.ndim == 4:
            y = y.transpose(0, 3, 1, 2)
            labels = [
                "%i | %s | %s" % (b, wl, det)
                for b in range(y.shape[1])
                for wl in wavelengths
                for det in detectors
            ]
            return plotfunc(
                x, y.reshape(network.env.num_t, -1), labels, **kwargs
            )
        else:
            raise RuntimeError(
                "When plotting in time mode, the max dim of y should be < 5"
            )

    # we should never get here:
    raise ValueError(
        "Could not plot detected array. Are you sure you are you sure the "  # pragma: no cover
        "current simulation environment corresponds to the environment for "
        "which the detected tensor was calculated?"
    )


def graph(network, draw=True):
    """ create a graph visualization of the network

    Args:
        draw (bool): draw the graph with matplotlib

    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # create graph
    G = nx.MultiGraph()
    G.add_nodes_from(network.components.keys())
    G.add_edges_from([conn.split(":")[::2] for conn in network.connections])

    if draw:
        pos = nx.drawing.spring_layout(G)
        _draw_nodes(G, network.components.values(), pos)
        _draw_edges(G, pos)
        plt.gca().set_axis_off()
    return G


def _draw_nodes(G, components, pos):
    """ helper function: draw the nodes of a networkx graph of a photontorch network

    Args:
        G (graph): networkx graph to draw
        components (list): list of Photontorch components in the graph.
        pos (list): list of positions to draw the graph nodes on.

    """
    import matplotlib.pyplot as plt

    nodelist = list(G)
    node_size = 300
    node_color = "r"
    node_shape = "o"
    xy = np.asarray([pos[v] for v in nodelist])

    def _get_bbox_properties(cls):
        dic = {
            None: {"fc": "C1"},
            "object": {"fc": "C1"},
            "Component": {"fc": "C1"},
            "Waveguide": None,
            "Connection": None,
            "DirectionalCoupler": {"fc": "C0"},
            "DirectionalCouplerWithLength": {"fc": "C0"},
            "RealisticDirectionalCoupler": {"fc": "C0"},
            "Term": None,
            "Source": {"fc": "C3"},
            "Detector": {"fc": "C2"},
            "Mirror": {"fc": "C0"},
            "GratingCoupler": {"fc": "C6"},
            "Soa": {"fc": "C4"},
        }
        bbox = dic.get(cls.__name__, -1)
        if bbox == -1:
            return _get_bbox_properties(cls.__bases__[0])
        if bbox is None:
            return bbox

        bbox["ec"] = bbox.get("ec", bbox.get("fc"))

        return bbox

    for (x, y), node, comp in zip(xy, nodelist, components):
        # plt.scatter(xy[:, 0], xy[:, 1], s=node_size, c=node_color, marker=node_shape, zorder=2)
        text = plt.text(
            x,
            y,
            node,
            zorder=2,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=_get_bbox_properties(comp.__class__),
        )


def _draw_edges(G, pos):
    """ helper function: draw the edges of a networkx graph of a photontorch network

    Args:
        G (graph): networkx graph to draw
        pos (list): list of positions to draw the edges between.

    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    edge_color = "k"
    style = "solid"
    edgelist = list(G.edges())
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    for (x1, y1), (x2, y2) in edge_pos:
        r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        a = np.arctan2((y2 - y1), (x2 - x1))
        a += (2 * np.random.rand() - 1) * np.pi / 4
        plt.plot(
            [x1, x1 + 0.5 * r * np.cos(a), x2],
            [y1, y1 + 0.5 * r * np.sin(a), y2],
            lw=1.5,
            ls="-",
            color="k",
            zorder=1,
        )
    plt.draw()
