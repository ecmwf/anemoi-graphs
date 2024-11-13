# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objs as go
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian


def is_in_range(val, rng):
    return val >= rng[0] and val <= rng[1]


def node_list(graph: HeteroData, nodes_name: str, mask: Optional[list[bool]] = None) -> tuple[list[float], list[float]]:
    """Get the latitude and longitude of the nodes.

    Parameters
    ----------
    graph : dict[str, torch.Tensor]
        Graph to plot.
    nodes_name : str
        Name of the nodes.
    mask : list[bool], optional
        Mask to filter the nodes. Default is None.

    Returns
    -------
    latitudes : list[float]
        Latitude coordinates of the nodes.
    longitudes : list[float]
        Longitude coordinates of the nodes.
    """
    coords = np.rad2deg(graph[nodes_name].x.numpy())
    latitudes = coords[:, 0]
    longitudes = coords[:, 1]
    if mask is not None:
        latitudes = latitudes[mask]
        longitudes = longitudes[mask]
    return latitudes.tolist(), longitudes.tolist()


def edge_list(graph: HeteroData, source_nodes_name: str, target_nodes_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the edge list.

    This method returns the edge list to be represented in a graph. It computes the coordinates of the points connected
    and include NaNs to separate the edges.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    source_nodes_name : str
        Name of the source nodes.
    target_nodes_name : str
        Name of the target nodes.

    Returns
    -------
    latitudes : np.ndarray
        Latitude coordinates of the points connected.
    longitudes : np.ndarray
        Longitude coordinates of the points connected.
    """
    sub_graph = graph[(source_nodes_name, "to", target_nodes_name)].edge_index
    x0 = np.rad2deg(graph[source_nodes_name].x[sub_graph[0]])
    y0 = np.rad2deg(graph[target_nodes_name].x[sub_graph[1]])
    nans = np.full_like(x0[:, :1], np.nan)
    latitudes = np.concatenate([x0[:, :1], y0[:, :1], nans], axis=1).flatten()
    longitudes = np.concatenate([x0[:, 1:2], y0[:, 1:2], nans], axis=1).flatten()
    return latitudes, longitudes


def compute_node_adjacencies(
    graph: HeteroData, source_nodes_name: str, target_nodes_name: str
) -> tuple[list[int], list[str]]:
    """Compute the number of adjacencies of each target node in a bipartite graph.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    source_nodes_name : str
        Name of the dimension of the coordinates for the head nodes.
    target_nodes_name : str
        Name of the dimension of the coordinates for the tail nodes.

    Returns
    -------
    num_adjacencies : np.ndarray
        Number of adjacencies of each node.
    """
    node_adjacencies = np.zeros(graph[target_nodes_name].num_nodes, dtype=int)
    vals, counts = np.unique(graph[(source_nodes_name, "to", target_nodes_name)].edge_index[1], return_counts=True)
    node_adjacencies[vals] = counts
    return node_adjacencies


def get_node_adjancency_attributes(graph: HeteroData) -> dict[str, tuple[str, np.ndarray]]:
    """Get the node adjacencies for each subgraph."""
    node_adj_attr = {}
    for (source_nodes_name, _, target_nodes_name), _ in graph.edge_items():
        attr_name = f"# connections from {source_nodes_name}"
        node_adj_vector = compute_node_adjacencies(graph, source_nodes_name, target_nodes_name)
        if target_nodes_name not in node_adj_attr:
            node_adj_attr[target_nodes_name] = {attr_name: node_adj_vector}
        else:
            node_adj_attr[target_nodes_name][attr_name] = node_adj_vector

    return node_adj_attr


def compute_isolated_nodes(graph: HeteroData) -> dict[str, tuple[list, list]]:
    """Compute the isolated nodes.

    Parameters
    ----------
    graph : HeteroData
        Graph.

    Returns
    -------
    dict[str, list[int]]
        Dictionary with the isolated nodes for each subgraph.
    """
    isolated_nodes = {}
    for (source_name, _, target_name), sub_graph in graph.edge_items():
        head_isolated = np.ones(graph[source_name].num_nodes, dtype=bool)
        tail_isolated = np.ones(graph[target_name].num_nodes, dtype=bool)
        head_isolated[sub_graph.edge_index[0]] = False
        tail_isolated[sub_graph.edge_index[1]] = False
        if np.any(head_isolated):
            isolated_nodes[f"{source_name} isolated (--> {target_name})"] = node_list(
                graph, source_name, mask=list(head_isolated)
            )
        if np.any(tail_isolated):
            isolated_nodes[f"{target_name} isolated ({source_name} -->)"] = node_list(
                graph, target_name, mask=list(tail_isolated)
            )

    return isolated_nodes


def get_node_attribute_dims(graph: HeteroData) -> dict[str, int]:
    """Get dimensions of the node attributes.

    Parameters
    ----------
    graph : HeteroData
        The graph to inspect.

    Returns
    -------
    dict[str, int]
        A dictionary with the attribute names as keys and the number of dimensions as values.
    """
    attr_dims = {}
    for nodes in graph.node_stores:
        for attr in nodes.node_attrs():
            if attr == "x" or not isinstance(nodes[attr], torch.Tensor):
                continue
            elif attr not in attr_dims:
                attr_dims[attr] = nodes[attr].shape[1]
            else:
                assert (
                    nodes[attr].shape[1] == attr_dims[attr]
                ), f"Attribute {attr} has different dimensions in different nodes."
    return attr_dims


def get_edge_attribute_dims(graph: HeteroData) -> dict[str, int]:
    """Get dimensions of the node attributes.

    Parameters
    ----------
    graph : HeteroData
        The graph to inspect.

    Returns
    -------
    dict[str, int]
        A dictionary with the attribute names as keys and the number of dimensions as values.
    """
    attr_dims = {}
    for edges in graph.edge_stores:
        for attr in edges.edge_attrs():
            if attr == "edge_index":
                continue
            elif attr not in attr_dims:
                attr_dims[attr] = edges[attr].shape[1]
            else:
                assert (
                    edges[attr].shape[1] == attr_dims[attr]
                ), f"Attribute {attr} has different dimensions in different edges."
    return attr_dims


def convert_and_plot_nodes(
    G, coords, x_range=range(-1, 1), y_range=[-1, 1], z_range=[-1, 1], scale=1.0, color="skyblue"
):
    """Filters coordinates of nodes in a graph, scales and plots them."""

    lat = coords[:, 0].numpy()  # Latitude
    lon = coords[:, 1].numpy()  # Longitude

    # Convert lat/lon to Cartesian coordinates for the filtered nodes
    x_nodes, y_nodes, z_nodes = latlon_rad_to_cartesian((lat, lon)).T

    # Filter nodes
    filtered_nodes = [
        i
        for i, (x, y, z) in enumerate(zip(x_nodes, y_nodes, z_nodes))
        if is_in_range(x, x_range) and is_in_range(y, y_range) and is_in_range(z, z_range)
    ]

    if not filtered_nodes:
        print("No nodes found in the given range.")
        return

    graph = G.subgraph(filtered_nodes).copy()

    # Extract node positions for Plotly
    x_nodes_filtered = [x_nodes[node] * scale for node in graph.nodes()]
    y_nodes_filtered = [y_nodes[node] * scale for node in graph.nodes()]
    z_nodes_filtered = [z_nodes[node] * scale for node in graph.nodes()]

    # Create traces for nodes
    node_trace = go.Scatter3d(
        x=x_nodes_filtered,
        y=y_nodes_filtered,
        z=z_nodes_filtered,
        mode="markers",
        marker=dict(size=3, color=color, opacity=0.8),
        text=list(graph.nodes()),
        hoverinfo="none",
    )

    return node_trace, graph, (x_nodes, y_nodes, z_nodes)


def get_edge_trace(g1, g2, n1, n2, scale_1, scale_2, color="blue", x_range=[-1, 1], y_range=[-1, 1], z_range=[-1, 1]):
    """Gets all edges between g1 and g2 (two separate graphs, hierarchical graph setting)."""
    edge_traces = []
    for edge in g1.edges():
        # Convert edge nodes to their new indices
        idx0, idx1 = edge[0], edge[1]

        if idx0 in g1.nodes and idx1 in g2.nodes:
            if (
                is_in_range(n1[0][idx0], x_range)
                and is_in_range(n2[0][idx1], x_range)
                and is_in_range(n1[1][idx0], y_range)
                and is_in_range(n2[1][idx1], y_range)
                and is_in_range(n1[2][idx0], z_range)
                and is_in_range(n2[2][idx1], z_range)
            ):
                x_edge = [n1[0][idx0] * scale_1, n2[0][idx1] * scale_2, None]
                y_edge = [n1[1][idx0] * scale_1, n2[1][idx1] * scale_2, None]
                z_edge = [n1[2][idx0] * scale_1, n2[2][idx1] * scale_2, None]
                edge_trace = go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line=dict(width=2, color=color),
                    showlegend=False,
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)
    return edge_traces


def make_layout(title, showbackground=True, axis_visible=True):
    # Create a layout for the plot
    layout = go.Layout(
        title={
            "text": f"<br><sup>{title}</sup>",
            "x": 0.5,  # Center the title horizontally
            "xanchor": "center",  # Anchor the title to the center of the plot area
            "y": 0.95,  # Position the title vertically
            "yanchor": "top",  # Anchor the title to the top of the plot area
        },
        scene=dict(
            xaxis=dict(showbackground=showbackground, visible=axis_visible, showgrid=axis_visible, range=(-1, 1)),
            yaxis=dict(showbackground=showbackground, visible=axis_visible, showgrid=axis_visible, range=(-1, 1)),
            zaxis=dict(showbackground=showbackground, visible=axis_visible, showgrid=axis_visible, range=(-1, 1)),
            aspectmode="manual",  # Manually set aspect ratios
            aspectratio=dict(x=2, y=2, z=2),  # Fixed aspect ratio
        ),
        autosize=False,  # Prevent autosizing based on data
        width=900,  # Increase width
        height=600,  # Increase height
        showlegend=False,
    )
    return layout


def generate_shades(color_name, num_shades):
    # Get the base color from the name
    base_color = mcolors.CSS4_COLORS.get(color_name.lower(), None)

    if num_shades == 1:
        return [base_color]

    if not base_color:
        raise ValueError(f"Color '{color_name}' is not recognized.")

    # Convert the base color to RGB
    base_rgb = mcolors.hex2color(base_color)

    # Create a colormap that transitions from the base color to a darker version of the base color
    dark_color = tuple([x * 0.6 for x in base_rgb])  # Darker shade of the base color
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [base_rgb, dark_color], N=num_shades)

    # Generate the shades
    shades = [mcolors.to_hex(cmap(i / (num_shades - 1))) for i in range(num_shades)]

    return shades
