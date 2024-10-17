import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import HeteroData
import torch_geometric
from torch_geometric.utils.convert import to_networkx

from anemoi.graphs.plotting.prepare import compute_isolated_nodes
from anemoi.graphs.plotting.prepare import compute_node_adjacencies
from anemoi.graphs.plotting.prepare import edge_list
from anemoi.graphs.plotting.prepare import node_list
from anemoi.graphs.plotting.prepare import generate_shades
from anemoi.graphs.plotting.prepare import make_layout
from anemoi.graphs.plotting.prepare import convert_and_plot_nodes
from anemoi.graphs.plotting.prepare import get_edge_trace
from anemoi.graphs.plotting.style import *

LOGGER = logging.getLogger(__name__)

def plot_interactive_subgraph(
    graph: HeteroData,
    edges_to_plot: tuple[str, str, str],
    out_file: Optional[Union[str, Path]] = None,
) -> None:
    """Plots a bipartite graph (bi-graph).

    This methods plots the bipartite graph passed in an interactive window (using Ploty).

    Parameters
    ----------
    graph : dict
        The graph to plot.
    edges_to_plot : tuple[str, str]
        Names of the edges to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    source_name, _, target_name = edges_to_plot
    edge_x, edge_y = edge_list(graph, source_nodes_name=source_name, target_nodes_name=target_name)
    assert source_name in graph.node_types, f"edges_to_plot ({source_name}) should be in the graph"
    assert target_name in graph.node_types, f"edges_to_plot ({target_name}) should be in the graph"
    lats_source_nodes, lons_source_nodes = node_list(graph, source_name)
    lats_target_nodes, lons_target_nodes = node_list(graph, target_name)

    # Compute node adjacencies
    node_adjacencies = compute_node_adjacencies(graph, source_name, target_name)
    node_text = [f"# of connections: {x}" for x in node_adjacencies]

    edge_trace = go.Scattergeo(
        lat=edge_x,
        lon=edge_y,
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
        mode="lines",
        name="Connections",
    )

    source_node_trace = go.Scattergeo(
        lat=lats_source_nodes,
        lon=lons_source_nodes,
        mode="markers",
        hoverinfo="text",
        name=source_name,
        marker={
            "showscale": False,
            "color": "red",
            "size": 2,
            "line_width": 2,
        },
    )

    target_node_trace = go.Scattergeo(
        lat=lats_target_nodes,
        lon=lons_target_nodes,
        mode="markers",
        hoverinfo="text",
        name=target_name,
        text=node_text,
        marker={
            "showscale": True,
            "colorscale": "YlGnBu",
            "reversescale": True,
            "color": list(node_adjacencies),
            "size": 10,
            "colorbar": {"thickness": 15, "title": "Node Connections", "xanchor": "left", "titleside": "right"},
            "line_width": 2,
        },
    )
    layout = go.Layout(
        title="<br>" + f"Graph {source_name} --> {target_name}",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=[edge_trace, source_node_trace, target_node_trace], layout=layout)
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()

    """Plot nodes.

    This method creates an interactive visualization of a set of nodes.

    Parameters
    ----------
    graph : HeteroData
        Graph.
    nodes_name : str
        Name of the nodes to plot.
    out_file : str, optional
        Name of the file to save the plot. Default is None.
    """
    node_latitudes, node_longitudes = node_list(graph, nodes_name)
    node_attrs = graph[nodes_name].node_attrs()
    # Remove x to avoid plotting the coordinates as an attribute
    node_attrs.remove("x")

    if len(node_attrs) == 0:
        LOGGER.warning(f"No node attributes found for {nodes_name} nodes.")
        return

    node_traces = {}
    for node_attr in node_attrs:
        node_attr_values = graph[nodes_name][node_attr].float().numpy()

        # Skip multi-dimensional attributes. Supported only: (N, 1) or (N,) tensors
        if node_attr_values.ndim > 1 and node_attr_values.shape[1] > 1:
            continue

        node_traces[node_attr] = go.Scattergeo(
            lat=node_latitudes,
            lon=node_longitudes,
            name=" ".join(node_attr.split("_")).capitalize(),
            mode="markers",
            hoverinfo="text",
            marker={
                "color": node_attr_values.squeeze().tolist(),
                "showscale": True,
                "colorscale": "RdBu",
                "colorbar": {"thickness": 15, "title": node_attr, "xanchor": "left"},
                "size": 5,
            },
            visible=False,
        )

    # Create and add slider
    slider_steps = []
    for i, node_attr in enumerate(node_traces.keys()):
        step = dict(
            label=f"Node attribute: {node_attr}",
            method="update",
            args=[{"visible": [False] * len(node_traces)}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        slider_steps.append(step)

    fig = go.Figure(
        data=list(node_traces.values()),
        layout=go.Layout(
            title=f"<br>Map of {nodes_name} nodes",
            sliders=[
                dict(active=0, currentvalue={"visible": False}, len=0.4, x=0.5, xanchor="center", steps=slider_steps)
            ],
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
            annotations=[annotations_style],
            xaxis=plotly_axis_config,
            yaxis=plotly_axis_config,
        ),
    )
    fig.data[0].visible = True

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()