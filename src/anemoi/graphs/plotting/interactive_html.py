import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.prepare import compute_isolated_nodes
from anemoi.graphs.plotting.prepare import compute_node_adjacencies
from anemoi.graphs.plotting.prepare import edge_list
from anemoi.graphs.plotting.prepare import node_list

annotations_style = {"text": "", "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.005, "y": -0.002}
plotly_axis_config = {"showgrid": False, "zeroline": False, "showticklabels": False}

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
    dst_num_nodes = graph[target_name].num_nodes
    node_adjacencies, node_text = compute_node_adjacencies(graph, source_name, target_name, dst_num_nodes)

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
            "color": node_adjacencies,
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


def plot_isolated_nodes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Plot isolated nodes.

    This method creates an interactive visualization of the isolated nodes in the graph.

    Parameters
    ----------
    graph : AnemoiGraph
        The graph to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    isolated_nodes = compute_isolated_nodes(graph)

    if len(isolated_nodes) == 0:
        LOGGER.warning("No orphan nodes found.")
        return

    colorbar = plt.cm.rainbow(np.linspace(0, 1, len(isolated_nodes)))
    nodes = []
    for name, (lat, lon) in isolated_nodes.items():
        nodes.append(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                hoverinfo="text",
                name=name,
                marker={"showscale": False, "color": colorbar[len(nodes)], "size": 10},
            ),
        )

    layout = go.Layout(
        title="<br>Orphan nodes",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=nodes, layout=layout)
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()


def plot_interactive_nodes(
    graph: HeteroData, nodes_name: str, mask: np.ndarray = None, out_file: Optional[str] = None
) -> None:
    """Plot nodes.

    This method creates an interactive visualization of a set of nodes.

    Parameters
    ----------
    graph : HeteroData
        Graph.
    nodes_name : str
        Name of the nodes to plot.
    mask : np.ndarray
        Array of boolean values to mask the nodes.
    out_file : str, optional
        Name of the file to save the plot. Default is None.
    """
    node_latitudes = graph[nodes_name].x[:, 0].numpy()
    node_longitudes = graph[nodes_name].x[:, 1].numpy()

    if mask is None:
        mask = np.ones_like(node_latitudes, dtype=bool)

    colors = ["blue" if m else "red" for m in mask]

    node_trace = go.Scattergeo(
        lat=node_latitudes,
        lon=node_longitudes,
        mode="markers",
        hoverinfo="text",
        marker={"color": colors, "size": 5},
    )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=f"<br>Map of {nodes_name} nodes",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
            annotations=[annotations_style],
            xaxis=plotly_axis_config,
            yaxis=plotly_axis_config,
        ),
    )

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()
