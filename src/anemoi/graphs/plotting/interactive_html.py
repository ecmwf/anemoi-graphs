import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.prepare import compute_node_adjacencies
from anemoi.graphs.plotting.prepare import edge_list
from anemoi.graphs.plotting.prepare import node_list

annotations_style = {"text": "", "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.005, "y": -0.002}
plotly_axis_config = {"showgrid": False, "zeroline": False, "showticklabels": False}

logger = logging.getLogger(__name__)


def plot_interactive_subgraph(
    graph: HeteroData,
    edges_to_plot: tuple[str, str],
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
    src_nodes, dst_nodes = edges_to_plot
    edge_x, edge_y = edge_list(graph, edges_to_plot)
    assert src_nodes in graph.node_types, f"edges_to_plot ({src_nodes}) should be in the graph"
    assert dst_nodes in graph.node_types, f"edges_to_plot ({dst_nodes}) should be in the graph"
    lats_nodes1, lons_nodes1 = node_list(graph, src_nodes)
    lats_nodes2, lons_nodes2 = node_list(graph, dst_nodes)
    dst_num_nodes = graph[dst_nodes].num_nodes
    node_adjacencies, node_text = compute_node_adjacencies(graph, edges_to_plot[0], edges_to_plot[1], dst_num_nodes)

    edge_trace = go.Scattergeo(
        lat=edge_x,
        lon=edge_y,
        line={"width": 0.5, "color": "#888"},
        hoverinfo="none",
        mode="lines",
        name="Connections",
    )

    node_trace1 = go.Scattergeo(
        lat=lats_nodes1,
        lon=lons_nodes1,
        mode="markers",
        hoverinfo="text",
        name=edges_to_plot[0],
        marker={
            "showscale": False,
            "color": "red",
            "size": 2,
            "line_width": 2,
        },
    )

    node_trace2 = go.Scattergeo(
        lat=lats_nodes2,
        lon=lons_nodes2,
        mode="markers",
        hoverinfo="text",
        name=edges_to_plot[1],
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
        title="<br>" + f"Graph {edges_to_plot[0]} --> {edges_to_plot[1]}",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=[edge_trace, node_trace1, node_trace2], layout=layout)
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
    isolated = {}
    for (src_nodes, _, dst_nodes), sub_graph in graph.edge_items():
        head_isolated = np.ones(graph[src_nodes].num_nodes, dtype=bool)
        tail_isolated = np.ones(graph[dst_nodes].num_nodes, dtype=bool)
        head_isolated[sub_graph.edge_index[0]] = False
        tail_isolated[sub_graph.edge_index[1]] = False
        if np.any(head_isolated):
            isolated[f"{src_nodes} isolated (--> {dst_nodes})"] = node_list(graph, src_nodes, mask=list(head_isolated))
        if np.any(tail_isolated):
            isolated[f"{dst_nodes} isolated ({src_nodes} -->)"] = node_list(graph, dst_nodes, mask=list(tail_isolated))

    if len(isolated) == 0:
        logger.info("No orphan nodes found.")
        return

    colorbar = plt.cm.rainbow(np.linspace(0, 1, len(isolated)))
    nodes = []
    for name, (lat, lon) in isolated.items():
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
    title: str, lats: np.ndarray, lons: np.ndarray, mask: np.ndarray = None, out_file: Optional[str] = None
) -> None:
    """Plot nodes.

    This method creates an interactive visualization of a set of nodes.

    Parameters
    ----------
    title : str
        The title of the graph.
    lats : np.ndarray
        Array of latitudes.
    lons : np.ndarray
        Array of longitudes.
    mask : np.ndarray
        Array of boolean values to mask the nodes.
    out_file : str, optional
        Name of the file to save the plot. Default is None.
    """
    if mask is None:
        mask = np.ones_like(lats, dtype=bool)

    colors = ["blue" if m else "red" for m in mask]

    node_trace = go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        hoverinfo="text",
        marker={"color": colors, "size": 5},
    )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title="<br>" + title,
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
