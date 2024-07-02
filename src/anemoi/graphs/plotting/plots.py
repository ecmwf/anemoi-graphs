import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

annotations_style = {"text": "", "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.005, "y": -0.002}
plotly_axis_config = {"showgrid": False, "zeroline": False, "showticklabels": False}


def get_node_names(edge_src: str, edge_dst: str) -> tuple[str, str]:
    """Get the names of the nodes.

    This method returns the names of the nodes from the edge.

    Parameters
    ----------
    edge_src : str
        Name of the source nodes. If the mapping is done with a filter,
        the name is composed of the source node and the filter.
    edge_dst : str
        Name of the destination node. If the mapping is done with a filter,
        the name is composed of the destination node and the filter.

    Returns
    -------
    src_nodes_name : str
        Name of the source node.
    dst_nodes_name : str
        Name of the destination node.
    """
    src_nodes_name = edge_src.split("-")[0] if "-" in edge_src else edge_src
    dst_nodes_name = edge_dst.split("-")[0] if "-" in edge_dst else edge_dst
    return src_nodes_name, dst_nodes_name


def node_list(graph: HeteroData, coord_dim: str, mask: Optional[list[bool]] = None) -> tuple[list[float], list[float]]:
    """Get the latitude and longitude of the nodes.

    Parameters
    ----------
    graph : dict[str, torch.Tensor]
        Graph to plot.
    coord_dim : str
        Name of the dimension with the graph's nodes coordinates.
    mask : list[bool], optional
        Mask to filter the nodes. Default is None.

    Returns
    -------
    latitudes : list[float]
        Latitude coordinates of the nodes.
    longitudes : list[float]
        Longitude coordinates of the nodes.
    """
    coords = np.rad2deg(graph[coord_dim].x.numpy())
    latitudes = coords[:, 0]
    longitudes = coords[:, 1]
    if mask is not None:
        latitudes = latitudes[mask]
        longitudes = longitudes[mask]
    return latitudes.tolist(), longitudes.tolist()


def edge_list(graph: HeteroData, plt_ids: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Get the edge list.

    This method returns the edge list to be represented in a graph. It computes the coordinates of the points connected
    and include NaNs to separate the edges.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    plt_ids : tuple[str, str]
        Names of the input and output dimensions to plot.

    Returns
    -------
    latitudes : np.ndarray
        Latitude coordinates of the points connected.
    longitudes : np.ndarray
        Longitude coordinates of the points connected.
    """
    src_nodes, dst_nodes = plt_ids
    sub_graph = graph[(src_nodes, "to", dst_nodes)].edge_index
    x0 = np.rad2deg(graph[src_nodes].x[sub_graph[0]])
    y0 = np.rad2deg(graph[dst_nodes].x[sub_graph[1]])
    nans = np.full_like(x0[:, :1], np.nan)
    latitudes = np.concatenate([x0[:, :1], y0[:, :1], nans], axis=1).flatten()
    longitudes = np.concatenate([x0[:, 1:2], y0[:, 1:2], nans], axis=1).flatten()
    return latitudes, longitudes


def compute_node_adjacencies(
    graph: HeteroData, edges_from: str, edges_to: str, num_nodes: int
) -> tuple[list[int], list[str]]:
    """Compute the number of adjacencies of each node in a bipartite graph.

    Parameters
    ----------
    graph : HeteroData
        Graph to plot.
    edges_from : str
        Name of the dimension of the coordinates for the head nodes.
    edges_to : str
        Name of the dimension of the coordinates for the tail nodes.
    num_nodes : int
        Number of nodes in the destination (including the not connected).

    Returns
    -------
    num_adjacencies : list[int]
        Number of adjacencies of each node.
    node_text : list[str]
        Text to show when hovering over the nodes.
    """
    node_adjacencies = np.zeros(num_nodes, dtype=int)
    vals, counts = np.unique(graph[(edges_from, "to", edges_to)].edge_index[1], return_counts=True)
    node_adjacencies[vals] = counts
    node_text = [f"# of connections: {x}" for x in node_adjacencies]
    return list(node_adjacencies), node_text


def plot_html_for_subgraph(
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
    dst_num_nodes = graph.num_node_features[dst_nodes]
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


def plot_graph_from_networkx(title: str, graph: nx.Graph, out_file: Optional[Union[str, Path]] = None) -> None:
    """Plot a graph from networkx.Graph.

    This method creates an interactive visualization of a graph object from networkx library.

    Parameters
    ----------
    title : str
        The title of the graph.
    graph : nx.Graph
        The graph to visualize.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = np.rad2deg(graph.nodes[edge[0]]["hcoords_rad"])
        x1, y1 = np.rad2deg(graph.nodes[edge[1]]["hcoords_rad"])
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))

    edge_trace = go.Scattergeo(
        lat=edge_x, lon=edge_y, line={"width": 0.5, "color": "#888"}, hoverinfo="none", mode="lines"
    )

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = np.rad2deg(graph.nodes[node]["hcoords_rad"])
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scattergeo(
        lat=node_x,
        lon=node_y,
        mode="markers",
        hoverinfo="text",
        marker={
            "showscale": True,
            "colorscale": "YlGnBu",
            "reversescale": True,
            "color": [],
            "size": 10,
            "colorbar": {"thickness": 15, "title": "Node Connections", "xanchor": "left", "titleside": "right"},
            "line_width": 2,
        },
    )

    node_adjacencies = []
    node_text = []
    for _, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append("# of connections: " + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
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
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()

    n_h3_nodes0 = 122
    n_h3_nodes1 = 842
    logger.debug(sorted(node_adjacencies, reverse=True)[0:n_h3_nodes0])
    logger.debug(sorted(node_adjacencies, reverse=True)[n_h3_nodes0 : n_h3_nodes0 + n_h3_nodes1])
    logger.debug(sorted(node_adjacencies, reverse=True)[n_h3_nodes0 + n_h3_nodes1 : n_h3_nodes0 + n_h3_nodes1 + 100])


def plot_connection_stats_graphdata(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    _, axs = plt.subplots(1, len(graph.edge_types), figsize=(10 * len(graph.edge_types), 10), sharex=True)
    for i, ((src, _, dst), sub_graph) in enumerate(graph.edge_items()):
        norm_dists = sub_graph["normed_dist"].squeeze()
        axs[i].hist(norm_dists, bins=50)
        axs[i].set_ylabel("Count")
        axs[i].set_xlabel("Edge Weight")
        axs[i].set_title(f"{src} --> {dst}")
    plt.suptitle("Edge weight (1 - L1_norm(edge length))", fontsize=14)
    plt.savefig(out_file)


def plot_orphan_nodes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Plot orphan nodes.

    This method creates an interactive visualization of the orphan nodes in the graph.

    Parameters
    ----------
    graph : AnemoiGraph
        The graph to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    orphans = {}
    for (src_nodes, _, dst_nodes), sub_graph in graph.edge_items():
        head_orphans = np.ones(graph.num_node_features[src_nodes], dtype=bool)
        tail_orphans = np.ones(graph.num_node_features[dst_nodes], dtype=bool)
        head_orphans[sub_graph.edge_index[0]] = False
        tail_orphans[sub_graph.edge_index[1]] = False
        if np.any(head_orphans):
            orphans[f"{src_nodes} orphans (--> {dst_nodes})"] = node_list(graph, src_nodes, mask=list(head_orphans))
        if np.any(tail_orphans):
            orphans[f"{dst_nodes} orphans ({src_nodes} -->)"] = node_list(graph, dst_nodes, mask=list(tail_orphans))

    if len(orphans) == 0:
        print("No orphan nodes found.")
        return

    colorbar = plt.cm.rainbow(np.linspace(0, 1, len(orphans)))
    nodes = []
    for name, (lat, lon) in orphans.items():
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
        print(f"Orphan nodes plot saved to {out_file}.")
    else:
        fig.show()


def plot_nodes(
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
