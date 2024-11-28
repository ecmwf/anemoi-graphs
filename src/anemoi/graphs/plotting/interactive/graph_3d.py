from typing import List
from typing import Tuple

import plotly.graph_objects as go
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx

from anemoi.graphs.plotting.prepare import convert_and_plot_nodes
from anemoi.graphs.plotting.prepare import generate_shades
from anemoi.graphs.plotting.prepare import get_edge_trace
from anemoi.graphs.plotting.prepare import make_layout


def plot_downscale(
    data_nodes,
    hidden_nodes,
    data_to_hidden_edges,
    downscale_edges,
    title=None,
    color="red",
    num_hidden=1,
    x_range=[-1, 1],
    y_range=[-1, 1],
    z_range=[-1, 1],
):
    """Plot all downscaling layers of a graph. Plots the encoder and the processor's downscaling layers if present.

    This method creates an interactive visualization of a set of nodes and edges.

    Parameters
    ----------
    data_nodes : tuple[list, list]
        List of nodes from the data lat lon mesh.
    hidden_nodes : tuple[list, list]
        List of nodes from the hidden mesh.
    data_to_hidden_edges :
        Edges from the lat lon mesh to the hidden mesh
    downscale_edges :
        Downscaling edges of the processor.
    title : str, optional
        Name of the plot. Default is None.
    color : str, optional
        Color of the plot
    num_hidden : int, optional
        Number of hidden layers of the graph. Default is 1.
    x_range : tuple[list, list], optional
        Range of the x coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    y_range : tuple[list, list], optional
        Range of the y coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    z_range : tuple[list, list], optional
        Range of the z coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1 / (num_hidden + 1)

    # Data
    g_data = to_networkx(
        torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges), node_attrs=["x"], edge_attrs=[]
    )

    # Hidden
    graphs = []
    for i in range(0, len(downscale_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[i], edge_index=downscale_edges[i]),
                node_attrs=["x"],
                edge_attrs=[],
            )
        )

    # Node trace
    node_trace_data, _, coords_data = convert_and_plot_nodes(
        g_data, data_nodes, x_range, y_range, z_range, scale=1.0, color="darkgrey"
    )
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []

    for i in range(max(num_hidden, 1)):
        trace, g, tmp_coords = convert_and_plot_nodes(
            graphs[i],
            hidden_nodes[i],
            x_range,
            y_range,
            z_range,
            scale=1.0 - (scale_increment * (i + 1)),
            color="skyblue",
        )
        node_trace_hidden.append(trace)
        graph_processed.append(g)
        coords_hidden.append(tmp_coords)
    node_trace_hidden = sum([node_trace_hidden], [])

    # Edge traces
    edge_traces = [
        get_edge_trace(
            g_data,
            graphs[0],
            coords_data,
            coords_hidden[0],
            1.0,
            1.0 - scale_increment,
            colorscale[i],
            x_range,
            y_range,
            z_range,
        )
    ]
    for i in range(0, num_hidden - 1):
        edge_traces.append(
            get_edge_trace(
                graphs[i],
                graphs[i + 1],
                coords_hidden[i],
                coords_hidden[i + 1],
                1.0 - (scale_increment * (i + 1)),
                1.0 - (scale_increment * (i + 2)),
                colorscale[i],
                x_range,
                y_range,
                z_range,
            )
        )

    edge_traces = sum(edge_traces, [])

    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    return fig


def plot_upscale(
    data_nodes,
    hidden_nodes,
    data_to_hidden_edges,
    upscale_edges,
    title=None,
    color="red",
    num_hidden=1,
    x_range=[-1, 1],
    y_range=[-1, 1],
    z_range=[-1, 1],
):
    """Plot all upscaling layers of a graph. Plots the decoder and the processor's upscaling layers if present.

    This method creates an interactive visualization of a set of nodes and edges.

    Parameters
    ----------
    data_nodes : tuple[list, list]
        List of nodes from the data lat lon mesh.
    hidden_nodes : tuple[list, list]
        List of nodes from the hidden mesh.
    data_to_hidden_edges :
        Edges from the lat lon mesh to the hidden mesh
    hidden_edges :
        Edges connecting the hidden mesh nodes.
    title : str, optional
        Name of the plot. Default is None.
    color : str, optional
        Color of the plot
    num_hidden : int, optional
        Number of hidden layers of the graph. Default is 1.
    x_range : tuple[list, list], optional
        Range of the x coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    y_range : tuple[list, list], optional
        Range of the y coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    z_range : tuple[list, list], optional
        Range of the z coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1 / (num_hidden + 1)

    # Hidden
    graphs = []
    for i in range(0, len(upscale_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[len(upscale_edges) - 1 - i], edge_index=upscale_edges[i]),
                node_attrs=["x"],
                edge_attrs=[],
            )
        )

    # Data
    g_data = to_networkx(
        torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges), node_attrs=["x"], edge_attrs=[]
    )

    # Node trace
    node_trace_data, _, coords_data = convert_and_plot_nodes(
        g_data, data_nodes, x_range, y_range, z_range, scale=1.0, color="darkgrey"
    )
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(
            graphs[i],
            hidden_nodes[len(upscale_edges) - 1 - i],
            x_range,
            y_range,
            z_range,
            scale=1 - ((num_hidden) * scale_increment) + (scale_increment * (i)),
            color="skyblue",
        )
        node_trace_hidden.append(trace)
        graph_processed.append(g)
        coords_hidden.append(tmp_coords)
    node_trace_hidden = sum([node_trace_hidden], [])

    # Edge traces
    edge_traces = []
    for i in range(0, len(graphs) - 1):
        edge_traces.append(
            get_edge_trace(
                graphs[i],
                graphs[i + 1],
                coords_hidden[i],
                coords_hidden[i + 1],
                1 - ((len(graphs) - i) * scale_increment),
                1 - ((len(graphs) - i - 1) * scale_increment),
                colorscale[-1 - i],
                x_range,
                y_range,
                z_range,
            )
        )

    edge_traces.append(
        get_edge_trace(
            graphs[-1],
            g_data,
            coords_hidden[-1],
            coords_data,
            1 - scale_increment,
            1.0,
            colorscale[-1 - i],
            x_range,
            y_range,
            z_range,
        )
    )

    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    return fig


def plot_level(
    data_nodes,
    hidden_nodes,
    data_to_hidden_edges,
    hidden_edges,
    title=None,
    color="red",
    num_hidden=1,
    x_range=[-1, 1],
    y_range=[-1, 1],
    z_range=[-1, 1],
):
    """Plot all hidden layers of a graph and the internal connections between its nodes.

    This method creates an interactive visualization of a set of nodes and edges.

    Parameters
    ----------
    data_nodes : tuple[list, list]
        List of nodes from the data lat lon mesh.
    hidden_nodes : tuple[list, list]
        List of nodes from the hidden mesh.
    data_to_hidden_edges :
        Edges from the lat lon mesh to the hidden mesh
    hidden_edges :
        Edges connecting the hidden mesh nodes.
    title : str, optional
        Name of the plot. Default is None.
    color : str, optional
        Color of the plot
    num_hidden : int, optional
        Number of hidden layers of the graph. Default is 1.
    x_range : tuple[list, list], optional
        Range of the x coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    y_range : tuple[list, list], optional
        Range of the y coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    z_range : tuple[list, list], optional
        Range of the z coordinates for nodes to be shown. Decrease for memory issues. default = [-1, 1]
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1 / (num_hidden + 1)

    # Data
    g_data = to_networkx(
        torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges), node_attrs=["x"], edge_attrs=[]
    )

    # Hidden
    graphs = []
    for i in range(0, len(hidden_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[i], edge_index=hidden_edges[i]),
                node_attrs=["x"],
                edge_attrs=[],
            )
        )

    # Node trace
    node_trace_data, _, _ = convert_and_plot_nodes(
        g_data, data_nodes, x_range, y_range, z_range, scale=1.0, color="darkgrey"
    )
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(
            graphs[i],
            hidden_nodes[i],
            x_range,
            y_range,
            z_range,
            scale=1.0 - (scale_increment * (i + 1)),
            color="skyblue",
        )
        node_trace_hidden.append(trace)
        graph_processed.append(g)
        coords_hidden.append(tmp_coords)
    node_trace_hidden = sum([node_trace_hidden], [])

    # Edge traces
    edge_traces = []
    for i in range(0, len(graphs)):
        edge_traces.append(
            get_edge_trace(
                graphs[i],
                graphs[i],
                coords_hidden[i],
                coords_hidden[i],
                1.0 - (scale_increment * (i + 1)),
                1.0 - (scale_increment * (i + 1)),
                colorscale[i],
                x_range,
                y_range,
                z_range,
            )
        )

    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)

    return fig


def plot_3d_graph(
    graph: HeteroData, nodes_coord: Tuple[List[float], List[float]], title: str = None, show_edges: bool = True
):
    """Plot a graph with his nodes and edges.
    This method creates an interactive visualization of a set of nodes and edges.
    Parameters
    ----------
    graph : HeteroData
        Graph.
    nodes_coord : tuple[list[float], list[float]]
       Coordinates of nodes to plot.
    title : str, optional
        Name of the plot. Default is None.
    show_edges : bool, optional
        Toggle to show edges between nodes too. Default is True.
    """

    # Create a layout for the plot
    layout = make_layout(title)

    # Assuming the node features contain latitude and longitude
    latitudes = nodes_coord[:, 0].numpy()  # Latitude
    longitudes = nodes_coord[:, 1].numpy()  # Longitude

    # Plot points
    node_trace, G, x_nodes, y_nodes, z_nodes = convert_and_plot_nodes(graph, latitudes, longitudes)

    # Plot edges
    if show_edges:
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            # Convert edge nodes to their new indices
            idx0, idx1 = edge[0], edge[1]

            if idx0 in G.nodes and idx1 in G.nodes:
                x_edge = [x_nodes[idx0], x_nodes[idx1], None]
                y_edge = [y_nodes[idx0], y_nodes[idx1], None]
                z_edge = [z_nodes[idx0], z_nodes[idx1], None]
                edge_trace = go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line=dict(width=2, color="red"),
                    showlegend=False,
                    hoverinfo="none",
                )
                edge_traces.append(edge_trace)

        # Combine traces and layout into a figure
        fig = go.Figure(data=edge_traces + [node_trace], layout=layout)

    else:
        fig = go.Figure(data=node_trace, layout=layout)

    # Show the plot
    return fig
