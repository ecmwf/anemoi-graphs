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
        LOGGER.warning("No isolated nodes found.")
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


def plot_interactive_nodes(graph: HeteroData, nodes_name: str, out_file: Optional[str] = None) -> None:
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


def plot_downscale(data_nodes, hidden_nodes, data_to_hidden_edges, downscale_edges, title=None, color='red', num_hidden=1, filter_limit=0.4):
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
    filter_limit : float, optional
        Percentage of first quadrant nodes to be shown. Decrease for memory issues. Default is 0.4.
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1/(num_hidden+1)

    # Data
    g_data = to_networkx(
                torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges),
                node_attrs=['x'],
                edge_attrs=[]
            )

    # Hidden
    graphs = []
    for i in range(0, len(downscale_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[i], edge_index=downscale_edges[i]),
                node_attrs=['x'],
                edge_attrs=[]
            )
        )

    # Node trace
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, filter_limit=filter_limit, scale=1.0, color='darkgrey')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    
    for i in range(max(num_hidden, 1)):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[i], filter_limit=filter_limit, scale=1.0-(scale_increment*(i+1)), color='skyblue')
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
            1.0, 1.0-scale_increment, 
            'yellowgreen', 
            filter_limit=filter_limit
            )
    ]
    for i in range(0, num_hidden-1):
        edge_traces.append(
            get_edge_trace(
                graphs[i], 
                graphs[i+1], 
                coords_hidden[i], 
                coords_hidden[i+1], 
                1.0-(scale_increment*(i+1)), 1.0-(scale_increment*(i+2)), 
                colorscale[i], 
                filter_limit=filter_limit
            )
        )

    edge_traces = sum(edge_traces, [])

    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()


def plot_upscale(data_nodes, hidden_nodes, data_to_hidden_edges, upscale_edges, title=None,  color='red', num_hidden=1, filter_limit=0.4):
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
    filter_limit : float, optional
        Percentage of first quadrant nodes to be shown. Decrease for memory issues. Default is 0.4.
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1/(num_hidden+1)

    # Hidden
    graphs = []
    for i in range(0, len(upscale_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[len(upscale_edges)-1-i], edge_index=upscale_edges[i]),
                node_attrs=['x'],
                edge_attrs=[]
            )
        )
    
    # Data
    g_data = to_networkx(
                torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges),
                node_attrs=['x'],
                edge_attrs=[]
            )

    # Node trace
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, filter_limit=filter_limit, scale=1.0, color='darkgrey')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[len(upscale_edges)-1-i], filter_limit=filter_limit, scale=1-((num_hidden)*scale_increment) + (scale_increment*(i)), color='skyblue')
        node_trace_hidden.append(trace)
        graph_processed.append(g)
        coords_hidden.append(tmp_coords)
    node_trace_hidden = sum([node_trace_hidden], [])
    
    # Edge traces
    edge_traces = []
    for i in range(0, len(graphs)-1):
        edge_traces.append(
            get_edge_trace(
                graphs[i], 
                graphs[i+1], 
                coords_hidden[i], 
                coords_hidden[i+1], 
                1-((len(graphs)-i)*scale_increment), 1-((len(graphs)-i-1)*scale_increment),
                colorscale[-1-i],
                filter_limit=filter_limit
            )
        )

    edge_traces.append(
        get_edge_trace(
            graphs[-1], 
            g_data, 
            coords_hidden[-1], 
            coords_data, 
            1-scale_increment, 1.0, 
            'yellowgreen',
            filter_limit=filter_limit
            )
    )
 

    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()


def plot_level(data_nodes, hidden_nodes, data_to_hidden_edges, hidden_edges, title=None, color='red', num_hidden=1, filter_limit=0.4):
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
    filter_limit : float, optional
        Percentage of first quadrant nodes to be shown. Decrease for memory issues. Default is 0.4.
    """
    colorscale = generate_shades(color, num_hidden)
    layout = make_layout(title)
    scale_increment = 1/(num_hidden+1)


    # Data
    g_data = to_networkx(
                torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges),
                node_attrs=['x'],
                edge_attrs=[]
            )

    # Hidden
    graphs = []
    for i in range(0, len(hidden_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[i], edge_index=hidden_edges[i]),
                node_attrs=['x'],
                edge_attrs=[]
            )
        )
    

    # Node trace
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, filter_limit=filter_limit, scale=1.0, color='darkgrey')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[i], filter_limit=filter_limit, scale=1.0-(scale_increment*(i+1)), color='skyblue')
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
                1.0-(scale_increment*(i+1)),  1.0-(scale_increment*(i+1)), 
                colorscale[i], 
                filter_limit=filter_limit
            )
        )
 
    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()


def plot_3d_graph(graph: HeteroData, nodes_coord: tuple[list[float], list[float]], title: str = None, show_edges: bool = True):
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
    longitudes = nodes_coord[:, 1].numpy() # Longitude

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
                    x=x_edge, y=y_edge, z=z_edge,
                    mode='lines',
                    line=dict(width=2, color='red'),
                    showlegend=False
                )
                edge_traces.append(edge_trace)

        # Combine traces and layout into a figure
        fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    
    else:
        fig = go.Figure(data=node_trace, layout=layout)

    # Show the plot
    fig.show()