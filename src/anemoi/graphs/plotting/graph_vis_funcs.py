import hydra
from omegaconf import DictConfig
from aifs.utils.logger import get_code_logger
from print_color import print
import plotly.graph_objs as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import to_networkx
import plotly.graph_objs as go
import plotly.io as pio
import torch_geometric
import numpy as np
from tqdm import tqdm
LOGGER = get_code_logger(__name__)

def get_edge_trace(g1, g2, n1, n2, scale_1, scale_2, color, filter_limit=0.5):
    """
    Gets all edges between g1 and g2 (two separate graphs, hierarchical graph setting).
    """
    edge_traces = []
    for edge in g1.edges():
        # Convert edge nodes to their new indices
        idx0, idx1 = edge[0], edge[1]

        if idx0 in g1.nodes and idx1 in g2.nodes:
            if n1[0][idx0] > filter_limit and n2[0][idx1] > filter_limit and \
               n1[1][idx0] > filter_limit and n2[1][idx1] > filter_limit and \
               n1[2][idx0] > filter_limit and n2[2][idx1] > filter_limit:
                x_edge = [n1[0][idx0] * scale_1, n2[0][idx1] * scale_2, None]
                y_edge = [n1[1][idx0] * scale_1, n2[1][idx1] * scale_2, None]
                z_edge = [n1[2][idx0] * scale_1, n2[2][idx1] * scale_2, None]
                edge_trace = go.Scatter3d(
                    x=x_edge, y=y_edge, z=z_edge,
                    mode='lines',
                    line=dict(width=2, color=color),
                    showlegend=False
                )
                edge_traces.append(edge_trace)
    return edge_traces

def convert_and_plot_nodes(G, coords, rads=True, filter=False, filter_limit=0, scale=1.0, color='skyblue'):
    """
    Filters coordinates of nodes in a graph, scales and plots them.
    """

    lat = coords[:, 0].numpy()  # Latitude
    lon = coords[:, 1].numpy() # Longitude
    
    # Convert lat/lon to Cartesian coordinates for the filtered nodes
    x_nodes, y_nodes, z_nodes = lat_lon_to_cartesian(lat, lon, rads=rads)

    # Filter nodes for the first quadrant
    if filter:
        first_quadrant_nodes = [i for i, (x, y, z) in enumerate(zip(x_nodes, y_nodes, z_nodes)) if x > filter_limit and y > filter_limit and z > filter_limit]

        if not first_quadrant_nodes:
            print("No nodes found in the first quadrant.")
            return

        graph = G.subgraph(first_quadrant_nodes).copy()

    else:
        graph = G
        
    # Extract node positions for Plotly
    x_nodes_filtered = [x_nodes[node] * scale for node in graph.nodes()]
    y_nodes_filtered = [y_nodes[node] * scale for node in graph.nodes()]
    z_nodes_filtered = [z_nodes[node] * scale for node in graph.nodes()]

    # Create traces for nodes
    node_trace = go.Scatter3d(
        x=x_nodes_filtered, y=y_nodes_filtered, z=z_nodes_filtered,
        mode='markers',
        marker=dict(size=3, color=color, opacity=0.8),
        text=list(graph.nodes()),
        hoverinfo='text'
    )

    return  node_trace, graph, (x_nodes, y_nodes, z_nodes)

def lat_lon_to_cartesian(lat, lon, rads=True):
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    if not rads:
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
    else:
        lat_rad = lat
        lon_rad = lon
        
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return x, y, z

def plot_3d_graph(G, nodes_coord, title, show_edges=True, filter=False):

    # Create a layout for the plot
    layout = go.Layout(
       title={
        'text': title,
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Anchor the title to the center of the plot area
        'y': 0.95,  # Position the title vertically
        'yanchor': 'top'  # Anchor the title to the top of the plot area
        },
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        width=1000,  # Increase width
        height=1000,  # Increase height
        showlegend=False
    )

    # Assuming the node features contain latitude and longitude
    latitudes = nodes_coord[:, 0].numpy()  # Latitude
    longitudes = nodes_coord[:, 1].numpy() # Longitude

    # Plot points
    node_trace, graph, x_nodes, y_nodes, z_nodes = convert_and_plot_nodes(G, latitudes, longitudes)

    # Plot edges
    if show_edges:
        # Create edge traces
        edge_traces = []
        for edge in graph.edges():
            # Convert edge nodes to their new indices
            idx0, idx1 = edge[0], edge[1]

            if idx0 in graph.nodes and idx1 in graph.nodes:
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

def generate_shades(color_name, num_shades):
    # Get the base color from the name
    base_color = mcolors.CSS4_COLORS.get(color_name.lower(), None)
    
    if not base_color:
        raise ValueError(f"Color '{color_name}' is not recognized.")
    
    # Convert the base color to RGB
    base_rgb = mcolors.hex2color(base_color)
    
    # Create a colormap that transitions from the base color to a darker version of the base color
    dark_color = tuple([x * 0.6 for x in base_rgb])  # Darker shade of the base color
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap', [base_rgb, dark_color], N=num_shades
    )
    
    # Generate the shades
    shades = [mcolors.to_hex(cmap(i / (num_shades - 1))) for i in range(num_shades)]
    
    return shades

def vis_downscale(data_nodes, hidden_nodes, data_to_hidden_edges, downscale_edges, layout, color='red', num_hidden=3, filter_limit=0.4):

    colorscale = generate_shades(color, num_hidden)

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
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, rads=True, filter=filter, filter_limit=filter_limit, scale=1.0, color='skyblue')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[i], rads=True, filter=filter, filter_limit=filter_limit, scale=1.0-(0.2*(i+1)), color='skyblue')
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
            1.0, 1.0-(0.2*1), 
            colorscale[0], 
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
                1.0-(0.2*(i+1)), 1.0-(0.2*(i+2)), 
                colorscale[i+1], 
                filter_limit=filter_limit
            )
        )

    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()

def vis_upscale(data_nodes, hidden_nodes, data_to_hidden_edges, upscale_edges, layout,  color='red', num_hidden=3, filter_limit=0.4):

    colorscale = generate_shades(color, num_hidden)

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
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, rads=True, filter=filter, filter_limit=filter_limit, scale=1.0, color='skyblue')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[len(upscale_edges)-1-i], rads=True, filter=filter, filter_limit=filter_limit, scale=1-((num_hidden)*0.2) + (0.2*(i)), color='skyblue')
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
                1-((len(graphs)-i)*0.2), 1-((len(graphs)-i-1)*0.2), 
                colorscale[i],
                filter_limit=filter_limit
            )
        )

    edge_traces.append(
        get_edge_trace(
            graphs[-1], 
            g_data, 
            coords_hidden[-1], 
            coords_data, 
            0.8, 1.0, 
            colorscale[-1],
            filter_limit=filter_limit
            )
    )
 

    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()

def vis_level(data_nodes, hidden_nodes, data_to_hidden_edges, hidden_edges, layout, color='red', num_hidden=3, filter_limit=0.4):

    colorscale = generate_shades(color, num_hidden)

    # Data
    g_data = to_networkx(
                torch_geometric.data.Data(x=data_nodes, edge_index=data_to_hidden_edges),
                node_attrs=['x'],
                edge_attrs=[]
            )

    # Hidden
    graphs = []
    for i in range(0, len(upscale_edges)):
        graphs.append(
            to_networkx(
                torch_geometric.data.Data(x=hidden_nodes[i], edge_index=hidden_edges[i]),
                node_attrs=['x'],
                edge_attrs=[]
            )
        )
    

    # Node trace
    node_trace_data, graph_data, coords_data = convert_and_plot_nodes(g_data, data_nodes, rads=True, filter=filter, filter_limit=filter_limit, scale=1.0, color='skyblue')
    node_trace_hidden = [node_trace_data]
    graph_processed = []
    coords_hidden = []
    for i in range(num_hidden):
        trace, g, tmp_coords = convert_and_plot_nodes(graphs[i], hidden_nodes[i], rads=True, filter=filter, filter_limit=filter_limit, scale=1.0-(0.2*(i+1)), color='skyblue')
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
                1.0-(0.2*(i+1)),  1.0-(0.2*(i+1)), 
                colorscale[i], 
                filter_limit=filter_limit
            )
        )
 
    edge_traces = sum(edge_traces, [])
    # Combine traces and layout into a figure
    fig = go.Figure(data=node_trace_hidden + edge_traces, layout=layout)
    fig.show()
    return

@hydra.main(version_base=None, config_path="aifs/config", config_name="graph_factory")
def main(config: DictConfig) -> None:
    from anemoi.graphs.creators import GraphCreator
    GraphCreator(config=config.graph).create(save_path='/scratch/mch/apennino/output/graphs/hierarchical_1.pt', overwrite=True)

if __name__ == "__main__":
    main()