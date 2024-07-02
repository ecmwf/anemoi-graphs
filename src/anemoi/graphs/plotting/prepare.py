from typing import Optional

import numpy as np
from torch_geometric.data import HeteroData


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


def _get_node_attribute_dims(graph: HeteroData) -> dict[str, int]:
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
            if attr == "x":
                continue
            elif attr not in attr_dims:
                attr_dims[attr] = nodes[attr].shape[1]
            else:
                assert (
                    nodes[attr].shape[1] == attr_dims[attr]
                ), f"Attribute {attr} has different dimensions in different nodes."
    return attr_dims


def _get_edge_attribute_dims(graph: HeteroData) -> dict[str, int]:
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
