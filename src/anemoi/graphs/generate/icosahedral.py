from collections.abc import Iterable
from typing import Optional

import networkx as nx
import numpy as np
import trimesh
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree

from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
import logging

logger = logging.getLogger(__name__)


def create_icosahedral_nodes(
    resolutions: list[int],
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh following AIFS strategy.

    This method relies on the trimesh python library.

    Parameters
    ----------
    resolutions : list[int]
        Levels of mesh resolution to consider.
    aoi_mask_builder : KNNAreaMaskBuilder
        KNNAreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (nodes & edges).
    vertices_rad : np.ndarray
        The vertices (not ordered) of the mesh in radians.
    node_ordering : list[int]
        Order of the nodes in the graph to be sorted by latitude and longitude.
    """
    sphere = create_sphere(resolutions[-1])
    coords_rad = cartesian_to_latlon_rad(sphere.vertices)

    node_ordering = get_node_ordering(coords_rad)

    # TODO: AOI mask builder is not used in the current implementation.

    nx_graph = create_icosahedral_nx_graph_from_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def create_icosahedral_nx_graph_from_coords(coords_rad: np.ndarray, node_ordering: list[int]):

    graph = nx.DiGraph()
    for ii, coords in enumerate(coords_rad[node_ordering]):
        node_id = node_ordering[ii]
        graph.add_node(node_id, hcoords_rad=coords)

    assert list(graph.nodes.keys()) == list(node_ordering), "Nodes are not correctly added to the graph."
    assert graph.number_of_nodes() == len(node_ordering), "The number of nodes must be the same."
    return graph


def get_node_ordering(vertices_rad: np.ndarray) -> np.ndarray:
    # Get indices to sort points by lon & lat in radians.
    ind1 = np.argsort(vertices_rad[:, 1])
    ind2 = np.argsort(vertices_rad[ind1][:, 0])[::-1]
    node_ordering = np.arange(vertices_rad.shape[0])[ind1][ind2]
    return node_ordering


def add_edges_to_nx_graph(
    graph: nx.DiGraph,
    resolutions: list[int],
    xhops: int = 1,
) -> None:
    """Adds the edges to the graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to add the edges. It should correspond to the mesh nodes, without edges.
    resolutions : list[int]
        Levels of mesh refinement levels to consider.
    xhops : int, optional
        Number of hops between 2 nodes to consider them neighbours, by default 1.
    aoi_mask_builder : KNNAreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area, by default None.
    margin_radius_km : float, optional
        Margin radius in km to consider when creating the processor mesh, by default 0.0.
    """
    assert xhops > 0, "xhops == 0, graph would have no edges ..."

    sphere = create_sphere(resolutions[-1])
    vertices_rad = cartesian_to_latlon_rad(sphere.vertices)
    x_hops = get_x_hops(sphere, xhops, valid_nodes=list(graph.nodes))

    for i, i_neighbours in x_hops.items():
        add_neigbours_edges(graph, vertices_rad, i, i_neighbours)

    tree = BallTree(vertices_rad, metric="haversine")

    for resolution in resolutions[:-1]:
        # Defined refined sphere
        r_sphere = create_sphere(resolution)
        r_vertices_rad = cartesian_to_latlon_rad(r_sphere.vertices)

        # TODO AOI mask builder is not used in the current implementation.
        valid_nodes = None

        x_rings = get_x_hops(r_sphere, xhops, valid_nodes=valid_nodes)

        _, idx = tree.query(r_vertices_rad, k=1)
        for i, i_neighbours in x_rings.items():
            add_neigbours_edges(graph, r_vertices_rad, i, i_neighbours, idx=idx)

    return graph


def create_sphere(subdivisions: int = 0, radius: float = 1.0) -> trimesh.Trimesh:
    """Creates a sphere.

    Parameters
    ----------
    subdivisions : int, optional
        How many times to subdivide the mesh. Note that the number of faces will grow as function of 4 ** subdivisions.
        Defaults to 0.
    radius : float, optional
        Radius of the sphere created, by default 1.0

    Returns
    -------
    trimesh.Trimesh
        Meshed sphere.
    """
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def get_x_hops(sp: trimesh.Trimesh, hops: int, valid_nodes: Optional[list[int]] = None) -> dict[int, set[int]]:
    """Get the neigbour connections in the graph.

    Parameters
    ----------
    sp : trimesh.Trimesh
        The mesh to consider.
    hops : int
        Number of hops between 2 nodes to consider them neighbours.
    valid_nodes : list[int], optional
        List of valid nodes to consider, by default None. It is useful to consider only a subset of the nodes to save
        computation time.

    Returns
    -------
    neighbours : dict[int, set[int]]
        A list with the neighbours for each vertex. The element at position 'i' correspond to the neighbours to the
        i-th vertex of the mesh.
    """
    edges = sp.edges_unique
    if valid_nodes is not None:
        edges = edges[np.isin(sp.edges_unique, valid_nodes).all(axis=1)]
    else:
        valid_nodes = list(range(len(sp.vertices)))
    g = nx.from_edgelist(edges)

    neighbours = {ii: set(nx.ego_graph(g, ii, radius=hops, center=False) if ii in g else []) for ii in valid_nodes}

    return neighbours


def add_neigbours_edges(
    graph: nx.Graph,
    vertices: np.ndarray,
    ii: int,
    neighbours: Iterable[int],
    self_loops: bool = False,
    idx: Optional[np.ndarray] = None,
) -> None:
    """Adds the edges of one node to its neighbours.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    vertices : np.ndarray
        A 2D array of shape (num_vertices, 2) with the planar coordinates of the mesh, in radians.
    ii : int
        The node considered.
    neighbours : list[int]
        The neighbours of the node.
    self_loops : bool, optional
        Whether is supported to add self-loops, by default False.
    idx : np.ndarray, optional
        Index to map the vertices from the refined sphere to the original one, by default None.
    """
    for ineighb in neighbours:
        if not self_loops and ii == ineighb:  # no self-loops
            continue

        loc_self = vertices[ii]
        loc_neigh = vertices[ineighb]
        edge_length = haversine_distances([loc_neigh, loc_self])[0][1]

        if idx is not None:
            # Use the same method to add edge in all spheres
            node_neigh = idx[ineighb][0]
            node = idx[ii][0]
        else:
            node, node_neigh = ii, ineighb

        # add edge to the graph
        if node in graph and node_neigh in graph:
            graph.add_edge(node_neigh, node, weight=edge_length)





