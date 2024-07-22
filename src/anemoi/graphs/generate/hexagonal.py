from typing import Optional

import h3
import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import haversine_distances


def create_hexagonal_nodes(
    resolutions: list[int],
    flat: bool = True,
    area: Optional[dict] = None,
) -> tuple[nx.Graph, torch.Tensor, list[int]]:
    """Creates a global mesh from a refined icosahedro.

    This method relies on the H3 python library, which covers the earth with hexagons (and 5 pentagons). At each
    refinement level, a hexagon cell has 7 child cells (aperture 7).

    Parameters
    ----------
    resolutions : list[int]
        Levels of mesh resolution to consider.
    flat : bool
        Whether or not all resolution levels of the mesh are included.
    area : dict
        A region, in GeoJSON data format, to be contained by all cells. Defaults to None, which computes the global
        mesh.
    aoi_mask_builder : KNNAreaMaskBuilder, optional
        KNNAreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (nodes & edges).
    """
    graph = nx.Graph()

    area_kwargs = {"area": area}

    for resolution in resolutions:
        add_nodes_for_resolution(graph, resolution, **area_kwargs)

    coords = np.array([h3.h3_to_geo(node) for node in graph.nodes])

    # Sort nodes by latitude and longitude
    node_ordering = np.lexsort(coords.T[::-1], axis=0)

    coords = coords[node_ordering]

    return graph, coords, node_ordering


def add_nodes_for_resolution(
    graph: nx.Graph,
    resolution: int,
    self_loop: bool = False,
    **area_kwargs: Optional[dict],
) -> None:
    """Add all nodes at a specified refinement level to a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the nodes.
    resolution : int
        The H3 refinement level. It can be an integer from 0 to 15.
    self_loop : int
        Whether to include self-loops in the nodes added or not.
    area_kwargs: dict
        Additional arguments to pass to the get_cells_at_resolution function.
    """

    cells = get_cells_at_resolution(resolution, **area_kwargs)

    for idx in cells:
        graph.add_node(idx, hcoords_rad=np.deg2rad(h3.h3_to_geo(idx)))
        if self_loop:
            # TODO: should that be add_self_loops(graph)?
            add_edge(graph, idx, idx, allow_self_loop=self_loop)


def get_cells_at_resolution(
    resolution: int,
    area: Optional[dict] = None,
) -> set[str]:
    """Get cells at a specified refinement level.

    Parameters
    ----------
    resolution : int
        The H3 refinement level. It can be an integer from 0 to 15.
    area : dict
        A region, in GeoJSON data format, to be contained by all cells. Defaults to None.
    aoi_mask_builder : KNNAreaMaskBuilder, optional
        KNNAreaMaskBuilder computes nask to limit the mesh area, by default None.

    Returns
    -------
    cells : set[str]
        The set of H3 indexes at the specified resolution level.
    """
    # TODO: What is area?
    cells = h3.uncompact(h3.get_res0_indexes(), resolution) if area is None else h3.polyfill(area, resolution)

    # TODO: AOI not used in the current implementation.

    return cells


def add_edges_to_nx_graph(
    graph: nx.Graph,
    resolutions: list[int],
    x_hops: int = 1,
    self_loop: bool = False,
    flat: bool = True,
    neighbour_children: bool = False,
    depth_children: int = 1,
) -> nx.Graph:
    """Creates a global mesh from a refined icosahedron.

    This method relies on the H3 python library, which covers the earth with hexagons (and 5 pentagons). At each
    refinement level, a hexagon cell has 7 child cells (aperture 7).

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the nodes.
    resolutions : list[int]
        Levels of mesh resolution to consider.
    xhops: int
        The number of hops to consider for the neighbours.
    self_loop : bool
        Whether include a self-loop in every node or not.
    flat : bool
        Whether or not all resolution levels of the mesh are included.
    neighbour_children : bool
        Whether to include connections with the children from the neighbours.
    depth_children : int
        The number of resolution levels to consider for the connections of children. Defaults to 1, which includes
        connections up to the next resolution level.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (nodes & edges).
    """
    if self_loop:
        add_self_loops(graph)

    add_neighbour_edges(graph, resolutions, x_hops, flat)
    add_children_edges(
        graph,
        resolutions,
        flat,
        neighbour_children,
        depth_children,
    )
    return graph


def add_self_loops(graph: nx.Graph) -> None:

    for idx in graph.nodes:
        add_edge(graph, idx, idx, allow_self_loop=True)


def add_neighbour_edges(
    graph: nx.Graph,
    refinement_levels: tuple[int],
    xhops: int = 1,
    flat: bool = True,
) -> None:
    for resolution in refinement_levels:
        cells = {node for node in graph.nodes if h3.h3_get_resolution(node) == resolution}
        for idx in cells:
            # neighbours
            for idx_neighbour in h3.k_ring(idx, k=xhops) & cells:
                if flat:
                    add_edge(
                        graph,
                        h3.h3_to_center_child(idx, refinement_levels[-1]),
                        h3.h3_to_center_child(idx_neighbour, refinement_levels[-1]),
                    )
                else:
                    add_edge(graph, idx, idx_neighbour)


def add_children_edges(
    graph: nx.Graph,
    refinement_levels: tuple[int],
    flat: bool = True,
    has_neighbour_children: bool = False,
    depth: Optional[int] = None,
) -> None:
    if depth is None:
        depth = len(refinement_levels)

    for i_level, resolution_parent in enumerate(refinement_levels[0:-1]):
        parent_cells = [node for node in graph.nodes if h3.h3_get_resolution(node) == resolution_parent]
        for parent_idx in parent_cells:
            # add own children
            for resolution_child in refinement_levels[i_level + 1 : i_level + depth + 1]:
                for child_idx in h3.h3_to_children(parent_idx, res=resolution_child):
                    if flat:
                        add_edge(
                            graph,
                            h3.h3_to_center_child(parent_idx, refinement_levels[-1]),
                            h3.h3_to_center_child(child_idx, refinement_levels[-1]),
                        )
                    else:
                        add_edge(graph, parent_idx, child_idx)

            # add neighbour children
            if has_neighbour_children:
                for idx_parent_neighbour in h3.k_ring(parent_idx, k=1) & parent_cells:
                    for resolution_child in refinement_levels[i_level + 1 : i_level + depth + 1]:
                        for idx_child_neighbour in h3.h3_to_children(idx_parent_neighbour, res=resolution_child):
                            if flat:
                                add_edge(
                                    graph,
                                    h3.h3_to_center_child(parent_idx, refinement_levels[-1]),
                                    h3.h3_to_center_child(idx_child_neighbour, refinement_levels[-1]),
                                )
                            else:
                                add_edge(graph, parent_idx, idx_child_neighbour)


def add_edge(
    graph: nx.Graph,
    source_node_h3_idx: str,
    target_node_h3_idx: str,
    allow_self_loop: bool = False,
) -> None:
    """Add edge between two nodes to a graph.

    The edge will only be added in case both tail and head nodes are included in the graph, G.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the nodes.
    idx1 : str
        The H3 index of the tail of the edge.
    idx2 : str
        The H3 index of the head of the edge.
    allow_self_loop : bool
        Whether to allow self-loops or not. Defaults to not allowing self-loops.
    """
    if not graph.has_node(source_node_h3_idx) or not graph.has_node(target_node_h3_idx):
        return

    if allow_self_loop or source_node_h3_idx != target_node_h3_idx:
        source_location = np.deg2rad(h3.h3_to_geo(source_node_h3_idx))
        target_location = np.deg2rad(h3.h3_to_geo(target_node_h3_idx))
        graph.add_edge(
            source_node_h3_idx, target_node_h3_idx, weight=haversine_distances([source_location, target_location])[0][1]
        )
