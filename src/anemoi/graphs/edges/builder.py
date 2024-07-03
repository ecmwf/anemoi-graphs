import logging
from abc import abstractmethod
from typing import Optional

import networkx as nx
import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate import hexagonal
from anemoi.graphs.generate import icosahedral
from anemoi.graphs.nodes.builder import HexRefinedIcosahedralNodes
from anemoi.graphs.nodes.builder import TriRefinedIcosahedralNodes
from anemoi.graphs.utils import get_grid_reference_distance

logger = logging.getLogger(__name__)


class BaseEdgeBuilder:
    """Base class for edge builders."""

    def __init__(self, src_name: str, dst_name: str):
        super().__init__()
        self.src_name = src_name
        self.dst_name = dst_name

    @abstractmethod
    def get_adj_matrix(self, src_nodes: NodeStorage, dst_nodes: NodeStorage): ...

    def register_edges(self, graph, head_indices, tail_indices):
        graph[(self.src_name, "to", self.dst_name)].edge_index = np.stack([head_indices, tail_indices], axis=0).astype(
            np.int32
        )
        return graph

    def register_edge_attribute(self, graph: HeteroData, name: str, values: np.ndarray):
        num_edges = graph[(self.src_name, "to", self.dst_name)].num_edges
        assert (
            values.shape[0] == num_edges
        ), f"Number of edge features ({values.shape[0]}) must match number of edges ({num_edges})."
        graph[self.src_name, "to", self.dst_name][name] = values.reshape(
            num_edges, -1
        )  # TODO: Check the [name] part works
        return graph

    def prepare_node_data(self, graph: HeteroData):
        return graph[self.src_name], graph[self.dst_name]

    def transform(self, graph: HeteroData, attrs_config: Optional[DotDict] = None) -> HeteroData:
        # Get source and destination nodes.
        src_nodes, dst_nodes = self.prepare_node_data(graph)

        # Compute adjacency matrix.
        adjmat = self.get_adj_matrix(src_nodes, dst_nodes)

        # Add edges to the graph and register normed distance.
        graph = self.register_edges(graph, adjmat.col, adjmat.row)

        if attrs_config is not None:
            for attr_name, attr_cfg in attrs_config.items():
                attr_values = instantiate(attr_cfg)(graph, self.src_name, self.dst_name)
                graph = self.register_edge_attribute(graph, attr_name, attr_values)

        return graph


class KNNEdges(BaseEdgeBuilder):
    """Computes KNN based edges and adds them to the graph."""

    def __init__(self, src_name: str, dst_name: str, num_nearest_neighbours: int):
        super().__init__(src_name, dst_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer"
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def get_adj_matrix(self, src_nodes: np.ndarray, dst_nodes: np.ndarray):
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        logger.debug(
            "Using %d nearest neighbours for KNN-Edges between %s and %s.",
            self.num_nearest_neighbours,
            self.src_name,
            self.dst_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(src_nodes.x.numpy())
        adj_matrix = nearest_neighbour.kneighbors_graph(
            dst_nodes.x.numpy(),
            n_neighbors=self.num_nearest_neighbours,
            mode="distance",
        ).tocoo()
        return adj_matrix


class CutOffEdges(BaseEdgeBuilder):
    """Computes cut-off based edges and adds them to the graph."""

    def __init__(self, src_name: str, dst_name: str, cutoff_factor: float):
        super().__init__(src_name, dst_name)
        assert isinstance(cutoff_factor, float), "Cutoff factor must be a float"
        assert cutoff_factor > 0, "Cutoff factor must be positive"
        self.cutoff_factor = cutoff_factor

    def get_cutoff_radius(self, graph: HeteroData, mask_attr: Optional[torch.Tensor] = None):
        dst_nodes = graph[self.dst_name]
        mask = dst_nodes[mask_attr] if mask_attr is not None else None
        dst_grid_reference_distance = get_grid_reference_distance(dst_nodes.x, mask)
        radius = dst_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData):
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def get_adj_matrix(self, src_nodes: NodeStorage, dst_nodes: NodeStorage):
        logger.debug("Using cut-off radius of %.1f km.", self.radius * EARTH_RADIUS)

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(src_nodes.x)
        adj_matrix = nearest_neighbour.radius_neighbors_graph(dst_nodes.x, radius=self.radius).tocoo()
        return adj_matrix


class TriIcosahedralEdges(BaseEdgeBuilder):
    """Computes icosahedral edges and adds them to a HeteroData graph."""

    def __init__(self, src_name: str, xhops: int):
        super().__init__(src_name, src_name)

        assert isinstance(xhops, int), "Number of xhops must be an integer"
        assert xhops > 0, "Number of xhops must be positive"

        self.xhops = xhops

    def transform(self, graph: HeteroData, attrs_config: Optional[DotDict] = None) -> HeteroData:

        assert (
            graph[self.src_name].node_type == TriRefinedIcosahedralNodes.__name__
        ), "IcosahedralConnection requires TriRefinedIcosahedralNodes."

        # TODO: Next assert doesn't exist anymore since filters were moved, make sure this is checked where appropriate
        # assert filter_src is None and filter_dst is None, "InheritConnection does not support filtering with attributes."

        return super().transform(graph, attrs_config)

    def get_adj_matrix(self, src_nodes: NodeStorage, dst_nodes: NodeStorage):

        src_nodes["nx_graph"] = icosahedral.add_edges_to_nx_graph(
            src_nodes["nx_graph"],
            resolutions=src_nodes["resolutions"],
            xhops=self.xhops,
            aoi_nneighb=None if "aoi_mask_builder" not in src_nodes else src_nodes["aoi_mask_builder"],
        )  # HeteroData refuses to accept None

        adjmat = nx.to_scipy_sparse_array(src_nodes["nx_graph"], nodelist=list(src_nodes["nx_graph"]), format="coo")
        graph_1_sorted = dict(zip(range(len(src_nodes["nx_graph"].nodes)), list(src_nodes["nx_graph"].nodes)))
        graph_2_sorted = dict(zip(src_nodes.node_ordering, range(len(src_nodes.node_ordering))))
        sort_func1 = np.vectorize(graph_1_sorted.get)
        sort_func2 = np.vectorize(graph_2_sorted.get)
        adjmat.row = sort_func2(sort_func1(adjmat.row))
        adjmat.col = sort_func2(sort_func1(adjmat.col))
        return adjmat


class HexagonalEdges(BaseEdgeBuilder):
    """Computes hexagonal edges and adds them to a HeteroData graph."""

    def __init__(self, src_name: str, add_neighbouring_children: bool = False, depth_children: Optional[int] = 1):
        super().__init__(src_name, src_name)
        self.add_neighbouring_children = add_neighbouring_children

        assert isinstance(depth_children, int), "Depth of children must be an integer"
        assert depth_children > 0, "Depth of children must be positive"
        self.depth_children = depth_children

    def transform(self, graph: HeteroData, attrs_config: Optional[DotDict] = None) -> HeteroData:
        assert (
            graph[self.src_name].node_type == HexRefinedIcosahedralNodes.__name__
        ), "HexagonalEdges requires HexRefinedIcosahedralNodes."

        # TODO: Next assert doesn't exist anymore since filters were moved, make sure this is checked where appropriate
        # assert filter_src is None and filter_dst is None, "InheritConnection does not support filtering with attributes."

        return super().transform(graph, attrs_config)

    def get_adj_matrix(self, src_nodes: NodeStorage, dst_nodes: NodeStorage):

        src_nodes["nx_graph"] = hexagonal.add_edges_to_nx_graph(
            src_nodes["nx_graph"],
            resolutions=src_nodes["resolutions"],
            neighbour_children=self.add_neighbouring_children,
            depth_children=self.depth_children,
        )

        adjmat = nx.to_scipy_sparse_array(src_nodes["nx_graph"], format="coo")
        graph_2_sorted = dict(zip(src_nodes["node_ordering"], range(len(src_nodes.node_ordering))))
        sort_func = np.vectorize(graph_2_sorted.get)
        adjmat.row = sort_func(adjmat.row)
        adjmat.col = sort_func(adjmat.col)
        return adjmat
