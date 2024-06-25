import logging
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import earth_radius
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

        # Normalize adjacency matrix.
        adjmat_norm = self.normalize_adjmat(adjmat)

        # Add edges to the graph and register normed distance.
        graph = self.register_edges(graph, adjmat.col, adjmat.row)

        self.register_edge_attribute(graph, "normed_dist", adjmat_norm.data)
        if attrs_config is not None:
            for attr_name, attr_cfg in attrs_config.items():
                attr_values = instantiate(attr_cfg)(graph, self.src_name, self.dst_name)
                graph = self.register_edge_attribute(graph, attr_name, attr_values)

        return graph

    def normalize_adjmat(self, adjmat):
        """Normalize a sparse adjacency matrix."""
        adjmat_norm = normalize(adjmat, norm="l1", axis=1)
        adjmat_norm.data = 1.0 - adjmat_norm.data
        return adjmat_norm


class KNNEdgeBuilder(BaseEdgeBuilder):
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


class CutOffEdgeBuilder(BaseEdgeBuilder):
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
        logger.debug("Using cut-off radius of %.1f km.", self.radius * earth_radius)

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(src_nodes.x)
        adj_matrix = nearest_neighbour.radius_neighbors_graph(dst_nodes.x, radius=self.radius).tocoo()
        return adj_matrix
