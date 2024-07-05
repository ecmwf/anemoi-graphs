import logging
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.utils import get_grid_reference_distance

logger = logging.getLogger(__name__)


class BaseEdgeBuilder:
    """Base class for edge builders."""

    def __init__(self, source_name: str, target_name: str):
        super().__init__()
        self.source_name = source_name
        self.target_name = target_name

    @abstractmethod
    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage): ...

    def register_edges(self, graph: HeteroData, source_indices: np.ndarray, target_indices: np.ndarray) -> HeteroData:
        """Register edges in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the edges.
        source_indices : np.ndarray of shape (N, )
            The indices of the source nodes.
        target_indices : np.ndarray of shape (N, )
            The indices of the target nodes.

        Returns
        -------
        HeteroData
            The graph with the registered edges.
        """
        edge_index = np.stack([source_indices, target_indices], axis=0).astype(np.int32)
        graph[(self.source_name, "to", self.target_name)].edge_index = torch.from_numpy(edge_index)
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        """Register attributes in the edges of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        for attr_name, attr_config in config.items():
            graph[self.source_name, "to", self.target_name][attr_name] = instantiate(attr_config).compute(
                graph, self.source_name, self.target_name
            )
        return graph

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare nodes information."""
        return graph[self.source_name], graph[self.target_name]

    def transform(self, graph: HeteroData, attrs_config: Optional[DotDict] = None) -> HeteroData:
        """Transform the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        attrs_config : DotDict
            The configuration of the edge attributes.

        Returns
        -------
        HeteroData
            The graph with the edges.
        """
        source_nodes, target_nodes = self.prepare_node_data(graph)

        adjmat = self.get_adjacency_matrix(source_nodes, target_nodes)

        graph = self.register_edges(graph, adjmat.col, adjmat.row)

        if attrs_config is None:
            return graph

        graph = self.register_attributes(graph, attrs_config)

        return graph


class KNNEdges(BaseEdgeBuilder):
    """Computes KNN based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    num_nearest_neighbours : int
        Number of nearest neighbours.
    """

    def __init__(self, source_name: str, target_name: str, num_nearest_neighbours: int):
        super().__init__(source_name, target_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer"
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def get_adjacency_matrix(self, source_nodes: np.ndarray, target_nodes: np.ndarray):
        """Compute the adjacency matrix for the KNN method.

        Parameters
        ----------
        source_nodes : np.ndarray
            The source nodes.
        target_nodes : np.ndarray
            The target nodes.
        """
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        logger.debug(
            "Using %d nearest neighbours for KNN-Edges between %s and %s.",
            self.num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_nodes.x.numpy())
        adj_matrix = nearest_neighbour.kneighbors_graph(
            target_nodes.x.numpy(),
            n_neighbors=self.num_nearest_neighbours,
            mode="distance",
        ).tocoo()
        return adj_matrix


class CutOffEdges(BaseEdgeBuilder):
    """Computes cut-off based edges and adds them to the graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    cutoff_factor : float
        Factor to multiply the grid reference distance to get the cut-off radius.
    radius : float
        Cut-off radius.
    """

    def __init__(self, source_name: str, target_name: str, cutoff_factor: float):
        super().__init__(source_name, target_name)
        assert isinstance(cutoff_factor, float), "Cutoff factor must be a float"
        assert cutoff_factor > 0, "Cutoff factor must be positive"
        self.cutoff_factor = cutoff_factor

    def get_cutoff_radius(self, graph: HeteroData, mask_attr: Optional[torch.Tensor] = None):
        """Compute the cut-off radius.

        The cut-off radius is computed as the product of the target nodes reference distance and the cut-off factor.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        mask_attr : torch.Tensor
            The mask attribute.

        Returns
        -------
        float
            The cut-off radius.
        """
        target_nodes = graph[self.target_name]
        mask = target_nodes[mask_attr] if mask_attr is not None else None
        target_grid_reference_distance = get_grid_reference_distance(target_nodes.x, mask)
        radius = target_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare nodes information."""
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def get_adjacency_matrix(self, source_nodes: NodeStorage, target_nodes: NodeStorage):
        """Get the adjacency matrix for the cut-off method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.
        """
        logger.debug("Using cut-off radius of %.1f km.", self.radius * EARTH_RADIUS)

        nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        nearest_neighbour.fit(source_nodes.x)
        adj_matrix = nearest_neighbour.radius_neighbors_graph(target_nodes.x, radius=self.radius).tocoo()
        return adj_matrix
