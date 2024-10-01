from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod

import networkx as nx
import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage
from torch_geometric.nn import knn
from torch_geometric.nn import radius

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate import hexagonal
from anemoi.graphs.generate import icosahedral
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_torch
from anemoi.graphs.nodes.builder import HexNodes
from anemoi.graphs.nodes.builder import TriNodes
from anemoi.graphs.utils import get_grid_reference_distance

LOGGER = logging.getLogger(__name__)


class BaseEdgeBuilder(ABC):
    """Base class for edge builders."""

    def __init__(self, source_name: str, target_name: str):
        self.source_name = source_name
        self.target_name = target_name

    @property
    def name(self) -> tuple[str, str, str]:
        """Name of the edge subgraph."""
        return self.source_name, "to", self.target_name

    @abstractmethod
    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor: ...

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
        return graph[self.source_name], graph[self.target_name]

    def get_edge_index(self, graph: HeteroData) -> torch.Tensor:
        """Get the edge index."""
        source_nodes, target_nodes = self.prepare_node_data(graph)
        return self.compute_edge_index(source_nodes, target_nodes)

    def register_edges(self, graph: HeteroData) -> HeteroData:
        """Register edges in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the edges.

        Returns
        -------
        HeteroData
            The graph with the registered edges.
        """
        graph[self.name].edge_index = self.get_edge_index(graph)
        graph[self.name].edge_type = type(self).__name__
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
            edge_index = graph[self.name].edge_index
            source_coords = graph[self.name[0]].x[edge_index[0]]
            target_coords = graph[self.name[2]].x[edge_index[1]]
            edge_builder = instantiate(attr_config)
            graph[self.name][attr_name] = edge_builder(x=(source_coords, target_coords), edge_index=edge_index)
        return graph

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        """Update the graph with the edges.

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
        t0 = time.time()
        graph = self.register_edges(graph)
        t1 = time.time()
        LOGGER.info("Time to register edge indices (%s): %.2f s", self.__class__.__name__, t1 - t0)

        if attrs_config is None:
            return graph

        t0 = time.time()
        graph = self.register_attributes(graph, attrs_config)
        t1 = time.time()
        LOGGER.info("Time to register edge attributes (%s): %.2f s", self.__class__.__name__, t1 - t0)

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

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(self, source_name: str, target_name: str, num_nearest_neighbours: int):
        super().__init__(source_name, target_name)
        assert isinstance(num_nearest_neighbours, int), "Number of nearest neighbours must be an integer"
        assert num_nearest_neighbours > 0, "Number of nearest neighbours must be positive"
        self.num_nearest_neighbours = num_nearest_neighbours

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge indices for the KNN method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            Indices of source and target nodes connected by an edge.
        """
        assert self.num_nearest_neighbours is not None, "number of neighbors required for knn encoder"
        LOGGER.info(
            "Using KNN-Edges (with %d nearest neighbours) between %s and %s.",
            self.num_nearest_neighbours,
            self.source_name,
            self.target_name,
        )
        edge_idx = knn(
            latlon_rad_to_cartesian_torch(source_nodes.x),
            latlon_rad_to_cartesian_torch(target_nodes.x),
            k=self.num_nearest_neighbours,
        )
        edge_idx = torch.flip(edge_idx, [0])
        return edge_idx


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

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(self, source_name: str, target_name: str, cutoff_factor: float, max_num_neighbours: int = 32):
        super().__init__(source_name, target_name)
        assert isinstance(cutoff_factor, (int, float)), "Cutoff factor must be a float"
        assert isinstance(max_num_neighbours, int), "Number of nearest neighbours must be an integer"
        assert cutoff_factor > 0, "Cutoff factor must be positive"
        assert max_num_neighbours > 0, "Number of nearest neighbours must be positive"
        self.cutoff_factor = cutoff_factor
        self.max_num_neighbours = max_num_neighbours

    def get_cutoff_radius(self, graph: HeteroData, mask_attr: torch.Tensor | None = None):
        """Compute the cut-off radius.

        The cut-off radius is computed as the product of the target nodes
        reference distance and the cut-off factor.

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
        if mask_attr is not None:
            # If masking target nodes, we have to recompute the grid reference distance only over the masked nodes
            mask = target_nodes[mask_attr].cpu()
            target_grid_reference_distance = get_grid_reference_distance(target_nodes.x.cpu(), mask)
        else:
            target_grid_reference_distance = target_nodes._grid_reference_distance

        radius = target_grid_reference_distance * self.cutoff_factor
        return radius

    def prepare_node_data(self, graph: HeteroData) -> tuple[NodeStorage, NodeStorage]:
        """Prepare node information and get source and target nodes."""
        self.radius = self.get_cutoff_radius(graph)
        return super().prepare_node_data(graph)

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage):
        """Get the adjacency matrix for the cut-off method.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.
        """
        LOGGER.info(
            "Using CutOff-Edges (with radius = %.1f km) between %s and %s.",
            self.radius * EARTH_RADIUS,
            self.source_name,
            self.target_name,
        )

        edge_idx = radius(
            latlon_rad_to_cartesian_torch(source_nodes.x),
            latlon_rad_to_cartesian_torch(target_nodes.x),
            r=self.radius,
            max_num_neighbors=self.max_num_neighbours,
        )
        edge_idx = torch.flip(edge_idx, [0])
        return edge_idx


class MultiScaleEdges(BaseEdgeBuilder):
    """Base class for multi-scale edges in the nodes of a graph.

    Attributes
    ----------
    source_name : str
        The name of the source nodes.
    target_name : str
        The name of the target nodes.
    x_hops : int
        Number of hops (in the refined icosahedron) between two nodes to connect
        them with an edge.

    Methods
    -------
    register_edges(graph)
        Register the edges in the graph.
    register_attributes(graph, config)
        Register attributes in the edges of the graph.
    update_graph(graph, attrs_config)
        Update the graph with the edges.
    """

    def __init__(self, source_name: str, target_name: str, x_hops: int):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        assert isinstance(x_hops, int), "Number of x_hops must be an integer"
        assert x_hops > 0, "Number of x_hops must be positive"
        self.x_hops = x_hops

    def adjacency_from_tri_nodes(self, source_nodes: NodeStorage):
        source_nodes["_nx_graph"] = icosahedral.add_edges_to_nx_graph(
            source_nodes["_nx_graph"],
            resolutions=source_nodes["_resolutions"],
            x_hops=self.x_hops,
        )  # HeteroData refuses to accept None

        adjmat = nx.to_scipy_sparse_array(
            source_nodes["_nx_graph"], nodelist=list(range(len(source_nodes["_nx_graph"]))), format="coo"
        )
        return adjmat

    def adjacency_from_hex_nodes(self, source_nodes: NodeStorage):

        source_nodes["_nx_graph"] = hexagonal.add_edges_to_nx_graph(
            source_nodes["_nx_graph"],
            resolutions=source_nodes["_resolutions"],
            x_hops=self.x_hops,
        )

        adjmat = nx.to_scipy_sparse_array(source_nodes["_nx_graph"], format="coo")
        return adjmat

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        if self.node_type == TriNodes.__name__:
            adjmat = self.adjacency_from_tri_nodes(source_nodes)
        elif self.node_type == HexNodes.__name__:
            adjmat = self.adjacency_from_hex_nodes(source_nodes)
        else:
            raise ValueError(f"Invalid node type {self.node_type}")

        adjmat = self.post_process_adjmat(source_nodes, adjmat)

        # Get source & target indices of the edges
        edge_index = np.stack([adjmat.col, adjmat.row], axis=0)

        return torch.from_numpy(edge_index).to(torch.int32)

    def post_process_adjmat(self, nodes: NodeStorage, adjmat):
        graph_sorted = {node_pos: i for i, node_pos in enumerate(nodes["_node_ordering"])}
        sort_func = np.vectorize(graph_sorted.get)
        adjmat.row = sort_func(adjmat.row)
        adjmat.col = sort_func(adjmat.col)
        return adjmat

    def update_graph(self, graph: HeteroData, attrs_config: DotDict | None = None) -> HeteroData:
        assert (
            graph[self.source_name].node_type == TriNodes.__name__
            or graph[self.source_name].node_type == HexNodes.__name__
        ), f"{self.__class__.__name__} requires {TriNodes.__name__} or {HexNodes.__name__}."

        self.node_type = graph[self.source_name].node_type

        return super().update_graph(graph, attrs_config)
