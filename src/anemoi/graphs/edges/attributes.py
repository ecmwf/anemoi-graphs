import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.directional import directional_edge_features
from anemoi.graphs.normalizer import NormalizerMixin
from anemoi.graphs.utils import haversine_distance

LOGGER = logging.getLogger(__name__)


@dataclass
class BaseEdgeAttribute(ABC, NormalizerMixin):
    """Base class for edge attributes."""

    norm: Optional[str] = None

    @abstractmethod
    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str, *args, **kwargs) -> np.ndarray: ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        return torch.tensor(values)

    def compute(self, graph: HeteroData, source_name: str, target_name: str, *args, **kwargs) -> torch.Tensor:
        """Compute the edge attributes."""
        assert (
            source_name in graph.node_types
        ), f"Node \"{source_name}\" not found in graph. Optional nodes are {', '.join(graph.node_types)}."
        assert (
            target_name in graph.node_types
        ), f"Node \"{target_name}\" not found in graph. Optional nodes are {', '.join(graph.node_types)}."

        values = self.get_raw_values(graph, source_name, target_name, *args, **kwargs)
        normed_values = self.normalize(values)
        return self.post_process(normed_values)


@dataclass
class EdgeDirection(BaseEdgeAttribute):
    """Compute directional features for edges.

    If using the rotated features, the direction of the edge is computed
    rotating the target nodes to the north pole. If not, it is computed
    as the diference in latitude and longitude between the source and
    target nodes.

    Attributes
    ----------
    norm : Optional[str]
        Normalization method.
    luse_rotated_features : bool
        Whether to use rotated features.

    Methods
    -------
    get_raw_values(graph, source_name, target_name)
        Compute directions between nodes connected by edges.
    compute(graph, source_name, target_name)
        Compute directional attributes.
    """

    norm: Optional[str] = None
    luse_rotated_features: bool = True

    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str) -> np.ndarray:
        """Compute directional features for edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        source_name : str
            The name of the source nodes.
        target_name : str
            The name of the target nodes.

        Returns
        -------
        np.ndarray
            The directional features.
        """
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]].T
        target_coords = graph[target_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(source_coords, target_coords, self.luse_rotated_features).T
        return edge_dirs


@dataclass
class EdgeLength(BaseEdgeAttribute):
    """Edge length feature.

    Attributes
    ----------
    norm : str
        Normalization method.
    invert : bool
        Whether to invert the edge lengths, i.e. 1 - edge_length.

    Methods
    -------
    get_raw_values(graph, source_name, target_name)
        Compute haversine distance between nodes connected by edges.
    compute(graph, source_name, target_name)
        Compute edge lengths attributes.
    """

    norm: str = "l1"
    invert: bool = True

    def get_raw_values(self, graph: HeteroData, source_name: str, target_name: str) -> np.ndarray:
        """Compute haversine distance (in kilometers) between nodes connected by edges.

        Parameters
        ----------
        graph : HeteroData
            The graph.
        source_name : str
            The name of the source nodes.
        target_name : str
            The name of the target nodes.

        Returns
        -------
        np.ndarray
            The edge lengths.
        """
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]]
        target_coords = graph[target_name].x.numpy()[edge_index[1]]
        edge_lengths = haversine_distance(source_coords, target_coords)
        return edge_lengths

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process edge lengths."""
        if self.invert:
            values = 1 - values
        return super().post_process(values)
