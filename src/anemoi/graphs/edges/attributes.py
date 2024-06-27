import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.directional import directional_edge_features
from anemoi.graphs.normalizer import NormalizerMixin
from anemoi.graphs.utils import haversine_distance

logger = logging.getLogger(__name__)


@dataclass
class BaseEdgeAttribute(ABC, NormalizerMixin):
    norm: Optional[str] = None

    @abstractmethod
    def compute(self, graph: HeteroData, *args, **kwargs) -> np.ndarray: ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        return torch.tensor(values)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        values = self.compute(*args, **kwargs)
        normed_values = self.normalize(values)
        if normed_values.ndim == 1:
            normed_values = normed_values[:, np.newaxis]
        return self.post_process(normed_values)


@dataclass
class DirectionalFeatures(BaseEdgeAttribute):
    """Compute directional features for edges."""

    norm: Optional[str] = None
    luse_rotated_features: bool = False

    def compute(self, graph: HeteroData, src_name: str, dst_name: str) -> torch.Tensor:
        edge_index = graph[(src_name, "to", dst_name)].edge_index
        src_coords = graph[src_name].x.numpy()[edge_index[0]].T
        dst_coords = graph[dst_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(src_coords, dst_coords, self.luse_rotated_features).T
        return edge_dirs


@dataclass
class HaversineDistance(BaseEdgeAttribute):
    """Edge length feature."""

    norm: str = "l1"
    invert: bool = True

    def compute(self, graph: HeteroData, src_name: str, dst_name: str):
        """Compute haversine distance (in kilometers) between nodes connected by edges."""
        assert src_name in graph.node_types, f"Node {src_name} not found in graph."
        assert dst_name in graph.node_types, f"Node {dst_name} not found in graph."
        edge_index = graph[(src_name, "to", dst_name)].edge_index
        src_coords = graph[src_name].x.numpy()[edge_index[0]]
        dst_coords = graph[dst_name].x.numpy()[edge_index[1]]
        edge_lengths = haversine_distance(src_coords, dst_coords)
        return coo_matrix((edge_lengths, (edge_index[1], edge_index[0])))

    def normalize(self, values) -> np.ndarray:
        """Normalize the edge length.

        This method scales the edge lengths to a unit norm, computing the norms
        for each source node (axis=1).
        """
        return normalize(values, norm="l1", axis=1).data

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        if self.invert:
            values = 1 - values
        return super().post_process(values)
