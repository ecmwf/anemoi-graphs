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

    def compute(self, graph: HeteroData, source_name: str, target_name: str) -> torch.Tensor:
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]].T
        target_coords = graph[target_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(source_coords, target_coords, self.luse_rotated_features).T
        return edge_dirs


@dataclass
class EdgeLength(BaseEdgeAttribute):
    """Edge length feature."""

    norm: str = "l1"
    invert: bool = True

    def compute(self, graph: HeteroData, source_name: str, target_name: str) -> np.ndarray:
        """Compute haversine distance (in kilometers) between nodes connected by edges."""
        assert source_name in graph.node_types, f"Node {source_name} not found in graph."
        assert target_name in graph.node_types, f"Node {target_name} not found in graph."
        edge_index = graph[(source_name, "to", target_name)].edge_index
        source_coords = graph[source_name].x.numpy()[edge_index[0]]
        target_coords = graph[target_name].x.numpy()[edge_index[1]]
        edge_lengths = haversine_distance(source_coords, target_coords)
        return edge_lengths

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        if self.invert:
            values = 1 - values
        return super().post_process(values)
