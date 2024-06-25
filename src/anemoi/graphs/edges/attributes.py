from abc import ABC
from abc import abstractmethod
from typing import Optional

import logging
import numpy as np
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.directional import directional_edge_features
from anemoi.graphs.normalizer import NormalizerMixin

logger = logging.getLogger(__name__)


class BaseEdgeAttribute(ABC, NormalizerMixin):
    norm: Optional[str] = None

    @abstractmethod
    def compute(self, graph: HeteroData, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        values = self.compute(*args, **kwargs)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        return self.normalize(values)


class DirectionalFeatures(BaseEdgeAttribute):
    norm: Optional[str] = None
    luse_rotated_features: bool = False

    def compute(self, graph: HeteroData, src_name: str, dst_name: str):
        edge_index = graph[(src_name, "to", dst_name)].edge_index
        src_coords = graph[src_name].x.numpy()[edge_index[0]].T
        dst_coords = graph[dst_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(src_coords, dst_coords, self.luse_rotated_features).T
        return edge_dirs
