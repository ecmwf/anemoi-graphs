from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import to_sphere_xyz
from scipy.spatial import SphericalVoronoi
from anemoi.graphs.normalizer import NormalizerMixin
import logging

logger = logging.getLogger(__name__)

class BaseWeights(ABC, NormalizerMixin):
    """Base class for the weights of the nodes."""

    def __init__(self, norm: Optional[str] = None):
        self.norm = norm

    @abstractmethod
    def compute(self, nodes: NodeStorage, *args, **kwargs): ...

    def get_weights(self, *args, **kwargs):
        weights = self.compute(*args, **kwargs)
        if weights.ndim == 1:
            weights = weights[:, np.newaxis]
        return self.normalize(weights)


class UniformWeights(BaseWeights):
    """Implements a uniform weight for the nodes."""

    def __init__(self, norm: str = "unit-max"):
        self.norm = norm

    def compute(self, nodes: NodeStorage) -> np.ndarray:
        return torch.ones(nodes.num_nodes)


class AreaWeights(BaseWeights):
    """Implements the area of the nodes as the weights."""

    def __init__(self, norm: str = "unit-max", radius: float = 1.0, centre: np.ndarray = np.array[0, 0, 0]):
        # Weighting of the nodes
        self.norm: str = norm
        self.radius: float = radius
        self.centre: np.ndarray = centre

    def compute(self, nodes: NodeStorage, *args, **kwargs) -> np.ndarray:
        # TODO: Check if works
        latitudes, longitudes = nodes.x[:, 0], nodes.x[:, 1]
        points = to_sphere_xyz((latitudes, longitudes))
        sv = SphericalVoronoi(points, self.radius, self.centre)
        area_weights = sv.calculate_areas()
        logger.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(area_weights),
            np.array(area_weights).sum(),
        )
        return area_weights
