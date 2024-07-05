import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import to_sphere_xyz
from anemoi.graphs.normalizer import NormalizerMixin

logger = logging.getLogger(__name__)


class BaseWeights(ABC, NormalizerMixin):
    """Base class for the weights of the nodes."""

    def __init__(self, norm: Optional[str] = None):
        self.norm = norm

    @abstractmethod
    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs): ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        return torch.tensor(values)

    def compute(self, nodes: NodeStorage, *args, **kwargs) -> torch.Tensor:
        """Get the node weights.

        Returns
        -------
        torch.Tensor
            Weights associated to the nodes.
        """
        weights = self.get_raw_values(nodes, *args, **kwargs)
        norm_weights = self.normalize(weights)
        return self.post_process(norm_weights)


class UniformWeights(BaseWeights):
    """Implements a uniform weight for the nodes."""

    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs) -> np.ndarray:
        """Compute the weights.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.

        Returns
        -------
        np.ndarray
            Weights.
        """
        return np.ones(nodes.num_nodes)


class AreaWeights(BaseWeights):
    """Implements the area of the nodes as the weights.

    Attributes
    ----------
    norm : str
        Normalization of the weights.
    radius : float
        Radius of the sphere.
    centre : np.ndarray
        Centre of the sphere.

    Methods
    -------
    get_raw_values(nodes, *args, **kwargs)
        Compute the area associated to each node.
    compute(nodes, *args, **kwargs)
        Compute the area attributes for each node.
    """

    def __init__(self, norm: str = "unit-max", radius: float = 1.0, centre: np.ndarray = np.array([0, 0, 0])):
        super().__init__(norm=norm)

        # Weighting of the nodes
        self.radius = radius
        self.centre = centre

    def get_raw_values(self, nodes: NodeStorage, *args, **kwargs) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.

        Returns
        -------
        np.ndarray
            Weights.
        """
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
