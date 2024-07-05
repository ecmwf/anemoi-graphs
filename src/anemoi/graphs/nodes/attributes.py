import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.normalizer import NormalizerMixin

LOGGER = logging.getLogger(__name__)


@dataclass
class BaseWeights(ABC, NormalizerMixin):
    """Base class for the weights of the nodes."""

    norm: Optional[str] = None

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


@dataclass
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

    norm: Optional[str] = "unit-max"
    radius: float = 1.0
    centre: np.ndarray = np.array([0, 0, 0])

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
        points = latlon_rad_to_cartesian((latitudes, longitudes))
        sv = SphericalVoronoi(points, self.radius, self.centre)
        area_weights = sv.calculate_areas()
        LOGGER.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(area_weights),
            np.array(area_weights).sum(),
        )
        return area_weights
