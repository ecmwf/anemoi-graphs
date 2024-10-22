from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from anemoi.datasets import open_dataset
from scipy.spatial import SphericalVoronoi
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.normalizer import NormalizerMixin

LOGGER = logging.getLogger(__name__)


class BaseNodeAttribute(ABC, NormalizerMixin):
    """Base class for the weights of the nodes."""

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        self.norm = norm
        self.dtype = dtype

    @abstractmethod
    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray: ...

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        norm_values = self.normalize(values)

        return torch.tensor(norm_values.astype(self.dtype))

    def compute(self, graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor:
        """Get the nodes attribute.

        Parameters
        ----------
        graph : HeteroData
            Graph.
        nodes_name : str
            Name of the nodes.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Attributes associated to the nodes.
        """
        nodes = graph[nodes_name]
        attributes = self.get_raw_values(nodes, **kwargs)
        return self.post_process(attributes)


class UniformWeights(BaseNodeAttribute):
    """Implements a uniform weight for the nodes.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        """Compute the weights.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Attributes.
        """
        return np.ones(nodes.num_nodes)


class AreaWeights(BaseNodeAttribute):
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
    compute(self, graph, nodes_name)
        Compute the area attributes for each node.
    """

    def __init__(
        self,
        norm: str | None = None,
        radius: float = 1.0,
        centre: np.ndarray = np.array([0, 0, 0]),
        dtype: str = "float32",
    ) -> None:
        super().__init__(norm, dtype)
        self.radius = radius
        self.centre = centre

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        """Compute the area associated to each node.

        It uses Voronoi diagrams to compute the area of each node.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Attributes.
        """
        latitudes, longitudes = nodes.x[:, 0], nodes.x[:, 1]
        points = latlon_rad_to_cartesian((np.asarray(latitudes), np.asarray(longitudes)))
        sv = SphericalVoronoi(points, self.radius, self.centre)
        area_weights = sv.calculate_areas()
        LOGGER.debug(
            "There are %d of weights, which (unscaled) add up a total weight of %.2f.",
            len(area_weights),
            np.array(area_weights).sum(),
        )
        return area_weights


class ZarrDatasetAttribute(BaseNodeAttribute):
    """Read the attribute from a Zarr dataset variable.

    Attributes
    ----------
    variable : str
        Variable to read from the Zarr dataset.
    norm : str
        Normalization of the weights.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self, variable: str, invert: bool = False, norm: str | None = None, dtype: str = "bool") -> None:
        super().__init__(norm, dtype)
        self.variable = variable
        self.invert = invert

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        ds = open_dataset(nodes["_dataset"], select=self.variable)[0]
        return ds.squeeze().astype(self.dtype)

    def post_process(self, values: np.ndarray) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = values[:, np.newaxis]

        norm_values = self.normalize(values)

        if self.invert:
            norm_values = 1 - norm_values

        return torch.tensor(norm_values.astype(self.dtype))
