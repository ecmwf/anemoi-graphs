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


class MissingZarrVariable(BaseNodeAttribute):
    """Mask of missing values of a Zarr dataset variable.

    It reads a variable from a Zarr dataset and returns a boolean mask of missing values in the first timestep.

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

    def __init__(self, variable: str, norm: str | None = None) -> None:
        super().__init__(norm, "bool")
        self.variable = variable

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        assert (
            nodes["node_type"] == "ZarrDatasetNodes"
        ), f"{self.__class__.__name__} can only be used with ZarrDatasetNodes."
        ds = open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()
        return np.isnan(ds)


class NotMissingZarrVariable(MissingZarrVariable):
    """Mask of valid (not missing) values of a Zarr dataset variable.

    It reads a variable from a Zarr dataset and returns a boolean mask of missing values in the first timestep.

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

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        return ~super().get_raw_values(nodes, **kwargs)


class CutOutMask(BaseNodeAttribute):
    """Cut out mask."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        assert isinstance(nodes["_dataset"], dict), "The 'dataset' attribute must be a dictionary."
        assert "cutout" in nodes["_dataset"], "The 'dataset' attribute must contain a 'cutout' key."
        n_cutout, n_grids = open_dataset(nodes["_dataset"]).grids
        return np.array([True] * n_cutout + [False] * n_grids, dtype=bool)
