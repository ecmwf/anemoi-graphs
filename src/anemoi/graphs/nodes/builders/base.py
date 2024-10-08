from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs.utils import get_grid_reference_distance

LOGGER = logging.getLogger(__name__)


class BaseNodeBuilder(ABC):
    """Base class for node builders.

    The node coordinates are stored in the `x` attribute of the nodes and they are stored in radians.

    Attributes
    ----------
    name : str
        name of the nodes, key for the nodes in the HeteroData graph object.
    aoi_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder, if any. Defaults to None.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.aoi_mask_builder = None

    def register_nodes(self, graph: HeteroData) -> HeteroData:
        """Register nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the nodes.

        Returns
        -------
        HeteroData
            The graph with the registered nodes.
        """
        graph[self.name].x = self.get_coordinates()
        graph[self.name].node_type = type(self).__name__
        graph[self.name]["_grid_reference_distance"] = get_grid_reference_distance(graph[self.name].x.cpu())
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict | None = None) -> HeteroData:
        """Register attributes in the nodes of the graph specified.

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
            graph[self.name][attr_name] = instantiate(attr_config).compute(graph, self.name)

        return graph

    @abstractmethod
    def get_coordinates(self) -> torch.Tensor: ...

    def reshape_coords(self, latitudes: np.ndarray, longitudes: np.ndarray) -> torch.Tensor:
        """Reshape latitude and longitude coordinates.

        Parameters
        ----------
        latitudes : np.ndarray of shape (num_nodes, )
            Latitude coordinates, in degrees.
        longitudes : np.ndarray of shape (num_nodes, )
            Longitude coordinates, in degrees.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = np.stack([latitudes, longitudes], axis=-1).reshape((-1, 2))
        coords = np.deg2rad(coords)
        return torch.tensor(coords, dtype=torch.float32)

    def update_graph(self, graph: HeteroData, attr_config: DotDict | None = None) -> HeteroData:
        """Update the graph with new nodes.

        Parameters
        ----------
        graph : HeteroData
            Input graph.
        attr_config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with new nodes included.
        """
        t0 = time.time()
        graph = self.register_nodes(graph)
        t1 = time.time()
        LOGGER.debug("Time to register node coordinates (%s): %.2f s", self.__class__.__name__, t1 - t0)

        if attr_config is None:
            return graph

        t0 = time.time()
        graph = self.register_attributes(graph, attr_config)
        t1 = time.time()
        LOGGER.debug("Time to register node coordinates (%s): %.2f s", self.__class__.__name__, t1 - t0)

        return graph
