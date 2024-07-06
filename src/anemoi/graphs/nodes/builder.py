import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class BaseNodeBuilder(ABC):
    """Base class for node builders."""

    def register_nodes(self, graph: HeteroData, name: str) -> None:
        """Register nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the nodes.
        name : str
            The name of the nodes.
        """
        graph[name].x = self.get_coordinates()
        graph[name].node_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, name: str, config: Optional[DotDict] = None) -> HeteroData:
        """Register attributes in the nodes of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        name : str
            The name of the nodes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        if config is None:
            return graph

        for attr_name, attr_config in config.items():
            graph[name][attr_name] = instantiate(attr_config).compute(graph[name])
        return graph

    @abstractmethod
    def get_coordinates(self) -> torch.Tensor: ...

    def reshape_coords(self, latitudes: np.ndarray, longitudes: np.ndarray) -> torch.Tensor:
        """Reshape latitude and longitude coordinates.

        Parameters
        ----------
        latitudes : np.ndarray of shape (N, )
            Latitude coordinates, in degrees.
        longitudes : np.ndarray of shape (N, )
            Longitude coordinates, in degrees.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = np.stack([latitudes, longitudes], axis=-1).reshape((-1, 2))
        coords = np.deg2rad(coords)
        return torch.tensor(coords, dtype=torch.float32)

    def update_graph(self, graph: HeteroData, name: str, attr_config: Optional[DotDict] = None) -> HeteroData:
        """Update the graph with new nodes.

        Parameters
        ----------
        graph : HeteroData
            Input graph.
        name : str
            The name of the nodes.
        attr_config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with new nodes included.
        """
        graph = self.register_nodes(graph, name)

        if attr_config is None:
            return graph

        graph = self.register_attributes(graph, name, attr_config)
        return graph


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    ds : zarr.core.Array
        The dataset.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, dataset: DotDict) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.ds = open_dataset(dataset)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        return self.reshape_coords(self.ds.latitudes, self.ds.longitudes)


class NPZFileNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids.

    Attributes
    ----------
    resolution : str
        The resolution of the grid.
    grid_definition_path : str
        Path to the folder containing the grid definition files.
    grid_definition : dict[str, np.ndarray]
        The grid definition.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, resolution: str, grid_definition_path: str) -> None:
        """Initialize the NPZFileNodes builder.

        The builder suppose the grids are stored in files with the name `grid-{resolution}.npz`.

        Parameters
        ----------
        resolution : str
            The resolution of the grid.
        grid_definition_path : str
            Path to the folder containing the grid definition files.
        """
        self.resolution = resolution
        self.grid_definition_path = grid_definition_path
        self.grid_definition = np.load(Path(self.grid_definition_path) / f"grid-{self.resolution}.npz")

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords
