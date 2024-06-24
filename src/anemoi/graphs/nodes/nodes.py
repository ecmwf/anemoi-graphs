from abc import abstractmethod
from pathlib import Path
from typing import Optional
from typing import Union

import h3
import numpy as np
import torch
from abc import ABC
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from aifs.graphs import GraphBuilder
from aifs.graphs.generate.hexagonal import create_hexagonal_nodes
from aifs.graphs.generate.icosahedral import create_icosahedral_nodes
import logging

logger = logging.getLogger(__name__)
earth_radius = 6371.0  # km


def latlon_to_radians(coords: np.ndarray) -> np.ndarray:
    return np.deg2rad(coords)


def rad_to_latlon(coords: np.ndarray) -> np.ndarray:
    """Converts coordinates from radians to degrees.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates in radians.

    Returns
    -------
    np.ndarray
        _description_
    """
    return np.rad2deg(coords)


class BaseNodeBuilder(ABC):

    def register_nodes(self, graph: HeteroData, name: str) -> None:
        graph[name].x = self.get_coordinates()
        graph[name].node_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, name: str, config: DotDict) -> HeteroData:
        for nodes_attr_name, attr_cfg in config.items():
            graph[name][nodes_attr_name] = instantiate(attr_cfg).get_weights(graph[name])
        return graph

    @abstractmethod
    def get_coordinates(self) -> np.ndarray: ...

    def reshape_coords(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        coords = np.stack([latitudes, longitudes], axis=-1).reshape((-1, 2))
        coords = np.deg2rad(coords)
        # TODO: type needs to be variable?
        return torch.tensor(coords, dtype=torch.float32)

    def transform(self, graph: HeteroData, name: str, attr_config: DotDict) -> HeteroData:
        graph = self.register_nodes(graph, name)
        graph = self.register_attributes(graph, name, attr_config)
        return graph


class ZarrNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset."""

    def __init__(self, dataset: DotDict) -> None:
        logger.info("Reading the dataset from %s.", dataset)
        self.ds = open_dataset(dataset)

    def get_coordinates(self) -> torch.Tensor:
        return self.reshape_coords(self.ds.latitudes, self.ds.longitudes)


class NPZNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids."""

    def __init__(self, resolution: str, grid_definition_path: str) -> None:
        self.resolution = resolution
        self.grid_definition_path = grid_definition_path
        self.grid_definition = np.load(Path(self.grid_definition_path) / f"grid-{self.resolution}.npz")

    def get_coordinates(self) -> np.ndarray:
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords
