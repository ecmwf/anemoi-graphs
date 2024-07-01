import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.hexagonal import create_hexagonal_nodes
from anemoi.graphs.generate.icosahedral import create_icosahedral_nodes

logger = logging.getLogger(__name__)


class BaseNodeBuilder(ABC):
    """Base class for node builders."""

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
        return torch.tensor(coords, dtype=torch.float32)

    def transform(self, graph: HeteroData, name: str, attr_config: DotDict) -> HeteroData:
        graph = self.register_nodes(graph, name)
        graph = self.register_attributes(graph, name, attr_config)
        return graph


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset."""

    def __init__(self, dataset: DotDict) -> None:
        logger.info("Reading the dataset from %s.", dataset)
        self.ds = open_dataset(dataset)

    def get_coordinates(self) -> torch.Tensor:
        return self.reshape_coords(self.ds.latitudes, self.ds.longitudes)


class NPZFileNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids."""

    def __init__(self, resolution: str, grid_definition_path: str) -> None:
        self.resolution = resolution
        self.grid_definition_path = grid_definition_path
        self.grid_definition = np.load(Path(self.grid_definition_path) / f"grid-{self.resolution}.npz")

    def get_coordinates(self) -> np.ndarray:
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords


class RefinedIcosahedralNodes(BaseNodeBuilder, ABC):
    """Processor mesh based on a triangular mesh.

    It is based on the icosahedral mesh, which is a mesh of triangles that covers the sphere.

    Parameters
    ----------
    resolution : list[int] | int
        Refinement level of the mesh.
    np_dtype : np.dtype, optional
        The numpy data type to use, by default np.float32.
    """

    def __init__(
        self,
        resolution: Union[int, list[int]],
        np_dtype: np.dtype = np.float32,
    ) -> None:
        self.np_dtype = np_dtype

        if isinstance(resolution, int):
            self.resolutions = list(range(resolution + 1))
        else:
            self.resolutions = resolution

        super().__init__()

    def get_coordinates(self) -> np.ndarray:
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return coords_rad[self.node_ordering]

    @abstractmethod
    def create_nodes(self) -> np.ndarray: ...

    def register_attributes(self, graph: HeteroData, name: str, config: DotDict) -> HeteroData:
        graph[name]["resolutions"] = self.resolutions
        graph[name]["nx_graph"] = self.nx_graph
        graph[name]["node_ordering"] = self.node_ordering
        # TODO: AOI mask builder is not used in the current implementation.
        return super().register_attributes(graph, name, config)


class TriRefinedIcosahedralNodes(RefinedIcosahedralNodes):
    """It depends on the trimesh Python library."""

    def create_nodes(self) -> np.ndarray:
        # TODO: AOI mask builder is not used in the current implementation.
        return create_icosahedral_nodes(resolutions=self.resolutions)


class HexRefinedIcosahedralNodes(RefinedIcosahedralNodes):
    """It depends on the h3 Python library."""

    def create_nodes(self) -> np.ndarray:
        return create_hexagonal_nodes(self.resolutions)
