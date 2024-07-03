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
from anemoi.graphs.nodes.masks import KNNAreaMaskBuilder

logger = logging.getLogger(__name__)


class BaseNodeBuilder(ABC):
    """Base class for node builders."""

    def __init__(self) -> None:
        self.aoi_mask_builder = None

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
        self.dataset = open_dataset(dataset)

    def get_coordinates(self) -> torch.Tensor:
        return self.reshape_coords(self.dataset.latitudes, self.dataset.longitudes)


class LimitedAreaZarrDatasetNodes(ZarrDatasetNodes):
    """Nodes from Zarr dataset."""

    def __init__(self, lam_dataset: str, forcing_dataset: str, thinning: int = 1, adjust: str = "all") -> None:
        dataset_config = {
            "cutout": [{"dataset": lam_dataset, "thinning": thinning}, {"dataset": forcing_dataset}],
            "adjust": adjust,
        }
        super().__init__(dataset_config)
        self.n_cutout, self.n_other = self.dataset.grids

    def register_attributes(self, graph: HeteroData, name: str, config: DotDict) -> None:
        # this is a mask to cutout the LAM area
        graph[name]["cutout"] = torch.tensor([True] * self.n_cutout + [False] * self.n_other, dtype=bool).reshape(
            (-1, 1)
        )
        return super().register_attributes(graph, name, config)


class NPZFileNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids."""

    def __init__(self, resolution: str, grid_definition_path: str) -> None:
        self.resolution = resolution
        self.grid_definition_path = grid_definition_path
        self.grid_definition = np.load(Path(self.grid_definition_path) / f"grid-{self.resolution}.npz")

    def get_coordinates(self) -> np.ndarray:
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords


class AreaNPZFileNodes(NPZFileNodes):
    """Processor mesh based on an NPZ defined grids using an area of interest."""

    def __init__(
        self,
        resolution: str,
        grid_definition_path: str,
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
    ) -> None:

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, grid_definition_path)

    def register_nodes(self, graph: HeteroData, name: str) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph, name)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        logger.info(
            "Limiting the processor mesh to a radius of %.2f km from the output mesh.",
            self.aoi_mask_builder.margin_radius_km,
        )
        aoi_mask = self.aoi_mask_builder.get_mask(np.deg2rad(coords))

        logger.info("Dropping %d nodes from the processor mesh.", len(aoi_mask) - aoi_mask.sum())
        coords = coords[aoi_mask]

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
        # TODO: Discuss np_dtype
        self.np_dtype = np_dtype

        if isinstance(resolution, int):
            self.resolutions = list(range(resolution + 1))
        else:
            self.resolutions = resolution

        super().__init__()

    def get_coordinates(self) -> torch.Tensor:
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return torch.tensor(coords_rad[self.node_ordering])

    @abstractmethod
    def create_nodes(self) -> np.ndarray: ...

    def register_attributes(self, graph: HeteroData, name: str, config: DotDict) -> HeteroData:
        graph[name]["resolutions"] = self.resolutions
        graph[name]["nx_graph"] = self.nx_graph
        graph[name]["node_ordering"] = self.node_ordering
        graph[name]["aoi_mask_builder"] = self.aoi_mask_builder
        return super().register_attributes(graph, name, config)


class TriRefinedIcosahedralNodes(RefinedIcosahedralNodes):
    """It depends on the trimesh Python library."""

    def create_nodes(self) -> np.ndarray:
        # TODO: AOI mask builder is not used in the current implementation.
        return create_icosahedral_nodes(resolutions=self.resolutions)


class HexRefinedIcosahedralNodes(RefinedIcosahedralNodes):
    """It depends on the h3 Python library."""

    def create_nodes(self) -> np.ndarray:
        # TODO: AOI mask builder is not used in the current implementation.
        return create_hexagonal_nodes(self.resolutions)


class AreaTriRefinedIcosahedralNodes(TriRefinedIcosahedralNodes):
    """Class to build icosahedral nodes with a limited area of interest."""

    def __init__(
        self,
        resolution: int | list[int],
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
        np_dtype: np.dtype = np.float32,
    ) -> None:

        super().__init__(resolution, np_dtype)

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData, name: str) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph, name)


class AreaHexRefinedIcosahedralNodes(HexRefinedIcosahedralNodes):
    """Class to build icosahedral nodes with a limited area of interest."""

    def __init__(
        self,
        resolution: int | list[int],
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
        np_dtype: np.dtype = np.float32,
    ) -> None:

        super().__init__(resolution, np_dtype)

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData, name: str) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph, name)
