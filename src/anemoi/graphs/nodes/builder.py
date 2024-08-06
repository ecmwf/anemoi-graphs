import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import networkx as nx
import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.hexagonal import create_hexagonal_nodes
from anemoi.graphs.generate.icosahedral import create_icosahedral_nodes
from anemoi.graphs.nodes.masks import KNNAreaMaskBuilder

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

    def register_nodes(self, graph: HeteroData) -> None:
        """Register nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the nodes.
        """
        graph[self.name].x = self.get_coordinates()
        graph[self.name].node_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, config: Optional[DotDict] = None) -> HeteroData:
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

    def update_graph(self, graph: HeteroData, attr_config: Optional[DotDict] = None) -> HeteroData:
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
        graph = self.register_nodes(graph)

        if attr_config is None:
            return graph

        graph = self.register_attributes(graph, attr_config)

        return graph


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    dataset : zarr.core.Array
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

    def __init__(self, dataset: DotDict, name: str) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = open_dataset(dataset)
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        return self.reshape_coords(self.dataset.latitudes, self.dataset.longitudes)


class LimitedAreaZarrDatasetNodes(ZarrDatasetNodes):
    """Nodes from Zarr dataset."""

    def __init__(
        self, name: str, lam_dataset: str, forcing_dataset: str, thinning: int = 1, adjust: str = "all"
    ) -> None:
        dataset_config = {
            "cutout": [{"dataset": lam_dataset, "thinning": thinning}, {"dataset": forcing_dataset}],
            "adjust": adjust,
        }
        super().__init__(dataset_config, name)
        self.n_cutout, self.n_other = self.dataset.grids

    def register_attributes(self, graph: HeteroData, config: DotDict) -> None:
        # this is a mask to cutout the LAM area
        graph[self.name]["cutout"] = torch.tensor([True] * self.n_cutout + [False] * self.n_other, dtype=bool).reshape(
            (-1, 1)
        )
        return super().register_attributes(graph, config)


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

    def __init__(self, resolution: str, grid_definition_path: str, name: str) -> None:
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
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords


class LimitedAreaNPZFileNodes(NPZFileNodes):
    """Nodes from NPZ defined grids using an area of interest."""

    def __init__(
        self,
        resolution: str,
        grid_definition_path: str,
        name: str,
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
    ) -> None:

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, grid_definition_path, name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        LOGGER.info(
            "Limiting the processor mesh to a radius of %.2f km from the output mesh.",
            self.aoi_mask_builder.margin_radius_km,
        )
        aoi_mask = self.aoi_mask_builder.get_mask(coords)

        LOGGER.info("Dropping %d nodes from the processor mesh.", len(aoi_mask) - aoi_mask.sum())
        coords = coords[aoi_mask]

        return coords


class IcosahedralNodes(BaseNodeBuilder, ABC):
    """Nodes based on iterative refinements of an icosahedron.

    Attributes
    ----------
    resolution : list[int] | int
        Refinement level of the mesh.
    """

    def __init__(
        self,
        resolution: Union[int, list[int]],
        name: str,
    ) -> None:
        if isinstance(resolution, int):
            self.resolutions = list(range(resolution + 1))
        else:
            self.resolutions = resolution

        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return torch.tensor(coords_rad[self.node_ordering], dtype=torch.float32)

    @abstractmethod
    def create_nodes(self) -> Tuple[nx.DiGraph, np.ndarray, list[int]]: ...

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        graph[self.name]["_resolutions"] = self.resolutions
        graph[self.name]["_nx_graph"] = self.nx_graph
        graph[self.name]["_node_ordering"] = self.node_ordering
        graph[self.name]["_aoi_mask_builder"] = self.aoi_mask_builder
        return super().register_attributes(graph, config)


class LimitedAreaIcosahedralNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    Attributes
    ----------
    aoi_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        resolution: int | list[int],
        name: str,
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
    ) -> None:

        super().__init__(resolution, name)

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)


class TriNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the trimesh Python library.
    """

    def create_nodes(self) -> Tuple[nx.Graph, np.ndarray, list[int]]:
        return create_icosahedral_nodes(resolutions=self.resolutions)


class HexNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the h3 Python library.
    """

    def create_nodes(self) -> Tuple[nx.Graph, np.ndarray, list[int]]:
        return create_hexagonal_nodes(self.resolutions)


class LimitedAreaTriNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the trimesh Python library.

    Parameters
    ----------
    aoi_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def create_nodes(self) -> Tuple[nx.Graph, np.ndarray, list[int]]:
        return create_icosahedral_nodes(resolutions=self.resolutions, aoi_mask_builder=self.aoi_mask_builder)


class LimitedAreaHexNodes(LimitedAreaIcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron using an area of interest.

    It depends on the h3 Python library.

    Parameters
    ----------
    aoi_mask_builder: KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def create_nodes(self) -> Tuple[nx.Graph, np.ndarray, list[int]]:
        return create_hexagonal_nodes(self.resolutions, aoi_mask_builder=self.aoi_mask_builder)


class HEALPixNodes(BaseNodeBuilder):
    """Nodes from HEALPix grid.

    HEALPix is an acronym for Hierarchical Equal Area isoLatitude Pixelization of a sphere.

    Attributes
    ----------
    resolution : int
        The resolution of the grid.

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

    def __init__(self, resolution: int, name: str) -> None:
        """Initialize the HEALPixNodes builder."""
        self.resolution = resolution
        super().__init__(name)

        assert isinstance(resolution, int), "Resolution must be an integer."
        assert resolution > 0, "Resolution must be positive."

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            Coordinates of the nodes, in radians.
        """
        import healpy as hp

        spatial_res_degrees = hp.nside2resol(2**self.resolution, arcmin=True) / 60
        LOGGER.info(f"Creating HEALPix nodes with resolution {spatial_res_degrees:.2} deg.")

        npix = hp.nside2npix(2**self.resolution)
        hpxlon, hpxlat = hp.pix2ang(2**self.resolution, range(npix), nest=True, lonlat=True)

        return self.reshape_coords(hpxlat, hpxlon)


class LimitedAreaHEALPixNodes(HEALPixNodes):
    """Nodes from HEALPix grid using an area of interest."""

    def __init__(
        self,
        resolution: str,
        name: str,
        reference_node_name: str,
        mask_attr_name: str,
        margin_radius_km: float = 100.0,
    ) -> None:

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        LOGGER.info(
            'Limiting the "%s" nodes to a radius of %.2f km from the nodes of interest.',
            self.name,
            self.aoi_mask_builder.margin_radius_km,
        )
        aoi_mask = self.aoi_mask_builder.get_mask(coords)

        LOGGER.info('Masking out %d nodes from "%s".', len(aoi_mask) - aoi_mask.sum(), self.name)
        coords = coords[aoi_mask]

        return coords
