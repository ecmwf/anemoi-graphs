import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class LimitedAreaRectilinearNodes(BaseNodeBuilder):
    """Limited area Rectangel"""

    bbox: tuple[float, float, float, float]

    def __init__(
        self,
        resolution: float,
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ):
        super().__init__(name)
        self.resolution = resolution
        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def set_bbox(self, graph: HeteroData) -> tuple[float, float, float, float]:
        coords = self.area_mask_builder.get_reference_coords(graph)
        lat_min, lon_min = coords.min(axis=1)
        lat_max, lon_max = coords.max(axis=1)
        self.bbox = lon_min, lat_min, lon_max, lat_max

    def register_nodes(self, graph: HeteroData) -> None:
        self.set_bbox(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> torch.Tensor:
        longitudes = torch.arange(self.bbox[0], self.bbox[2], self.resolution)
        latitudes = torch.arange(self.bbox[1], self.bbox[3], self.resolution)

        return self.reshape_coords(latitudes, longitudes)
