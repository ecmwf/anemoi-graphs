# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import BBoxAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class LimitedAreaRectilinearNodes(BaseNodeBuilder):
    """Limited area Rectilinear grid."""

    def __init__(
        self,
        resolution: float,
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_degrees: float = 1.0,
    ):
        super().__init__(name)
        self.resolution = resolution
        self.area_mask_builder = BBoxAreaMaskBuilder(reference_node_name, margin_radius_degrees, mask_attr_name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"bbox", "resolution"}

    def register_nodes(self, graph: HeteroData) -> None:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> torch.Tensor:
        lons = torch.arange(self.area_mask_builder.bbox[0], self.area_mask_builder.bbox[2], self.resolution)
        lats = torch.arange(self.area_mask_builder.bbox[1], self.area_mask_builder.bbox[3], self.resolution)

        longitudes, latitudes = torch.meshgrid(lons, lats, indexing="xy")

        return self.reshape_coords(latitudes.flatten(), longitudes.flatten())
