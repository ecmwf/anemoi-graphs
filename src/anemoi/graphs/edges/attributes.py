from __future__ import annotations

import logging

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size

from anemoi.graphs.utils import haversine_distance_torch

LOGGER = logging.getLogger(__name__)


class EdgeAttributeBuilderMixin:
    def forward(self, x: PairTensor, edge_index: Adj, size: Size = None):
        return self.propagate(edge_index, x=x, size=size)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return values
        elif self.norm == "l1":
            return values / values.abs().sum()
        elif self.norm == "l2":
            return values / values.norm(2)
        elif self.norm == "unit-max":
            return values / values.abs().max()
        elif self.norm == "unit-std":
            return values / values.std()

        raise ValueError(f"Unknown normalization {self.norm}")

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_features = self.compute_attribute(x_i, x_j)
        return self.normalize(edge_features[:, None])

    def aggregate(self, edge_features: torch.Tensor) -> torch.Tensor:
        return edge_features


class EdgeLength(MessagePassing, EdgeAttributeBuilderMixin):
    """Computes edge features for bipartite graphs."""

    def __init__(self, norm: str | None = None) -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1
        self.norm = norm

    def compute_attribute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance_torch(x_i, x_j)
        return edge_length


class EdgeDirection(MessagePassing, EdgeAttributeBuilderMixin):
    """Computes edge features for bipartite graphs."""

    def __init__(self, rotated_features: bool = True, norm: str | None = None) -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1
        self.norm = norm
        self.rotated_features = rotated_features

    def compute_attribute(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        if not self.rotated_features:
            return x_j - x_i

        # Forward bearing, in radians: https://www.movable-type.co.uk/scripts/latlong.html
        a11 = torch.cos(x_i[:, self._idx_lat]) * torch.sin(x_j[:, self._idx_lat])
        a12 = (
            torch.sin(x_i[:, self._idx_lat])
            * torch.cos(x_j[:, self._idx_lat])
            * torch.cos(x_j[..., self._idx_lon] - x_i[..., self._idx_lon])
        )
        a1 = a11 - a12
        a2 = torch.sin(x_j[..., self._idx_lon] - x_i[..., self._idx_lon]) * torch.cos(x_j[:, self._idx_lat])
        edge_dir = torch.atan2(a2, a1)
        return edge_dir
