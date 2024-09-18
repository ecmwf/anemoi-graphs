from __future__ import annotations

import logging

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size

LOGGER = logging.getLogger(__name__)


def haversine_distance_torch(source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Haversine distance.

    Parameters
    ----------
    source_coords : torch.Tensor of shape (N, 2)
        Source coordinates in radians.
    target_coords : torch.Tensor of shape (N, 2)
        Destination coordinates in radians.

    Returns
    -------
    torch.Tensor of shape (N,)
        Haversine distance between source and destination coordinates.
    """
    dlat = target_coords[:, 0] - source_coords[:, 0]
    dlon = target_coords[:, 1] - source_coords[:, 1]
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(source_coords[:, 0]) * torch.cos(target_coords[:, 0]) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return c


class EdgeAttributeBuilderMixin:
    def forward(self, x: PairTensor, edge_index: Adj, size: Size = None):
        return self.propagate(edge_index, x=x, size=size)

    def aggregate(self, edge_features: torch.Tensor) -> torch.Tensor:
        return edge_features


class EdgeLength(MessagePassing, EdgeAttributeBuilderMixin):
    """Computes edge features for bipartite graphs."""

    def __init__(self, norm: str | None = None) -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1
        self.norm = norm

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_length = haversine_distance_torch(x_i, x_j)
        return edge_length[:, None]


class EdgeDirection(MessagePassing, EdgeAttributeBuilderMixin):
    """Computes edge features for bipartite graphs."""

    def __init__(self, rotated_features: bool = True, norm: str | None = None) -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1
        self.norm = norm
        self.rotated_features = rotated_features

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
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
        return edge_dir[:, None]
