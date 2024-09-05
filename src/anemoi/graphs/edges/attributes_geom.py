import logging

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.typing import Size

from anemoi.graphs.utils import haversine_distance_torch

LOGGER = logging.getLogger(__name__)


class EdgeFeatureBuilder(MessagePassing):
    """Computes edge features for bipartite graphs."""

    def __init__(self) -> None:
        super().__init__()
        self._idx_lat = 0
        self._idx_lon = 1

    def forward(self, x: PairTensor, edge_index: Adj, size: Size = None):
        return self.propagate(edge_index, x=x, size=size)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # Haversine distance (unit sphere)
        edge_f1 = haversine_distance_torch(x_i, x_j)

        # l1 normalization
        edge_f1 = edge_f1 / torch.sum(torch.abs(edge_f1))

        # Forward bearing, in radians: https://www.movable-type.co.uk/scripts/latlong.html
        a11 = torch.cos(x_i[:, self._idx_lat]) * torch.sin(x_j[:, self._idx_lat])
        a12 = (
            torch.sin(x_i[:, self._idx_lat])
            * torch.cos(x_j[:, self._idx_lat])
            * torch.cos(x_j[..., self._idx_lon] - x_i[..., self._idx_lon])
        )
        a1 = a11 - a12
        a2 = torch.sin(x_j[..., self._idx_lon] - x_i[..., self._idx_lon]) * torch.cos(x_j[:, self._idx_lat])
        edge_f2 = torch.atan2(a2, a1)

        # loc_target - loc_source
        edge_f3 = x_i - x_j

        return torch.cat([edge_f1[:, None], edge_f2[:, None], edge_f3], dim=-1)

    def aggregate(self, edge_features: torch.Tensor) -> torch.Tensor:
        return edge_features
