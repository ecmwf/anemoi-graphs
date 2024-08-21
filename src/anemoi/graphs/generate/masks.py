from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS

LOGGER = logging.getLogger(__name__)


class KNNAreaMaskBuilder:
    """Class to build a mask based on distance to masked reference nodes using KNN.

    Attributes
    ----------
    nearest_neighbour : NearestNeighbors
        Nearest neighbour object to compute the KNN.
    margin_radius_km : float
        Maximum distance to the reference nodes to consider a node as valid, in kilometers. Defaults to 100 km.
    reference_node_name : str
        Name of the reference nodes in the graph to consider for the Area Mask.
    mask_attr_name : str
        Name of a node to attribute to mask the reference nodes, if desired. Defaults to consider all reference nodes.

    Methods
    -------
    fit(graph: HeteroData)
        Fit the KNN model to the reference nodes.
    get_mask(coords_rad: np.ndarray) -> np.ndarray
        Get the mask for the nodes based on the distance to the reference nodes.
    """

    def __init__(self, reference_node_name: str, margin_radius_km: float = 100, mask_attr_name: str | None = None):
        assert isinstance(margin_radius_km, (int, float)), "The margin radius must be a number."
        assert margin_radius_km > 0, "The margin radius must be positive."

        self.nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

    def fit(self, graph: HeteroData):
        """Fit the KNN model to the nodes of interest."""
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in the graph.'
        reference_mask_str = self.reference_node_name

        coords_rad = graph[self.reference_node_name].x.numpy()
        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in the reference nodes.'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]
            reference_mask_str += f" ({self.mask_attr_name})"

        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )
        self.nearest_neighbour.fit(coords_rad)

    def get_mask(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute a mask based on the distance to the reference nodes."""

        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius_km
        return mask
