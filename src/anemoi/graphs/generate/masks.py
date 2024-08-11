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

    def __init__(self, reference_node_name: str, margin_radius_km: float = 100, mask_attr_name: str = None):

        self.nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

    def get_reference_coords(self, graph: HeteroData) -> np.ndarray:
        coords_rad = graph[self.reference_node_name].x.numpy()
        if self.mask_attr_name is not None:
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]

        return coords_rad

    def fit_coords(self, coords_rad: np.ndarray):
        """Fit the KNN model to the coordinates in radians."""
        self.nearest_neighbour.fit(coords_rad)

    def fit(self, graph: HeteroData):
        """Fit the KNN model to the nodes of interest."""
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f" ({self.mask_attr_name})"

        coords_rad = self.get_reference_coords(graph)
        self.fit_coords(coords_rad)
        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )

    def get_mask(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute a mask based on the distance to the reference nodes."""

        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius_km
        return mask