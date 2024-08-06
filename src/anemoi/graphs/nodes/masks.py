import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS


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

    def fit(self, graph: HeteroData):
        """Fit the KNN model to the nodes of interest."""
        coords_rad = graph[self.reference_node_name].x.numpy()
        if self.mask_attr_name is not None:
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]

        self.nearest_neighbour.fit(coords_rad)

    def get_mask(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute a mask based on the distance to the reference nodes."""

        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius_km
        return mask
