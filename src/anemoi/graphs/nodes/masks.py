import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS


class KNNAreaMaskBuilder:
    """Class to build a mask based on distance to masked reference nodes using KNN."""

    def __init__(self, reference_node_name: str, margin_radius_km: float, mask_attr_name: str):

        self.nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

    def fit(self, graph: HeteroData):
        coords_rad = graph[self.reference_node_name].x.numpy()
        mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
        self.nearest_neighbour.fit(coords_rad[mask, :])

    def get_mask(self, coords_rad: np.ndarray):

        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius_km
        return mask
