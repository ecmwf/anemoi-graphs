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
from abc import ABC
from abc import abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS

LOGGER = logging.getLogger(__name__)


class BaseAreaMaskBuilder(ABC):
    """Abstract class for area masking."""

    def __init__(self, reference_node_name: str, margin_radius: float = 100, mask_attr_name: str | None = None):
        assert isinstance(margin_radius, (int, float)), "The margin radius must be a number."
        assert margin_radius > 0, "The margin radius must be positive."

        self.margin_radius = margin_radius
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

    def get_reference_coords(self, graph: HeteroData) -> np.ndarray:
        """Retrive coordinates from the reference nodes."""
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in the graph.'

        coords_rad = graph[self.reference_node_name].x.numpy()
        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in the reference nodes.'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]

        return coords_rad

    @abstractmethod
    def fit_coords(self, coords_rad: np.ndarray): ...

    def fit(self, graph: HeteroData):
        """Fit the KNN model to the nodes of interest."""
        # Prepare string for logging
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f" ({self.mask_attr_name})"

        # Fit to the reference nodes
        coords_rad = self.get_reference_coords(graph)
        self.fit_coords(coords_rad)

        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )


class KNNAreaMaskBuilder(BaseAreaMaskBuilder):
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
    fit_coords(coords_rad: np.ndarray)
        Fit the KNN model to the coordinates in radians.
    fit(graph: HeteroData)
        Fit the KNN model to the reference nodes.
    get_mask(coords_rad: np.ndarray) -> np.ndarray
        Get the mask for the nodes based on the distance to the reference nodes.
    """

    reference_node_name: str
    mask_attr_name: str | None
    margin_radius_km: float
    nearest_neighbour: NearestNeighbors

    def __init__(self, reference_node_name: str, margin_radius_km: float = 100, mask_attr_name: str | None = None):
        super().__init__(reference_node_name, margin_radius_km, mask_attr_name)
        self.nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)

    def fit_coords(self, coords_rad: np.ndarray):
        """Fit the KNN model to the coordinates in radians."""
        self.nearest_neighbour.fit(coords_rad)

    def get_mask(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute a mask based on the distance to the reference nodes."""
        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius
        return mask


class BBoxAreaMaskBuilder(BaseAreaMaskBuilder):
    """Class to build a mask based on distance to masked reference nodes using bounding box."""

    reference_node_name: str
    mask_attr_name: str | None
    margin_radius_degrees: float
    bbox: tuple[float, float, float, float]

    def __init__(self, reference_node_name: str, margin_radius_degrees: float = 1.0, mask_attr_name: str | None = None):
        super().__init__(reference_node_name, margin_radius_degrees, mask_attr_name)
        self.bbox = None

    def fit_coords(self, coords_rad: np.ndarray) -> None:
        """Compute the bounding box, in degrees.

        Parameters
        ----------
        coords_rad : np.ndarray
            Coordinates, in radians.
        """
        coords = np.rad2deg(coords_rad)
        lat_min, lon_min = coords.min(axis=0) - self.margin_radius
        lat_max, lon_max = coords.max(axis=0) + self.margin_radius
        self.bbox = lon_min, lat_min, lon_max, lat_max
