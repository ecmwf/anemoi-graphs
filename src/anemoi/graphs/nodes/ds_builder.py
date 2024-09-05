from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
import sys
import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.hexagonal import create_hexagonal_nodes
from anemoi.graphs.generate.icosahedral import create_icosahedral_nodes
from anemoi.graphs.nodes.builder import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class ZarrDownscalingDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    dataset : zarr.core.Array
        The dataset.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, dataset: DotDict, name: str, idx_in: int = None) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = open_dataset(dataset)
        self.idx_in = idx_in
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        if self.name == "in":
            return self.reshape_coords(
                self.dataset.latitudes[0][self.idx_in],
                self.dataset.longitudes[0][self.idx_in],
            )
        elif self.name == "out":
            return self.reshape_coords(
                self.dataset.latitudes[1],
                self.dataset.longitudes[1],
            )
        else:
            sys.exit("IO not recognized")
