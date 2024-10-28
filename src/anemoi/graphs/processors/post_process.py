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

import torch
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class PostProcessor(ABC):

    @abstractmethod
    def update_graph(self, graph: HeteroData) -> HeteroData:
        raise NotImplementedError(f"The {self.__class__.__name__} class does not implement the method update_graph().")


class RemoveUnconnectedNodes(PostProcessor):
    """Remove unconnected nodes in the graph.

    Attributes
    ----------
    nodes_name: str
        Name of the unconnected nodes to remove.
    ignore: str, optional
        Name of an attribute to ignore when removing nodes. Nodes with
        this attribute set to True will not be removed.
    save_mask_indices_to_attr: str, optional
        Name of the attribute to save the mask indices. If provided,
        the indices of the kept nodes will be saved in this attribute.

    Methods
    -------
    compute_mask(graph)
        Compute the mask of the connected nodes.
    prune_graph(graph, mask)
        Prune the nodes with the specified mask.
    add_attribute(graph, mask)
        Add an attribute to the graph with the indices of the mask.
    update_graph(graph)
        Post-process the graph.
    """

    def __init__(
        self,
        nodes_name: str,
        ignore: str | None,
        save_mask_indices_to_attr: str | None,
    ) -> None:
        self.nodes_name = nodes_name
        self.ignore = ignore
        self.save_mask_indices_to_attr = save_mask_indices_to_attr

    def compute_mask(self, graph: HeteroData) -> torch.Tensor:
        """Compute the mask of connected nodes."""
        nodes = graph[self.nodes_name]
        connected_mask = torch.zeros(nodes.num_nodes, dtype=torch.bool)

        if self.ignore is not None:
            LOGGER.info(f"The nodes with {self.ignore}=True will not be removed.")
            connected_mask[nodes[self.ignore].bool().squeeze()] = True

        for (source_name, _, target_name), edges in graph.edge_items():
            if source_name == self.nodes_name:
                connected_mask[edges.edge_index[0]] = True

            if target_name == self.nodes_name:
                connected_mask[edges.edge_index[1]] = True

        return connected_mask

    def removing_nodes(self, graph: HeteroData, mask: torch.Tensor) -> HeteroData:
        """Remove nodes based on the mask passed."""
        for attr_name in graph[self.nodes_name].node_attrs():
            graph[self.nodes_name][attr_name] = graph[self.nodes_name][attr_name][mask]

        return graph

    def update_edge_indices(self, graph: HeteroData, idx_mapping: dict[int, int]) -> HeteroData:
        """Update the edge indices to the new position of the nodes."""
        for edges_name in graph.edge_types:
            if edges_name[0] == self.nodes_name:
                graph[edges_name].edge_index[0] = graph[edges_name].edge_index[0].apply_(idx_mapping.get)

            if edges_name[2] == self.nodes_name:
                graph[edges_name].edge_index[1] = graph[edges_name].edge_index[1].apply_(idx_mapping.get)

        return graph

    def prune_graph(self, graph: HeteroData, mask: torch.Tensor) -> HeteroData:
        """Prune the nodes of the graph."""
        LOGGER.info(f"Removing {(~mask).sum()} nodes from {self.nodes_name}.")

        # Pruning nodes
        graph = self.removing_nodes(graph, mask)

        # Updating edge indices
        idx_mapping = dict(zip(torch.where(mask)[0].tolist(), list(range(mask.sum()))))
        graph = self.update_edge_indices(graph, idx_mapping)

        return graph

    def add_attribute(self, graph: HeteroData, mask: torch.Tensor) -> HeteroData:
        """Add an attribute of the mask indices as node attribute."""
        if self.save_mask_indices_to_attr is not None:
            LOGGER.info(
                f"An attribute {self.save_mask_indices_to_attr} has been added with the indices to mask the nodes from the original graph."
            )
            graph[self.nodes_name][self.save_mask_indices_to_attr] = torch.where(mask)[0]

        return graph

    def update_graph(self, graph: HeteroData) -> HeteroData:
        """Update graph.

        Parameters
        ----------
        graph: HeteroData
            The graph to post-process.

        Returns
        -------
        HeteroData
            The post-processed graph.
        """
        connected_mask = self.compute_mask(graph)
        graph = self.prune_graph(graph, connected_mask)
        graph = self.add_attribute(graph, connected_mask)
        return graph
