import math
from itertools import chain
from pathlib import Path
from typing import Union

import torch
from anemoi.utils.humanize import bytes
from anemoi.utils.text import table


class GraphDescriptor:
    """Class for descripting the graph."""

    def __init__(self, path: Union[str, Path], **kwargs):
        self.path = path
        self.graph = torch.load(self.path)

    @property
    def total_size(self):
        """Total size of the tensors in the graph (in bytes)."""
        total_size = 0

        for store in chain(self.graph.node_stores, self.graph.edge_stores):
            for value in store.values():
                if isinstance(value, torch.Tensor):
                    total_size += value.numel() * value.element_size()

        return total_size

    def get_node_summary(self) -> list[list]:
        """Summary of the nodes in the graph.

        Returns
        -------
        list[list]
            Returns a list for each subgraph with the following information:
            - Node name.
            - Number of nodes.
            - List of attribute names.
            - Total dimension of the attributes.
            - Min. latitude.
            - Max. latitude.
            - Min. longitude.
            - Max. longitude.
        """
        node_summary = []
        for name, nodes in self.graph.node_items():
            attributes = nodes.node_attrs()
            attributes.remove("x")

            node_summary.append(
                [
                    name,
                    nodes.num_nodes,
                    ", ".join(attributes),
                    sum(nodes[attr].shape[1] for attr in attributes if isinstance(nodes[attr], torch.Tensor)),
                    nodes.x[:, 0].min().item() / 2 / math.pi * 360,
                    nodes.x[:, 0].max().item() / 2 / math.pi * 360,
                    nodes.x[:, 1].min().item() / 2 / math.pi * 360,
                    nodes.x[:, 1].max().item() / 2 / math.pi * 360,
                ]
            )
        return node_summary

    def get_edge_summary(self) -> list[list]:
        """Summary of the edges in the graph.

        Returns
        -------
        list[list]
            Returns a list for each subgraph with the following information:
            - Source node name.
            - Destination node name.
            - Number of edges.
            - Number of isolated source nodes.
            - Number of isolated target nodes.
            - Total dimension of the attributes.
            - List of attribute names.
        """
        edge_summary = []
        for (src_name, _, dst_name), edges in self.graph.edge_items():
            attributes = edges.edge_attrs()
            attributes.remove("edge_index")

            edge_summary.append(
                [
                    src_name,
                    dst_name,
                    edges.num_edges,
                    self.graph[src_name].num_nodes - len(torch.unique(edges.edge_index[0])),
                    self.graph[dst_name].num_nodes - len(torch.unique(edges.edge_index[1])),
                    sum(edges[attr].shape[1] for attr in attributes),
                    ", ".join(attributes),
                ]
            )
        return edge_summary

    def describe(self) -> None:
        """Describe the graph."""
        print()
        print(f"📦 Path       : {self.path}")
        print(f"💽 Size       : {bytes(self.total_size)} ({self.total_size})")
        print()
        print("🪩 Nodes summary")
        print()
        print(
            table(
                self.get_node_summary(),
                header=[
                    "Nodes name",
                    "Num. nodes",
                    "Attributes",
                    "Attribute dim",
                    "Min. latitude",
                    "Max. latitude",
                    "Min. longitude",
                    "Max. longitude",
                ],
                align=["<", ">", ">", ">", ">", ">", ">", ">"],
                margin=3,
            )
        )
        print()
        print()
        print("🌐  Edges summary")
        print()
        print(
            table(
                self.get_edge_summary(),
                header=[
                    "Source",
                    "Target",
                    "Num. edges",
                    "Isolated Source",
                    "Isolated Target",
                    "Attribute dim",
                    "Attributes",
                ],
                align=["<", "<", ">", ">", ">", ">", ">"],
                margin=3,
            )
        )
        print("🔋 Graph ready.")
        print()
