import logging
import math
import os
from itertools import chain
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from anemoi.utils.humanize import bytes
from anemoi.utils.text import table

from anemoi.graphs.plotting.displots import plot_distribution_edge_attributes
from anemoi.graphs.plotting.displots import plot_distribution_node_attributes
from anemoi.graphs.plotting.interactive_html import plot_interactive_nodes
from anemoi.graphs.plotting.interactive_html import plot_interactive_subgraph
from anemoi.graphs.plotting.interactive_html import plot_isolated_nodes

LOGGER = logging.getLogger(__name__)


class GraphDescription:
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


class GraphInspectorTool:
    """Inspect the graph."""

    def __init__(
        self,
        path: Union[str, Path],
        output_path: Path,
        show_attribute_distributions: Optional[bool] = True,
        show_nodes: Optional[bool] = False,
        **kwargs,
    ):
        self.path = path
        self.graph = torch.load(self.path)
        self.output_path = output_path
        self.show_attribute_distributions = show_attribute_distributions
        self.show_nodes = show_nodes

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        os.makedirs(self.output_path, exist_ok=True)

        assert self.output_path.is_dir(), f"Path {self.output_path} is not a directory."
        assert os.access(self.output_path, os.W_OK), f"Path {self.output_path} is not writable."

    def inspect(self):
        """Run all the inspector methods."""
        LOGGER.info("Saving interactive plots of isolated nodes ...")
        plot_isolated_nodes(self.graph, self.output_path / "isolated_nodes.html")

        LOGGER.info("Saving interactive plots of subgraphs ...")
        for edges_subgraph in self.graph.edge_types:
            ofile = self.output_path / f"{edges_subgraph[0]}_to_{edges_subgraph[2]}.html"
            plot_interactive_subgraph(self.graph, edges_subgraph, out_file=ofile)

        if self.show_attribute_distributions:
            plot_distribution_edge_attributes(self.graph, self.output_path / "distribution_edge_attributes.png")
            plot_distribution_node_attributes(self.graph, self.output_path / "distribution_node_attributes.png")

        if self.show_nodes:
            LOGGER.info("Saving interactive plots of nodes ...")
            for nodes_name in self.graph.node_types:
                plot_interactive_nodes(self.graph, nodes_name, out_file=self.output_path / f"{nodes_name}_nodes.html")
