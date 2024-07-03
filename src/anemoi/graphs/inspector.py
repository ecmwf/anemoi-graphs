import logging
import math
import os
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from anemoi.utils.humanize import bytes
from anemoi.utils.humanize import number
from anemoi.utils.text import table

from anemoi.graphs.plotting.displots import plot_dist_edge_attributes
from anemoi.graphs.plotting.displots import plot_dist_node_attributes
from anemoi.graphs.plotting.interactive_html import plot_interactive_nodes
from anemoi.graphs.plotting.interactive_html import plot_interactive_subgraph
from anemoi.graphs.plotting.interactive_html import plot_orphan_nodes

logger = logging.getLogger(__name__)


class GraphInspectorTool:
    """Inspect the graph."""

    def __init__(
        self,
        path: Union[str, Path],
        output_path: Path,
        show_attribute_distributions: Optional[bool] = True,
        **kwargs,
    ):
        self.path = path
        self.graph = torch.load(self.path)
        self.output_path = output_path
        self.show_attribute_distributions = show_attribute_distributions

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        assert self.output_path.exists(), f"Path {self.output_path} does not exist."
        assert self.output_path.is_dir(), f"Path {self.output_path} is not a directory."
        assert os.access(self.output_path, os.W_OK), f"Path {self.output_path} is not writable."

    @property
    def total_size(self):
        """Total size of the tensors in the graph (in bytes)."""
        total_size = 0

        for node_store in self.graph.node_stores:
            for value in node_store.values():
                if isinstance(value, torch.Tensor):
                    total_size += value.numel() * value.element_size()

        for edge_store in self.graph.edge_stores:
            for value in edge_store.values():
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
                    number(nodes.num_nodes),
                    ", ".join(attributes),
                    sum(nodes[attr].shape[1] for attr in attributes),
                    number(nodes.x[:, 0].min().item() / 2 / math.pi * 360),
                    number(nodes.x[:, 0].max().item() / 2 / math.pi * 360),
                    number(nodes.x[:, 1].min().item() / 2 / math.pi * 360),
                    number(nodes.x[:, 1].max().item() / 2 / math.pi * 360),
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
            - Total dimension of the attributes.
            - List of attribute names.
        """
        edge_summary = []
        for (src_nodes, _, dst_nodes), edges in self.graph.edge_items():
            attributes = edges.edge_attrs()
            attributes.remove("edge_index")

            edge_summary.append(
                [
                    src_nodes,
                    dst_nodes,
                    number(edges.num_edges),
                    sum(edges[attr].shape[1] for attr in attributes),
                    ", ".join(attributes),
                ]
            )
        return edge_summary

    def describe(self) -> None:
        """Describe the graph."""
        print()
        print(f"📦 Path       : {self.path}")
        print(f"💽 Size       : {bytes(self.total_size)} ({number(self.total_size)})")
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
                header=["Source", "Destination", "Num. edges", "Attribute dim", "Attributes"],
                align=["<", "<", ">", ">", ">"],
                margin=3,
            )
        )
        print("🔋 Graph ready.")
        print()

    def run_all(self):
        """Run all the inspector methods."""
        self.describe()

        if self.show_attribute_distributions:
            plot_dist_edge_attributes(self.graph, self.output_path / "distribution_edge_attributes.png")
            plot_dist_node_attributes(self.graph, self.output_path / "distribution_node_attributes.png")

        plot_orphan_nodes(self.graph, self.output_path / "orphan_nodes.html")

        logger.info("Saving interactive plots of nodes ...")
        for nodes_name, nodes_store in self.graph.node_items():
            ofile = self.output_path / f"{nodes_name}_nodes.html"
            title = f"Map of {nodes_name} nodes"
            plot_interactive_nodes(title, nodes_store.x[:, 0].numpy(), nodes_store.x[:, 1].numpy(), out_file=ofile)

        logger.info("Saving interactive plots of subgraphs ...")
        for src_nodes, _, dst_nodes in self.graph.edge_types:
            ofile = self.output_path / f"{src_nodes}_to_{dst_nodes}.html"
            plot_interactive_subgraph(self.graph, (src_nodes, dst_nodes), out_file=ofile)
