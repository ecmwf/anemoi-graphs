# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

import torch

from anemoi.graphs.plotting.displots import plot_distribution_edge_attributes
from anemoi.graphs.plotting.displots import plot_distribution_node_attributes
from anemoi.graphs.plotting.displots import plot_distribution_node_derived_attributes
from anemoi.graphs.plotting.interactive_html import plot_interactive_nodes
from anemoi.graphs.plotting.interactive_html import plot_interactive_subgraph
from anemoi.graphs.plotting.interactive_html import plot_isolated_nodes

LOGGER = logging.getLogger(__name__)


class GraphInspector:
    """Inspect the graph.

    Attributes
    ----------
    path: Union[str, Path]
        Path to the graph file.
    output_path: Path
        Path to the output directory where the plots will be saved.
    show_attribute_distributions: Optional[bool]
        Whether to show the distribution of the node and edge attributes.
    show_nodes: Optional[bool]
        Whether to show the interactive plots of the nodes.
    """

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
            LOGGER.info("Saving distribution plots of node ande edge attributes ...")
            plot_distribution_node_derived_attributes(self.graph, self.output_path / "distribution_node_adjancency.png")
            plot_distribution_edge_attributes(self.graph, self.output_path / "distribution_edge_attributes.png")
            plot_distribution_node_attributes(self.graph, self.output_path / "distribution_node_attributes.png")

        if self.show_nodes:
            LOGGER.info("Saving interactive plots of nodes ...")
            for nodes_name in self.graph.node_types:
                plot_interactive_nodes(self.graph, nodes_name, out_file=self.output_path / f"{nodes_name}_nodes.html")
