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
from anemoi.graphs.plotting.interactive.edges import plot_interactive_subgraph
from anemoi.graphs.plotting.interactive.graph_3d import plot_downscale
from anemoi.graphs.plotting.interactive.graph_3d import plot_level
from anemoi.graphs.plotting.interactive.graph_3d import plot_upscale
from anemoi.graphs.plotting.interactive.nodes import plot_interactive_nodes
from anemoi.graphs.plotting.interactive.nodes import plot_isolated_nodes

LOGGER = logging.getLogger(__name__)


class GraphInspector:
    """Inspect the graph."""

    def __init__(
        self,
        path: Union[str, Path],
        output_path: Path,
        show_attribute_distributions: Optional[bool] = True,
        show_nodes: Optional[bool] = False,
        show_3d_graph: Optional[bool] = False,
        num_hidden_layers: Optional[int] = 1,
        **kwargs,
    ):
        self.path = path
        self.graph = torch.load(self.path)
        self.output_path = output_path
        self.show_attribute_distributions = show_attribute_distributions
        self.show_nodes = show_nodes
        self.show_3d_graph = show_3d_graph
        self.num_hidden_layers = num_hidden_layers

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

        if self.show_3d_graph:

            data_nodes = self.graph["data"].x
            hidden_nodes = []
            hidden_edges = []
            downscale_edges = []
            upscale_edges = []

            if self.num_hidden_layers > 1:

                data_to_hidden_edges = self.graph[("data", "to", "hidden_1")].edge_index

                for i in range(1, self.num_hidden_layers):
                    hidden_nodes.append(self.graph[f"hidden_{i}"].x)
                    hidden_edges.append(self.graph[(f"hidden_{i}", "to", f"hidden_{i}")].edge_index)
                    downscale_edges.append(self.graph[(f"hidden_{i}", "to", f"hidden_{i+1}")].edge_index)
                    upscale_edges.append(
                        self.graph[
                            (f"hidden_{self.num_hidden_layers+1-i}", "to", f"hidden_{self.num_hidden_layers-i}")
                        ].edge_index
                    )

                # Add hidden-most layer
                hidden_nodes.append(self.graph[f"hidden_{self.num_hidden_layers}"].x)
                hidden_edges.append(
                    self.graph[
                        (f"hidden_{self.num_hidden_layers}", "to", f"hidden_{self.num_hidden_layers}")
                    ].edge_index
                )
                # Add symbolic graphs for last layers of downscaling and upscaling -> they do not have edges
                downscale_edges.append(self.graph[(f"hidden_{self.num_hidden_layers}", "to", f"hidden_{i}")].edge_index)
                upscale_edges.append(self.graph[("hidden_1", "to", "data")].edge_index)

                hidden_to_data_edges = self.graph[("hidden_1", "to", "data")].edge_index

            else:
                data_to_hidden_edges = self.graph[("data", "to", "hidden")].edge_index
                hidden_nodes.append(self.graph["hidden"].x)
                hidden_edges.append(self.graph[("hidden", "to", "hidden")].edge_index)
                downscale_edges.append(self.graph[("data", "to", "hidden")].edge_index)
                upscale_edges.append(self.graph[("hidden", "to", "data")].edge_index)
                hidden_to_data_edges = self.graph[("hidden", "to", "data")].edge_index

            # Encoder
            ofile = self.output_path / "encoder.html"
            encoder_fig = plot_downscale(
                data_nodes,
                hidden_nodes,
                data_to_hidden_edges,
                downscale_edges,
                title="Downscaling",
                color="red",
                num_hidden=self.num_hidden_layers,
                x_range=[0, 0.4],
                y_range=[0, 0.4],
                z_range=[0, 0.4],
            )
            encoder_fig.write_html(ofile)

            # Processor
            ofile = self.output_path / "processor.html"
            level_fig = plot_level(
                data_nodes,
                hidden_nodes,
                data_to_hidden_edges,
                hidden_edges,
                title="Level Processing",
                color="green",
                num_hidden=self.num_hidden_layers,
                x_range=[0, 0.4],
                y_range=[0, 0.4],
                z_range=[0, 0.4],
            )
            level_fig.write_html(ofile)

            # Decoder
            ofile = self.output_path / "dencoder.html"
            decoder_fig = plot_upscale(
                data_nodes,
                hidden_nodes,
                hidden_to_data_edges,
                upscale_edges,
                title="Upscaling",
                color="blue",
                num_hidden=self.num_hidden_layers,
                x_range=[0, 0.4],
                y_range=[0, 0.4],
                z_range=[0, 0.4],
            )
            decoder_fig.write_html(ofile)
