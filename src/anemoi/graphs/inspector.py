from dataclasses import dataclass
from pathlib import Path
from typing import Union
import torch
import logging
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.plots import plot_connection_stats_graphdata
from anemoi.graphs.plotting.plots import plot_html_for_subgraph
from anemoi.graphs.plotting.plots import plot_nodes
from anemoi.graphs.plotting.plots import plot_orphan_nodes

logger = logging.getLogger(__name__)


@dataclass
class GraphInspectorTool:
    """Inspect the graph."""
    graph: Union[HeteroData, str]
    output_path: Path

    def __post_init__(self):
        if not isinstance(self.graph, HeteroData):
            self.graph = torch.load(self.graph)

    def run_all(self):
        """Run all the inspector methods."""
        plot_orphan_nodes(self.graph, self.output_path / "orphan_nodes.html")
        plot_connection_stats_graphdata(self.graph, self.output_path / "subgraphs_stats.png")

        for nodes_name, nodes_store in self.graph.node_items():
            ofile = self.output_path / f"{nodes_name}_nodes.html"
            title = f"Map of {nodes_name} nodes"
            plot_nodes(title, nodes_store.x[:, 0].numpy(), nodes_store.x[:, 1].numpy(), out_file=ofile)

        for src_nodes, _, dst_nodes in self.graph.edge_types:
            ofile = self.output_path / f"{src_nodes}_to_{dst_nodes}.html"
            plot_html_for_subgraph(self.graph, (src_nodes, dst_nodes))


if __name__ == "__main__":
    import torch
    graph = torch.load("my_graph.pt")
    GraphInspectorTool(graph, Path(".")).run_all()
