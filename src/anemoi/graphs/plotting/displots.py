import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.prepare import _get_edge_attribute_dims
from anemoi.graphs.plotting.prepare import _get_node_attribute_dims

LOGGER = logging.getLogger(__name__)


def plot_dist_node_attributes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Figure with the distribution of the node attributes.

    Each row represents a node type and each column an attribute dimension.
    """
    num_nodes = len(graph.node_types)
    attr_dims = _get_node_attribute_dims(graph)
    dim_attrs = sum(attr_dims.values())

    if dim_attrs == 0:
        LOGGER.warning("No node attributes found in the graph.")
        return None

    # Define the layout
    _, axs = plt.subplots(num_nodes, dim_attrs, figsize=(10 * len(graph.node_types), 10))
    if axs.ndim == 1:
        axs = axs.reshape(num_nodes, dim_attrs)

    for i, (nodes_name, nodes_store) in enumerate(graph.node_items()):
        for j, (attr_name, attr_values) in enumerate(attr_dims.items()):
            for dim in range(attr_values):
                if attr_name in nodes_store:
                    axs[i, j + dim].hist(nodes_store[attr_name][:, dim].float(), bins=50)
                    if j + dim == 0:
                        axs[i, j + dim].set_ylabel(nodes_name)
                    if i == 0:
                        axs[i, j + dim].set_title(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                    elif i == num_nodes - 1:
                        axs[i, j + dim].set_xlabel(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                else:
                    axs[i, j + dim].set_axis_off()

    plt.suptitle("Node attributes distribution", fontsize=14)
    plt.savefig(out_file)


def plot_dist_edge_attributes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Figure with the distribution of the edge attributes.

    Each row represents a edge type and each column an attribute dimension.
    """
    num_edges = len(graph.edge_types)
    attr_dims = _get_edge_attribute_dims(graph)
    dim_attrs = sum(attr_dims.values())

    if dim_attrs == 0:
        LOGGER.warning("No edge attributes found in the graph.")
        return None

    # Define the layout
    _, axs = plt.subplots(num_edges, dim_attrs, figsize=(10 * len(graph.edge_types), 10))
    if axs.ndim == 1:
        axs = axs.reshape(num_edges, dim_attrs)

    for i, (edge_name, edge_store) in enumerate(graph.edge_items()):
        for j, (attr_name, attr_values) in enumerate(attr_dims.items()):
            for dim in range(attr_values):
                if attr_name in edge_store:
                    axs[i, j + dim].hist(edge_store[attr_name][:, dim].float(), bins=50)
                    if j + dim == 0:
                        axs[i, j + dim].set_ylabel("".join(edge_name).replace("to", " --> "))
                    if i == 0:
                        axs[i, j + dim].set_title(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                    elif i == num_edges - 1:
                        axs[i, j + dim].set_xlabel(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                else:
                    axs[i, j + dim].set_axis_off()

    plt.suptitle("Edge attributes distribution", fontsize=14)
    plt.savefig(out_file)
