import logging
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.plotting.prepare import get_edge_attribute_dims
from anemoi.graphs.plotting.prepare import get_node_attribute_dims

LOGGER = logging.getLogger(__name__)


def plot_distribution_node_attributes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Figure with the distribution of the node attributes.

    Each row represents a node type and each column an attribute dimension.
    """
    num_nodes = len(graph.node_types)
    attr_dims = get_node_attribute_dims(graph)
    plot_distribution_attributes(graph.node_items(), num_nodes, attr_dims, "Node", out_file)


def plot_distribution_edge_attributes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Figure with the distribution of the edge attributes.

    Each row represents a edge type and each column an attribute dimension.
    """
    num_edges = len(graph.edge_types)
    attr_dims = get_edge_attribute_dims(graph)
    plot_distribution_attributes(graph.edge_items(), num_edges, attr_dims, "Edge", out_file)


def plot_distribution_attributes(
    graph_items: Union[NodeStorage, EdgeStorage],
    num_items: int,
    attr_dims: dict,
    item_type: Literal["Edge", "Node"],
    out_file: Optional[Union[str, Path]] = None,
) -> None:
    """Figure with the distribution of the node and edge attributes.

    Each row represents a node or edge type and each column an attribute dimension.
    """
    dim_attrs = sum(attr_dims.values())

    if dim_attrs == 0:
        LOGGER.warning("No edge attributes found in the graph.")
        return None

    # Define the layout
    _, axs = plt.subplots(num_items, dim_attrs, figsize=(10 * num_items, 10))
    if axs.ndim == 1:
        axs = axs.reshape(num_items, dim_attrs)

    for i, (item_name, item_store) in enumerate(graph_items):
        for j, (attr_name, attr_values) in enumerate(attr_dims.items()):
            for dim in range(attr_values):
                if attr_name in item_store:
                    axs[i, j + dim].hist(item_store[attr_name][:, dim].float(), bins=50)
                    if j + dim == 0:
                        axs[i, j + dim].set_ylabel("".join(item_name).replace("to", " --> "))
                    if i == 0:
                        axs[i, j + dim].set_title(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                    elif i == num_items - 1:
                        axs[i, j + dim].set_xlabel(attr_name if attr_values == 1 else f"{attr_name}_{dim}")
                else:
                    axs[i, j + dim].set_axis_off()

    plt.suptitle(f"{item_type} Attributes distribution", fontsize=14)
    plt.savefig(out_file)
