from abc import ABC
from abc import abstractmethod

import hydra
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

import logging

logger = logging.getLogger(__name__)


def generate_graph(graph_config):
    graph = HeteroData()

    for name, nodes_cfg in graph_config.nodes.items():
        graph = instantiate(nodes_cfg.node_type).transform(graph, name, nodes_cfg.get("attributes", {}))

    for edges_cfg in graph_config.edges:
        graph = instantiate(edges_cfg.edge_type, **edges_cfg.nodes).transform(graph, edges_cfg.get("attributes", {}))

    return graph


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):

    graph = generate_graph(config)

    return graph


if __name__ == "__main__":
    main()
