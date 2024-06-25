import logging
import os

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def generate_graph(graph_config: DotDict) -> HeteroData:
    graph = HeteroData()

    for name, nodes_cfg in graph_config.nodes.items():
        graph = instantiate(nodes_cfg.node_type).transform(graph, name, nodes_cfg.get("attributes", {}))

    for edges_cfg in graph_config.edges:
        graph = instantiate(edges_cfg.edge_type, **edges_cfg.nodes).transform(graph, edges_cfg.get("attributes", {}))

    return graph


class GraphCreator:
    def __init__(
        self,
        path,
        config=None,
        cache=None,
        print=print,
        overwrite=False,
        **kwargs,
    ):
        self.path = path  # Output path
        self.config = config
        self.cache = cache
        self.print = print
        self.overwrite = overwrite

    def init(self):
        assert os.path.exists(self.config), f"Path {self.config} does not exist."

        if self._path_readable() and not self.overwrite:
            raise Exception(f"{self.path} already exists. Use overwrite=True to overwrite.")

    def load(self) -> HeteroData:
        config = DotDict.from_file(self.config)
        graph = generate_graph(config)
        return graph

    def save(self, graph: HeteroData) -> None:
        if not os.path.exists(self.path) or self.overwrite:
            torch.save(graph, self.path)
            self.print(f"Graph saved at {self.path}.")

    def create(self):
        self.init()
        graph = self.load()
        self.save(graph)

    def _path_readable(self) -> bool:
        import torch

        try:
            torch.load(self.path, "r")
            return True
        except FileNotFoundError:
            return False
