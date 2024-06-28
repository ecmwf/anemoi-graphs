import logging
import os

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class GraphCreator:
    """Graph creator."""

    def __init__(
        self,
        path,
        config=None,
        cache=None,
        print=print,
        overwrite=False,
        **kwargs,
    ):
        if isinstance(config, str) or isinstance(config, os.PathLike):
            self.config = DotDict.from_file(self.config)
        else:
            self.config = config

        self.path = path  # Output path
        self.cache = cache
        self.print = print
        self.overwrite = overwrite

    def init(self):
        if self._path_readable() and not self.overwrite:
            raise Exception(f"{self.path} already exists. Use overwrite=True to overwrite.")

    def generate_graph(self) -> HeteroData:
        graph = HeteroData()
        for name, nodes_cfg in self.config.nodes.items():
            graph = instantiate(nodes_cfg.node_builder).transform(graph, name, nodes_cfg.get("attributes", {}))

        for edges_cfg in self.config.edges:
            graph = instantiate(edges_cfg.edge_builder, **edges_cfg.nodes).transform(
                graph, edges_cfg.get("attributes", {})
            )

        return graph

    def save(self, graph: HeteroData) -> None:
        if not os.path.exists(self.path) or self.overwrite:
            torch.save(graph, self.path)
            self.print(f"Graph saved at {self.path}.")

    def create(self) -> HeteroData:
        self.init()
        graph = self.generate_graph()
        self.save(graph)
        return graph

    def _path_readable(self) -> bool:
        import torch

        try:
            torch.load(self.path, "r")
            return True
        except FileNotFoundError:
            return False
