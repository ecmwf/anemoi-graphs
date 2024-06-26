import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch
from anemoi.utils.config import DotDict
import numpy as np
from torch_geometric.data import HeteroData
from hydra.utils import instantiate

from anemoi.graphs.edges.directional import directional_edge_features
from anemoi.graphs.normalizer import NormalizerMixin

logger = logging.getLogger(__name__)

class NodeAttributeBuilder():

    def transform(self, graph: HeteroData, graph_config: DotDict): 

        for name, nodes_cfg in graph_config.nodes.items():
            graph = self.register_node_attributes(graph, name, nodes_cfg.get("attributes", {}))

    def register_node_attributes(self, graph: HeteroData, node_name: str, node_config: DotDict):
        assert node_name in graph.keys(), f"Node {node_name} does not exist in the graph."
        for attr_name, attr_cfg in node_config.items():
            graph[node_name][attr_name] = instantiate(attr_cfg).compute(graph, node_name) 
        return graph

class EdgeAttributeBuilder():

    def transform(self, graph: HeteroData, graph_config: DotDict):
        for edges_cfg in graph_config.edges:
            graph =  self.register_edge_attributes(graph, edges_cfg.nodes.src_name, edges_cfg.nodes.dst_name, edges_cfg.get("attributes", {}))
        return graph

    def register_edge_attributes(self, graph: HeteroData, src_name: str, dst_name: str, edge_config: DotDict):

        for attr_name, attr_cfg in edge_config.items():
            attr_values = instantiate(attr_cfg).compute(graph, src_name, dst_name)
            graph = self.register_edge_attribute(graph, src_name, dst_name, attr_name, attr_values)
        return graph
            
    def register_edge_attribute(self, graph: HeteroData, src_name: str, dst_name: str, attr_name: str, attr_values: torch.Tensor):
        num_edges = graph[(src_name, "to", dst_name)].num_edges
        assert ( attr_values.shape[0] == num_edges), f"Number of edge features ({attr_values.shape[0]}) must match number of edges ({num_edges})."

        graph[(src_name, "to", dst_name)][attr_name] = attr_values 
        return graph


class BaseEdgeAttribute(ABC, NormalizerMixin):
    norm: Optional[str] = None

    @abstractmethod
    def compute(self, graph: HeteroData, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        values = self.compute(*args, **kwargs)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        return self.normalize(values)


class DirectionalFeatures(BaseEdgeAttribute):
    norm: Optional[str] = None
    luse_rotated_features: bool = False

    def compute(self, graph: HeteroData, src_name: str, dst_name: str):
        edge_index = graph[(src_name, "to", dst_name)].edge_index
        src_coords = graph[src_name].x.numpy()[edge_index[0]].T
        dst_coords = graph[dst_name].x.numpy()[edge_index[1]].T
        edge_dirs = directional_edge_features(src_coords, dst_coords, self.luse_rotated_features).T
        return edge_dirs
