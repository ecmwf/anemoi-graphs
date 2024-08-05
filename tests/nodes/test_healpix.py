import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builder import BaseNodeBuilder
from anemoi.graphs.nodes.builder import HEALPixNodes


@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_init(resolution: int):
    """Test HEALPixNodes initialization."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    assert isinstance(node_builder, BaseNodeBuilder)
    assert isinstance(node_builder, HEALPixNodes)


@pytest.mark.parametrize("resolution", ["2", 4.3, -7])
def test_fail_init(resolution: int):
    """Test HEALPixNodes initialization with invalid resolution."""
    with pytest.raises(AssertionError):
        HEALPixNodes(resolution, "test_nodes")


@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_register_nodes(resolution: int):
    """Test HEALPixNodes register correctly the nodes."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape[1] == 2
    assert graph["test_nodes"].node_type == "HEALPixNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights])
@pytest.mark.parametrize("resolution", [2, 5, 7])
def test_register_attributes(graph_with_nodes: HeteroData, attr_class, resolution: int):
    """Test HEALPixNodes register correctly the weights."""
    node_builder = HEALPixNodes(resolution, "test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
