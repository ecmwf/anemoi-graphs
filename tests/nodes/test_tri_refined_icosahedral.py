import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes import builder


@pytest.mark.parametrize("resolution", [0, 2])
def test_init(resolution: list[int]):
    """Test TrirefinedIcosahedralNodes initialization."""

    node_builder = builder.TriRefinedIcosahedralNodes(resolution)
    assert isinstance(node_builder, builder.BaseNodeBuilder)
    assert isinstance(node_builder, builder.TriRefinedIcosahedralNodes)


def test_get_coordinates():
    """Test get_coordinates method."""
    node_builder = builder.TriRefinedIcosahedralNodes(2)
    coords = node_builder.get_coordinates()
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (162, 2)


def test_transform():
    """Test transform method."""
    node_builder = builder.TriRefinedIcosahedralNodes(1)
    graph = HeteroData()
    graph = node_builder.transform(graph, "test", {})
    assert "resolutions" in graph["test"]
    assert "nx_graph" in graph["test"]
    assert "node_ordering" in graph["test"]
    assert len(graph["test"]["node_ordering"]) == graph["test"].num_nodes