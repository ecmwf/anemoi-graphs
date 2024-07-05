import pytest
import torch

from anemoi.graphs.edges.attributes import EdgeDirection


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-std"])
@pytest.mark.parametrize("luse_rotated_features", [True, False])
def test_directional_features(graph_nodes_and_edges, norm, luse_rotated_features: bool):
    """Test EdgeDirection compute method."""
    edge_attr_builder = EdgeDirection(norm=norm, luse_rotated_features=luse_rotated_features)
    edge_attr = edge_attr_builder.compute(graph_nodes_and_edges, "test_nodes", "test_nodes")
    assert isinstance(edge_attr, torch.Tensor)


def test_fail_directional_features(graph_nodes_and_edges):
    """Test EdgeDirection compute method."""
    edge_attr_builder = EdgeDirection()
    with pytest.raises(AttributeError):
        edge_attr_builder.compute(graph_nodes_and_edges, "test_nodes", "unknown_nodes")
