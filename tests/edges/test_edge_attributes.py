import pytest
import torch

from anemoi.graphs.edges.attributes import DirectionalFeatures


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-sum", "unit-std"])
@pytest.mark.parametrize("luse_rotated_features", [True, False])
def test_directional_features(graph_nodes_and_edges, norm, luse_rotated_features: bool):
    """Test DirectionalFeatures compute method."""
    edge_attr_builder = DirectionalFeatures(norm=norm, luse_rotated_features=luse_rotated_features)
    edge_attr = edge_attr_builder(graph_nodes_and_edges, "test_nodes", "test_nodes")
    assert isinstance(edge_attr, torch.Tensor)


def test_fail_directional_features(graph_nodes_and_edges):
    """Test DirectionalFeatures compute method."""
    edge_attr_builder = DirectionalFeatures()
    with pytest.raises(AttributeError):
        edge_attr_builder(graph_nodes_and_edges, "test_nodes", "unknown_nodes")
