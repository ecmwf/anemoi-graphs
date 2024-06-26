import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from torch_geometric.data import HeteroData


@pytest.mark.parametrize("norm", [None, "l1", "l2", "unit-max", "unit-sum", "unit-std"])
def test_uniform_weights(graph_with_nodes: HeteroData, norm: str):
    """Test NPZNodes register correctly the weights."""
    config = {"_target_": "anemoi.graphs.nodes.weights.UniformWeights", "norm": norm}

    weights = instantiate(config).get_weights(graph_with_nodes["test_nodes"])

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


@pytest.mark.parametrize("norm", ["l3", "invalide"])
def test_uniform_weights_fail(graph_with_nodes: HeteroData, norm: str):
    """Test NPZNodes register correctly the weights."""
    config = {"_target_": "anemoi.graphs.nodes.weights.UniformWeights", "norm": norm}

    with pytest.raises(ValueError):
        instantiate(config).get_weights(graph_with_nodes["test_nodes"])


def test_area_weights(graph_with_nodes: HeteroData):
    """Test NPZNodes register correctly the weights."""
    config = {
        "_target_": "anemoi.graphs.nodes.weights.AreaWeights",
        "radius": 1.0,
        "centre": np.array([0, 0, 0]),
    }

    weights = instantiate(config).get_weights(graph_with_nodes["test_nodes"])

    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == graph_with_nodes["test_nodes"].x.shape[0]


@pytest.mark.parametrize("radius", [-1.0, "hello", None])
def test_area_weights_fail(graph_with_nodes: HeteroData, radius: float):
    config = {
        "_target_": "anemoi.graphs.nodes.weights.AreaWeights",
        "radius": radius,
        "centre": np.array([0, 0, 0]),
    }

    with pytest.raises(ValueError):
        instantiate(config).get_weights(graph_with_nodes["test_nodes"])
