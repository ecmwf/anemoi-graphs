import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builder import NPZFileNodes


@pytest.mark.parametrize("resolution", ["o16", "o48", "5km5"])
def test_init(mock_grids_path: tuple[str, int], resolution: str):
    """Test NPZNodes initialization."""
    grid_definition_path, _ = mock_grids_path
    node_builder = NPZFileNodes(resolution, grid_definition_path=grid_definition_path)
    assert isinstance(node_builder, NPZFileNodes)


@pytest.mark.parametrize("resolution", ["o17", 13, "ajsnb", None])
def test_fail_init_wrong_resolution(mock_grids_path: tuple[str, int], resolution: str):
    """Test NPZNodes initialization with invalid resolution."""
    grid_definition_path, _ = mock_grids_path
    with pytest.raises(FileNotFoundError):
        NPZFileNodes(resolution, grid_definition_path=grid_definition_path)


def test_fail_init_wrong_path():
    """Test NPZNodes initialization with invalid path."""
    with pytest.raises(FileNotFoundError):
        NPZFileNodes("o16", "invalid_path")


@pytest.mark.parametrize("resolution", ["o16", "o48", "5km5"])
def test_register_nodes(mock_grids_path: str, resolution: str):
    """Test NPZNodes register correctly the nodes."""
    graph = HeteroData()
    grid_definition_path, num_nodes = mock_grids_path
    node_builder = NPZFileNodes(resolution, grid_definition_path=grid_definition_path)

    graph = node_builder.register_nodes(graph, "test_nodes")

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (num_nodes, 2)
    assert graph["test_nodes"].node_type == "NPZFileNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights])
def test_register_attributes(graph_with_nodes: HeteroData, mock_grids_path: tuple[str, int], attr_class):
    """Test NPZNodes register correctly the weights."""
    grid_definition_path, _ = mock_grids_path
    node_builder = NPZFileNodes("o16", grid_definition_path=grid_definition_path)
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, "test_nodes", config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]