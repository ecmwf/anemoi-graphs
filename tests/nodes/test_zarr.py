import pytest
import torch
import zarr
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders import from_file


def test_init(mocker, mock_zarr_dataset):
    """Test ZarrDatasetNodes initialization."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")

    assert isinstance(node_builder, from_file.BaseNodeBuilder)
    assert isinstance(node_builder, from_file.ZarrDatasetNodes)


def test_fail_init():
    """Test ZarrDatasetNodes initialization with invalid resolution."""
    with pytest.raises(zarr.errors.PathNotFoundError):
        from_file.ZarrDatasetNodes("invalid_path.zarr", name="test_nodes")


def test_register_nodes(mocker, mock_zarr_dataset):
    """Test ZarrDatasetNodes register correctly the nodes."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (node_builder.dataset.num_nodes, 2)
    assert graph["test_nodes"].node_type == "ZarrDatasetNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights])
def test_register_attributes(mocker, graph_with_nodes: HeteroData, attr_class):
    """Test ZarrDatasetNodes register correctly the weights."""
    mocker.patch.object(from_file, "open_dataset", return_value=None)
    node_builder = from_file.ZarrDatasetNodes("dataset.zarr", name="test_nodes")
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
