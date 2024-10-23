import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.attributes import CutOutMask
from anemoi.graphs.nodes.attributes import UniformWeights
from anemoi.graphs.nodes.builders import from_file


def test_init(mocker, mock_zarr_dataset_cutout):
    """Test ZarrDatasetNodes initialization with cutout."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.ZarrDatasetNodes(forcing_dataset="global.zarr", lam_dataset="lam.zarr", name="test_nodes")

    assert isinstance(node_builder, from_file.BaseNodeBuilder)
    assert isinstance(node_builder, from_file.ZarrDatasetNodes)


def test_register_nodes(mocker, mock_zarr_dataset_cutout):
    """Test ZarrDatasetNodes register correctly the nodes with cutout operation."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.ZarrDatasetNodes(
        OmegaConf.create({"cutout": ["global.zarr", "lam.zarr"]}), name="test_nodes"
    )
    graph = HeteroData()

    graph = node_builder.register_nodes(graph)

    assert graph["test_nodes"].x is not None
    assert isinstance(graph["test_nodes"].x, torch.Tensor)
    assert graph["test_nodes"].x.shape == (node_builder.ds.num_nodes, 2)
    assert graph["test_nodes"].node_type == "ZarrDatasetNodes"


@pytest.mark.parametrize("attr_class", [UniformWeights, AreaWeights, CutOutMask])
def test_register_attributes(mocker, mock_zarr_dataset_cutout, graph_with_nodes: HeteroData, attr_class):
    """Test ZarrDatasetNodes register correctly the weights with cutout operation."""
    mocker.patch.object(from_file, "open_dataset", return_value=mock_zarr_dataset_cutout)
    node_builder = from_file.ZarrDatasetNodes(
        OmegaConf.create({"cutout": ["global.zarr", "lam.zarr"]}), name="test_nodes"
    )
    config = {"test_attr": {"_target_": f"anemoi.graphs.nodes.attributes.{attr_class.__name__}"}}

    graph = node_builder.register_attributes(graph_with_nodes, config)

    assert graph["test_nodes"]["test_attr"] is not None
    assert isinstance(graph["test_nodes"]["test_attr"], torch.Tensor)
    assert graph["test_nodes"]["test_attr"].shape[0] == graph["test_nodes"].x.shape[0]
