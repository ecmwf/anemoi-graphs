import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

lats = [-0.15, 0, 0.15]
lons = [0, 0.25, 0.5, 0.75]


class MockZarrDataset:
    """Mock Zarr dataset with latitudes and longitudes attributes."""

    def __init__(self, latitudes, longitudes):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.num_nodes = len(latitudes)


@pytest.fixture
def mock_zarr_dataset() -> MockZarrDataset:
    """Mock zarr dataset with nodes."""
    coords = 2 * torch.pi * np.array([[lat, lon] for lat in lats for lon in lons])
    return MockZarrDataset(latitudes=coords[:, 0], longitudes=coords[:, 1])


@pytest.fixture
def mock_grids_path(tmp_path) -> tuple[str, int]:
    """Mock grid_definition_path with files for 3 resolutions."""
    num_nodes = len(lats) * len(lons)
    for resolution in ["o16", "o48", "5km5"]:
        file_path = tmp_path / f"grid-{resolution}.npz"
        np.savez(file_path, latitudes=np.random.rand(num_nodes), longitudes=np.random.rand(num_nodes))
    return str(tmp_path), num_nodes


@pytest.fixture
def graph_with_nodes() -> HeteroData:
    """Graph with 12 nodes over the globe, stored in \"test_nodes\"."""
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    return graph


@pytest.fixture
def graph_nodes_and_edges() -> HeteroData:
    """Graph with 1 set of nodes and edges."""
    coords = np.array([[lat, lon] for lat in lats for lon in lons])
    graph = HeteroData()
    graph["test_nodes"].x = 2 * torch.pi * torch.tensor(coords)
    graph[("test_nodes", "to", "test_nodes")].edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    return graph


@pytest.fixture
def config_file(tmp_path) -> tuple[str, str]:
    """Mock grid_definition_path with files for 3 resolutions."""
    cfg = {
        "nodes": {
            "test_nodes": {
                "node_builder": {
                    "_target_": "anemoi.graphs.nodes.NPZFileNodeBuilder",
                    "grid_definition_path": str(tmp_path),
                    "resolution": "o16",
                },
            }
        },
        "edges": [
            {
                "nodes": {"src_name": "test_nodes", "dst_name": "test_nodes"},
                "edge_builder": {
                    "_target_": "anemoi.graphs.edges.KNNEdgeBuilder",
                    "num_nearest_neighbours": 3,
                },
                "attributes": {
                    "dist_norm": {
                        "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                        "norm": "l1",
                        "invert": True,
                    },
                    "directional_features": {"_target_": "anemoi.graphs.edges.attributes.DirectionalFeatures"},
                },
            },
        ],
    }
    file_name = "config.yaml"

    with (tmp_path / file_name).open("w") as file:
        yaml.dump(cfg, file)

    return tmp_path, file_name