import numpy as np
import pandas as pd
import pytest
from anemoi.utils.config import DotDict
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.attributes import EdgeFeaturesTorch
from anemoi.graphs.edges.builder import CutOffEdges
from anemoi.graphs.edges.builder import KNNEdges
from anemoi.graphs.edges.builder import MultiScaleEdges
from anemoi.graphs.nodes.attributes import AreaWeights
from anemoi.graphs.nodes.builder import DataframeNodes
from anemoi.graphs.nodes.builder import TriNodes


@pytest.fixture
def obs_df(num_obs: int = 200) -> pd.DataFrame:
    """Create a fake dataframe with observations."""
    latlons = np.random.rand(num_obs, 2)
    df = pd.DataFrame()
    df[["cos_latitude", "cos_longitude"]] = np.cos(latlons)
    df[["sin_latitude", "sin_longitude"]] = np.sin(latlons)
    df["obsvalue_1"] = np.random.rand(num_obs)
    return df


def test_dynamic_graph_creation(obs_df: pd.DataFrame):
    # Create the backbone graph
    node_attrs = DotDict({"node_attrs": {"_target_": AreaWeights(norm="unit-max")}})
    edge_attrs = DotDict({"edge_attrs": {"_target_": EdgeFeaturesTorch}})
    backbone_graph = TriNodes(resolution=4, name="hidden").update_graph(HeteroData(), node_attrs)
    backbone_graph = MultiScaleEdges(x_hops=1, source_name="hidden", target_name="hidden").update_graph(
        backbone_graph, edge_attrs
    )

    # Training loop. Iterate over different sets of observations
    dynamic_graph = DataframeNodes(obs_df, attribute_columns=["obsvalue_1"], name="data").update_graph(
        backbone_graph.copy()
    )
    dynamic_graph = CutOffEdges(cutoff_factor=0.6, source_name="data", target_name="hidden").update_graph(
        dynamic_graph, edge_attrs
    )
    dynamic_graph = KNNEdges(num_nearest_neighbours=3, source_name="hidden", target_name="data").update_graph(
        dynamic_graph, edge_attrs
    )

    assert backbone_graph.node_types == ["hidden"]
    assert backbone_graph.edge_types == [("hidden", "to", "hidden")]
    assert backbone_graph.edge_attrs() == ["edge_index", "edge_attrs"]

    assert dynamic_graph.node_types == ["hidden", "data"]
    assert dynamic_graph.edge_types == [("hidden", "to", "hidden"), ("data", "to", "hidden"), ("hidden", "to", "data")]
    assert dynamic_graph.edge_attrs() == ["edge_index", "edge_attrs"]
