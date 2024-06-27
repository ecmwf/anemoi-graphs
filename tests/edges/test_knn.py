import pytest

from anemoi.graphs.edges import KNNEdgeBuilder


def test_init():
    """Test CutOffEdgeBuilder initialization."""
    KNNEdgeBuilder("test_nodes1", "test_nodes2", 3)


@pytest.mark.parametrize("num_nearest_neighbours", [-1, 2.6, "hello", None])
def test_fail_init(num_nearest_neighbours: str):
    """Test KNNEdgeBuilder initialization with invalid number of nearest neighbours."""
    with pytest.raises(AssertionError):
        KNNEdgeBuilder("test_nodes1", "test_nodes2", num_nearest_neighbours)


def test_knn(graph_with_nodes):
    """Test KNNEdgeBuilder."""
    builder = KNNEdgeBuilder("test_nodes", "test_nodes", 3)
    graph = builder.transform(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types
