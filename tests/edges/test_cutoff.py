import pytest

from anemoi.graphs.edges import CutOffEdges


def test_init():
    """Test CutOffEdges initialization."""
    CutOffEdges("test_nodes1", "test_nodes2", 0.5)


@pytest.mark.parametrize("cutoff_factor", [-0.5, "hello", None])
def test_fail_init(cutoff_factor: str):
    """Test CutOffEdges initialization with invalid cutoff."""
    with pytest.raises(AssertionError):
        CutOffEdges("test_nodes1", "test_nodes2", cutoff_factor)


def test_cutoff(graph_with_nodes):
    """Test CutOffEdges."""
    builder = CutOffEdges("test_nodes", "test_nodes", 0.5)
    graph = builder.update_graph(graph_with_nodes)
    assert ("test_nodes", "to", "test_nodes") in graph.edge_types
