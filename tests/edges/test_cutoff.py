import pytest

from anemoi.graphs.edges import CutOffEdgeBuilder


def test_init():
    """Test CutOffEdgeBuilder initialization."""
    CutOffEdgeBuilder("test_nodes1", "test_nodes2", 0.5)


@pytest.mark.parametrize("cutoff_factor", [-0.5, "hello", None])
def test_fail_init(cutoff_factor: str):
    """Test CutOffEdgeBuilder initialization with invalid cutoff."""
    with pytest.raises(AssertionError):
        CutOffEdgeBuilder("test_nodes1", "test_nodes2", cutoff_factor)
