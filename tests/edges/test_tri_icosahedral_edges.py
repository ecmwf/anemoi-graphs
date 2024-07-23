import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import TriIcosahedralEdges
from anemoi.graphs.nodes import TriNodes


class TestTriIcosahedralEdgesInit:
    def test_init(self):
        """Test TriIcosahedralEdges initialization."""
        assert isinstance(TriIcosahedralEdges("test_nodes", "test_nodes", 1), TriIcosahedralEdges)

    @pytest.mark.parametrize("x_hops", [-0.5, "hello", None, -4])
    def test_fail_init(self, x_hops: str):
        """Test TriIcosahedralEdges initialization with invalid x_hops."""
        with pytest.raises(AssertionError):
            TriIcosahedralEdges("test_nodes", "test_nodes", x_hops)

    def test_fail_init_diff_nodes(self):
        """Test TriIcosahedralEdges initialization with invalid nodes."""
        with pytest.raises(AssertionError):
            TriIcosahedralEdges("test_nodes", "test_nodes2", 0)


class TestTriIcosahedralEdgesTransform:

    @pytest.fixture()
    def ico_graph(self) -> HeteroData:
        """Return a HeteroData object with TriRefinedIcosahedralNodes."""
        graph = HeteroData()
        graph = TriNodes(1, "test_nodes").update_graph(graph, {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    def test_transform_same_src_dst_nodes(self, ico_graph: HeteroData):
        """Test TriIcosahedralEdges update method."""

        tri_icosahedral_edges = TriIcosahedralEdges("test_nodes", "test_nodes", 1)
        graph = tri_icosahedral_edges.update_graph(ico_graph)
        assert ("test_nodes", "to", "test_nodes") in graph.edge_types

    def test_transform_fail_nodes(self, ico_graph: HeteroData):
        """Test TriIcosahedralEdges update method with wrong node type."""
        tri_icosahedral_edges = TriIcosahedralEdges("fail_nodes", "fail_nodes", 1)
        with pytest.raises(AssertionError):
            tri_icosahedral_edges.update_graph(ico_graph)
