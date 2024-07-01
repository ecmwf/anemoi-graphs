import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import TriIcosahedralEdges
from anemoi.graphs.nodes import TriRefinedIcosahedralNodes


class TestTriIcosahedralEdgesInit:
    def test_init(self):
        """Test TriIcosahedralEdges initialization."""
        assert isinstance(TriIcosahedralEdges("test_nodes", 1), TriIcosahedralEdges)

    @pytest.mark.parametrize("xhops", [-0.5, "hello", None, -4])
    def test_fail_init(self, xhops: str):
        """Test TriIcosahedralEdges initialization with invalid cutoff."""
        with pytest.raises(AssertionError):
            TriIcosahedralEdges("test_nodes", xhops)


class TestTriIcosahedralEdgesTransform:

    @pytest.fixture()
    def ico_graph(self) -> HeteroData:
        """Return a HeteroData object with TriRefinedIcosahedralNodes."""
        graph = HeteroData()
        graph = TriRefinedIcosahedralNodes(0).transform(graph, "test_nodes", {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    def test_transform_same_src_dst_nodes(self, ico_graph: HeteroData):
        """Test TriIcosahedralEdges transform method."""

        tri_icosahedral_edges = TriIcosahedralEdges("test_nodes", 1)
        graph = tri_icosahedral_edges.transform(ico_graph)
        assert ("test_nodes", "to", "test_nodes") in graph.edge_types

    def test_transform_fail_nodes(self, ico_graph: HeteroData):
        """Test TriIcosahedralEdges transform method with wrong node type."""
        tri_icosahedral_edges = TriIcosahedralEdges("fail_nodes", 1)
        with pytest.raises(AssertionError):
            tri_icosahedral_edges.transform(ico_graph)
