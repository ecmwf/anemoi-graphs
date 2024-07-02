import pytest
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import HexagonalEdges
from anemoi.graphs.nodes import HexRefinedIcosahedralNodes


class TestTriIcosahedralEdgesInit:
    def test_init(self):
        """Test TriIcosahedralEdges initialization."""
        assert isinstance(HexagonalEdges("test_nodes"), HexagonalEdges)

    @pytest.mark.parametrize("depth_children", [-0.5, "hello", None, -4])
    def test_fail_init(self, depth_children: str):
        """Test HexagonalEdges initialization with invalid cutoff."""
        with pytest.raises(AssertionError):
            HexagonalEdges("test_nodes", True, depth_children)


class TestTriIcosahedralEdgesTransform:

    @pytest.fixture()
    def ico_graph(self) -> HeteroData:
        """Return a HeteroData object with HexRefinedIcosahedralNodes."""
        graph = HeteroData()
        graph = HexRefinedIcosahedralNodes(0).transform(graph, "test_nodes", {})
        graph["fail_nodes"].x = [1, 2, 3]
        graph["fail_nodes"].node_type = "FailNodes"
        return graph

    def test_transform_same_src_dst_nodes(self, ico_graph: HeteroData):
        """Test HexagonalEdges transform method."""

        tri_icosahedral_edges = HexagonalEdges("test_nodes")
        graph = tri_icosahedral_edges.transform(ico_graph)
        assert ("test_nodes", "to", "test_nodes") in graph.edge_types

    def test_transform_fail_nodes(self, ico_graph: HeteroData):
        """Test HexagonalEdges transform method with wrong node type."""
        tri_icosahedral_edges = HexagonalEdges("fail_nodes")
        with pytest.raises(AssertionError):
            tri_icosahedral_edges.transform(ico_graph)
