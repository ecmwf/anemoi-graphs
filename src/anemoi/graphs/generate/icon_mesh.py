# (C) Copyright 2024 ECMWF, DWD.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, the above institutions do not waive the privileges
# and immunities granted to it by virtue of its status as an intergovernmental
# organisation  nor does it submit to any jurisdiction.
#

import itertools
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import netCDF4
import numpy as np
import scipy
import torch
import torch_geometric
from typeguard import typechecked
from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


@typechecked
class NodeSet:
    """Stores nodes on the unit sphere."""

    id_iter: int = itertools.count()  # unique ID for each object
    gc_vertices: np.ndarray  # geographical (lat/lon) coordinates [rad], shape [:,2]

    def __init__(self, lon: np.ndarray, lat: np.ndarray):
        self.gc_vertices = np.column_stack((lon, lat))
        self.id = next(self.id_iter)

    @property
    def num_verts(self) -> int:
        return self.gc_vertices.shape[0]

    @cached_property
    def cc_vertices(self):
        """Cartesian coordinates [rad], shape [:,3]."""
        return self._gc_to_cartesian()

    def __add__(self, other: Self) -> Self:
        """concatenates two node sets."""
        gc_vertices = np.concatenate((self.gc_vertices, other.gc_vertices))
        return NodeSet(gc_vertices[:, 0], gc_vertices[:, 1])

    def __eq__(self, other: Self) -> bool:
        """Compares two node sets."""
        return self.id == other.id

    def _gc_to_cartesian(self, radius: float = 1.0) -> np.ndarray:
        """Returns Cartesian coordinates of the node set, shape [:,3]."""
        xyz = (
            radius * np.cos(lat_rad := self.gc_vertices[:, 1]) * np.cos(lon_rad := self.gc_vertices[:, 0]),
            radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        )
        return np.stack(xyz, axis=-1)


@typechecked
@dataclass
class EdgeID:
    """Stores additional categorical data for each edge (IDs for heterogeneous input)."""

    edge_id: np.ndarray
    num_classes: int

    def __add__(self, other: Self):
        """Concatenates two edge ID datasets."""
        assert self.num_classes == other.num_classes
        return EdgeID(edge_id=np.concatenate((self.edge_id, other.edge_id)), num_classes=self.num_classes)


@typechecked
class GeneralGraph:
    """Stores edges for a given node set, calculates edge features."""

    nodeset: NodeSet  # graph nodes
    edge_vertices: np.ndarray  # vertex indices for each edge, shape [:,2]

    def __init__(self, nodeset: NodeSet, bidirectional: bool, edge_vertices: np.ndarray):
        self.nodeset = nodeset
        # (optional) duplicate edges (bi-directional):
        if bidirectional:
            self.edge_vertices = np.concatenate([edge_vertices, np.fliplr(edge_vertices)])
        else:
            self.edge_vertices = edge_vertices

    @property
    def num_verts(self) -> int:
        return self.nodeset.num_verts

    @property
    def num_edges(self) -> int:
        return self.edge_vertices.shape[0]

    @cached_property
    def edge_features(self) -> torch.tensor:
        return torch.tensor(
            get_edge_attributes(
                self.nodeset.gc_vertices[self.edge_vertices[:, 1]],
                self.nodeset.cc_vertices[self.edge_vertices[:, 1]],
                self.nodeset.gc_vertices[self.edge_vertices[:, 0]],
                self.nodeset.cc_vertices[self.edge_vertices[:, 0]],
            ),
            dtype=torch.float32,
        )


@typechecked
class BipartiteGraph:
    """Graph defined on a pair of NodeSets."""

    nodeset: tuple[NodeSet, NodeSet]  # source and target node set
    edge_vertices: np.ndarray  # vertex indices for each edge, shape [:,2]
    edge_id: np.ndarray  # additional ID for each edge (markers for heterogeneous input)

    def __init__(self, nodeset: tuple[NodeSet, NodeSet], edge_vertices: np.ndarray, edge_id: Optional[EdgeID] = None):
        self.nodeset = nodeset
        self.edge_vertices = edge_vertices
        if edge_id is None:
            self.edge_id = None
        else:
            self.edge_id = edge_id

    @property
    def num_edges(self) -> int:
        return self.edge_vertices.shape[0]

    @cached_property
    def edge_features(self) -> torch.tensor:
        edge_feature_arr = torch.tensor(
            get_edge_attributes(
                self.nodeset[0].gc_vertices[self.edge_vertices[:, 0]],
                self.nodeset[0].cc_vertices[self.edge_vertices[:, 0]],
                self.nodeset[1].gc_vertices[self.edge_vertices[:, 1]],
                self.nodeset[1].cc_vertices[self.edge_vertices[:, 1]],
            ),
            dtype=torch.float32,
        )
        if self.edge_id is None:
            return edge_feature_arr
        else:
            # one-hot-encoded categorical data (`edge_id`)
            one_hot = torch_geometric.utils.one_hot(
                index=torch.tensor(self.edge_id.edge_id), num_classes=self.edge_id.num_classes
            )
            return torch.concatenate((one_hot, edge_feature_arr), dim=1)

    def set_constant_edge_id(self, edge_id: int, num_classes: int):
        self.edge_id = EdgeID(edge_id=np.full(self.num_edges, fill_value=edge_id), num_classes=num_classes)

    def __add__(self, other: "BipartiteGraph"):
        """Concatenates two bipartite graphs that share a common target node set.
        Shifts the node indices of the second bipartite graph.
        """

        if not self.nodeset[1] == other.nodeset[1]:
            raise ValueError("Only bipartite graphs with common target node set can be merged.")
        shifted_edge_vertices = other.edge_vertices
        shifted_edge_vertices[:, 0] += self.nodeset[0].num_verts
        # (Optional:) merge one-hot-encoded categorical data (`edge_id`)
        if None not in (self.edge_id, other.edge_id):
            edge_id = self.edge_id + other.edge_id
        else:
            edge_id = None
        return BipartiteGraph(
            nodeset=(self.nodeset[0] + other.nodeset[0], self.nodeset[1]),
            edge_vertices=np.concatenate((self.edge_vertices, shifted_edge_vertices)),
            edge_id=edge_id,
        )


@typechecked
class BipartiteGraphProximity(BipartiteGraph):
    """Graph defined on a pair of NodeSets, where the definition of the
    graph edges is based on geographical proximity (Euclidean distance
    in Cartesian coordinates).
    """

    def __init__(self, nodeset: tuple[NodeSet, NodeSet], radius: float, edge_id: Optional[EdgeID] = None):

        src, tgt = nodeset
        point_tree = scipy.spatial.KDTree(tgt.cc_vertices)
        neighbor_list = point_tree.query_ball_point(src.cc_vertices, r=radius)
        # turn `neighbor_list` into a list of pairs:
        nb_list = [(idx, x) for idx, x in enumerate(neighbor_list)]
        nb_list = np.asarray([(x, k) for x, y in nb_list for k in y])
        super().__init__(nodeset=nodeset, edge_id=edge_id, edge_vertices=nb_list)


@typechecked
class ICONMultiMesh(GeneralGraph):
    """Reads vertices and topology from an ICON grid file; creates multi-mesh."""

    uuidOfHGrid: str
    max_level: int
    nodeset: NodeSet  # set of ICON grid vertices
    cell_vertices: np.ndarray

    def __init__(self, icon_grid_filename: str, max_level: Optional[int] = None, iverbosity: int = 0):

        # open file, representing the finest level
        if iverbosity > 0:
            LOGGER.info(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:
            # read vertex coordinates
            vlon = read_coordinate_array(ncfile, "vlon", "vertex")
            vlat = read_coordinate_array(ncfile, "vlat", "vertex")

            edge_verts_fine = np.asarray(ncfile.variables["edge_vertices"][:] - 1, dtype=np.int64).transpose()
            assert ncfile.variables["edge_vertices"].dimensions == ("nc", "edge")

            cell_verts_fine = np.asarray(ncfile.variables["vertex_of_cell"][:] - 1, dtype=np.int64).transpose()
            assert ncfile.variables["vertex_of_cell"].dimensions == ("nv", "cell")

            reflvl_vertex = ncfile.variables["refinement_level_v"][:]
            assert ncfile.variables["refinement_level_v"].dimensions == ("vertex",)

            self.uuidOfHGrid = ncfile.uuidOfHGrid

        if max_level is not None:
            self.max_level = max_level
        else:
            self.max_level = reflvl_vertex.max()

        # generate edge-vertex relations for coarser levels:
        (edge_vertices, cell_vertices) = self._get_hierarchy_of_icon_edge_graphs(
            edge_verts_fine=edge_verts_fine,
            cell_verts_fine=cell_verts_fine,
            reflvl_vertex=reflvl_vertex,
            iverbosity=iverbosity,
        )
        # restrict edge-vertex list to multi_mesh level "max_level":
        if self.max_level < len(edge_vertices):
            (self.edge_vertices, self.cell_vertices, vlon, vlat) = self._restrict_multi_mesh_level(
                edge_vertices, cell_vertices, reflvl_vertex=reflvl_vertex, vlon=vlon, vlat=vlat
            )
        # store vertices as a `NodeSet`:
        self.nodeset = NodeSet(vlon, vlat)
        # concatenate edge-vertex lists (= edges of the multi-level mesh):
        multi_mesh_edges = np.concatenate([edges for edges in self.edge_vertices], axis=0)
        # generate multi-mesh graph data structure:
        super().__init__(nodeset=self.nodeset, bidirectional=True, edge_vertices=multi_mesh_edges)

    def _restrict_multi_mesh_level(
        self,
        edge_vertices: list[np.ndarray],
        cell_vertices: np.ndarray,
        reflvl_vertex: np.ndarray,
        vlon: np.ndarray,
        vlat: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Creates a new mesh with only the vertices at the desired level."""

        num_verts = reflvl_vertex.shape[0]
        vertex_idx = reflvl_vertex <= self.max_level
        vertex_glb2loc = np.zeros(num_verts, dtype=int)
        vertex_glb2loc[vertex_idx] = np.arange(vertex_idx.sum())
        return (
            [vertex_glb2loc[M] for M in edge_vertices[: self.max_level + 1]],
            # cell_vertices: preserve negative indices (incomplete cells)
            np.where(cell_vertices >= 0, vertex_glb2loc[cell_vertices], cell_vertices),
            vlon[vertex_idx],
            vlat[vertex_idx],
        )

    def _get_hierarchy_of_icon_edge_graphs(
        self, edge_verts_fine: np.ndarray, cell_verts_fine: np.ndarray, reflvl_vertex: np.ndarray, iverbosity: int = 1
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Returns a list of edge-vertex relations (coarsest to finest level)."""

        edge_vertices = [edge_verts_fine]  # list of edge-vertex relations (coarsest to finest level).

        num_verts = reflvl_vertex.shape[0]
        # edge-to-vertex adjacency matrix with 2 non-zero entries per row:
        e2v_matrix = convert_list_to_adjacency_matrix(edge_verts_fine, num_verts)
        # cell-to-vertex adjacency matrix with 3 non-zero entries per row:
        c2v_matrix = convert_list_to_adjacency_matrix(cell_verts_fine, num_verts)
        v2v_matrix = e2v_matrix.transpose() * e2v_matrix
        v2v_matrix.setdiag(np.ones(num_verts))  # vertices are self-connected

        s_v_coarse = scipy.sparse.diags(np.ones(num_verts), dtype=bool)

        # coarsen edge-vertex list from level `ilevel -> ilevel - 1`:
        for ilevel in reversed(range(1, reflvl_vertex.max() + 1)):
            if iverbosity > 0:
                LOGGER.info(f"  edges[{ilevel}] = {edge_vertices[0].shape[0] : >9}")

            # define edge selection matrix (selecting only edges of which have
            # exactly one coarse vertex):
            #
            # get a boolean mask, matching all edges where one of its vertices
            # has refinement level index `ilevel`:
            ref_level = reflvl_vertex[edge_vertices[0]] == ilevel
            edges_coarse = np.logical_xor(ref_level[:, 0], ref_level[:, 1])  # = bisected coarse edges
            idx_e2e = np.argwhere(edges_coarse).flatten()
            s_edges = selection_matrix(idx_e2e, edges_coarse.shape[0])

            # define vertex selection matrix selecting only vertices of
            # level `ilevel`:
            idx_v_fine = np.argwhere(reflvl_vertex == ilevel).flatten()
            s_v_fine = selection_matrix(idx_v_fine, num_verts)
            # define vertex selection matrix, selecting only vertices of
            # level < `ilevel`, by successively removing `s_fine` from an identity matrix.
            s_v_coarse.data[0][idx_v_fine] = False

            # create an adjacency matrix which links each fine level
            # vertex to its two coarser neighbor vertices:
            v2v_fine2coarse = s_v_fine * v2v_matrix * s_v_coarse
            # remove rows that have only one non-zero entry
            # (corresponding to incomplete parent triangles in LAM grids):
            csum = v2v_fine2coarse * np.ones((v2v_fine2coarse.shape[1], 1))
            s_v2v = selection_matrix(np.argwhere(csum == 2).flatten(), v2v_fine2coarse.shape[0])
            v2v_fine2coarse = s_v2v * v2v_fine2coarse

            # then construct the edges-to-parent-vertex adjacency matrix:
            parent_edge_verts = s_edges * e2v_matrix * v2v_fine2coarse
            # again, we have might have selected edges within
            # `s_edges` which are part of an incomplete parent edge
            # (LAM case). We filter these here:
            csum = parent_edge_verts * np.ones((parent_edge_verts.shape[1], 1))
            s_e2e = selection_matrix(np.argwhere(csum == 2).flatten(), parent_edge_verts.shape[0])
            parent_edge_verts = s_e2e * parent_edge_verts

            # note: the edges-vertex adjacency matrix still has duplicate
            # rows, since two child edges have the same parent edge.
            edge_verts_coarse = convert_adjacency_matrix_to_list(parent_edge_verts, ncols_per_row=2)
            edge_vertices.insert(0, edge_verts_coarse)

            # store cell-to-vert adjacency matrix
            if ilevel > self.max_level:
                c2v_matrix = c2v_matrix * v2v_fine2coarse
                # similar to the treatment above, we need to handle
                # coarse LAM cells which are incomplete.
                csum = c2v_matrix * np.ones((c2v_matrix.shape[1], 1))
                s_c2c = selection_matrix(np.argwhere(csum == 3).flatten(), c2v_matrix.shape[0])
                c2v_matrix = s_c2c * c2v_matrix

            # replace edge-to-vertex and vert-to-vert adjacency matrices (for next level):
            if ilevel > 1:
                v2v_matrix = s_v_coarse * v2v_matrix * v2v_fine2coarse
                e2v_matrix = convert_list_to_adjacency_matrix(edge_verts_coarse, num_verts)

        # Fine-level cells outside of multi-mesh (LAM boundary)
        # correspond to empty rows in the adjacency matrix. We
        # substitute these by three additional, non-existent vertices:
        csum = 3 - c2v_matrix * np.ones((c2v_matrix.shape[1], 1))
        nvmax = c2v_matrix.shape[1]
        c2v_matrix = scipy.sparse.csr_matrix(scipy.sparse.hstack((c2v_matrix, csum, csum, csum)))

        # build a list of cell-vertices [1..num_cells,1..3] for all
        # fine-level cells:
        cell_vertices = convert_adjacency_matrix_to_list(c2v_matrix, remove_duplicates=False, ncols_per_row=3)

        # finally, translate non-existent vertices into "-1" indices:
        cell_vertices = np.where(cell_vertices >= nvmax, -np.ones(cell_vertices.shape, dtype=np.int32), cell_vertices)

        return (edge_vertices, cell_vertices)


@typechecked
class ICONCellDataGrid(BipartiteGraph):
    """Reads cell locations from an ICON grid file; builds grid-to-mesh edges based on ICON topology."""

    uuidOfHGrid: str
    nodeset: NodeSet  # set of ICON cell circumcenters
    max_level: int
    select_c: np.ndarray

    def __init__(
        self,
        icon_grid_filename: str,
        multi_mesh: Optional[ICONMultiMesh] = None,
        max_level: Optional[int] = None,
        iverbosity: int = 0,
    ):
        # open file, representing the finest level
        if iverbosity > 0:
            LOGGER.info(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:
            # read cell circumcenter coordinates
            clon = read_coordinate_array(ncfile, "clon", "cell")
            clat = read_coordinate_array(ncfile, "clat", "cell")

            reflvl_cell = ncfile.variables["refinement_level_c"][:]
            assert ncfile.variables["refinement_level_c"].dimensions == ("cell",)

            self.uuidOfHGrid = ncfile.uuidOfHGrid

        if max_level is not None:
            self.max_level = max_level
        else:
            self.max_level = reflvl_cell.max()

        # restrict to level `max_level`:
        self.select_c = np.argwhere(reflvl_cell <= self.max_level)
        # generate source grid node set:
        self.nodeset = NodeSet(clon[self.select_c], clat[self.select_c])

        if multi_mesh is not None:
            # generate edges between source grid nodes and multi-mesh nodes:
            edge_vertices = self._get_grid2mesh_edges(self.select_c, multi_mesh=multi_mesh)
            super().__init__((self.nodeset, multi_mesh.nodeset), edge_vertices)

    def _get_grid2mesh_edges(self, select_c: np.ndarray, multi_mesh: ICONMultiMesh) -> np.ndarray:
        """Create "grid-to-mesh" edges, ie. edges from (clat,clon) to the
        vertices of the multi-mesh.
        """

        num_cells = select_c.shape[0]
        num_verts_per_cell = multi_mesh.cell_vertices.shape[1]
        src_list = np.kron(np.arange(num_cells), np.ones((1, num_verts_per_cell), dtype=np.int64)).flatten()
        dst_list = multi_mesh.cell_vertices[select_c[:, 0], :].flatten()
        edge_vertices = np.stack((src_list, dst_list), axis=1, dtype=np.int64)
        return edge_vertices


# -------------------------------------------------------------


@typechecked
def get_icon_mesh_and_grid(
    iverbosity: int, grid_file: str, max_level_multimesh: int, max_level_dataset: int
) -> tuple[ICONMultiMesh, ICONCellDataGrid]:
    """Factory function, creating an ICON multi-mesh and an ICON cell-grid."""
    return (
        multi_mesh := ICONMultiMesh(grid_file, max_level=max_level_multimesh, iverbosity=iverbosity),
        ICONCellDataGrid(grid_file, multi_mesh, max_level=max_level_dataset, iverbosity=iverbosity),
    )


@typechecked
def get_edge_attributes(
    gc_send: np.ndarray, cc_send: np.ndarray, gc_recv: np.ndarray, cc_recv: np.ndarray
) -> np.ndarray:
    """Build edge features using the position on the unit sphere of the mesh
    nodes.

    The features are (4 input features in total):

    - length of the edge
    - vector difference between the 3d positions of the sender node and the
      receiver node computed in a local coordinate system of the receiver.

    The local coordinate system of the receiver is computed by applying two elemental
    rotations:

    - the first rotation changes the azimuthal angle until that receiver
      node lies at longitude 0,
    - the second rotation changes the polar angle until that receiver
      node lies at latitude 0.

    Literature: Appdx. A2.4 in
       Lam, R. et al. (2022). GraphCast: Learning skillful medium-range
       global weather forecasting. 10.48550/arXiv.2212.12794.

    """

    phi_theta = gc_recv  # precession and nutation angle
    theta__phi = np.fliplr(phi_theta)
    theta__phi[:, 1] *= -1.0  # angles = [theta, -phi]
    # rotation matrix:
    R = scipy.spatial.transform.Rotation.from_euler(seq="YZ", angles=theta__phi)
    edge_length = np.array([arc_length(cc_recv, cc_send)])
    distance = R.apply(cc_send) - [1.0, 0.0, 0.0]  # subtract the rotated position of sender
    return np.concatenate((edge_length.transpose(), distance), axis=1)


@typechecked
def arc_length(v1, v2) -> np.ndarray:
    """Calculate length of great circle arc on the unit sphere."""
    return np.arccos(np.clip(np.einsum("ij,ij->i", v1, v2), a_min=0.0, a_max=1.0))


@typechecked
def read_coordinate_array(ncfile, arrname: str, dimname: str) -> np.ndarray:
    """Auxiliary routine, reading a coordinate array, checking consistency."""
    arr = ncfile.variables[arrname][:]
    assert ncfile.variables[arrname].dimensions == (dimname,)
    assert ncfile.variables[arrname].units == "radian"
    # netCDF4 returns all variables as numpy.ma.core.MaskedArray.
    # -> convert to regular arrays
    assert not arr.mask.any(), f"There are missing values in {arrname}"
    return arr.data


@typechecked
def convert_list_to_adjacency_matrix(list_matrix: np.ndarray, ncols: int = 0) -> scipy.sparse.csr_matrix:
    """Convert an edge list into an adjacency matrix.

    Parameters
    ----------
    list_matrix : np.ndarray
        boolean matrix given by list of column indices for each row.
    ncols : int
        number of columns in result matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix [nrows, ncols]
    """
    nrows, ncols_per_row = list_matrix.shape
    indptr = np.arange(ncols_per_row * (nrows + 1), step=ncols_per_row)
    indices = list_matrix.ravel()
    return scipy.sparse.csr_matrix((np.ones(nrows * ncols_per_row), indices, indptr), dtype=bool, shape=(nrows, ncols))


@typechecked
def convert_adjacency_matrix_to_list(
    adj_matrix: scipy.sparse.csr_matrix,
    ncols_per_row: int,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Convert an adjacency matrix into an edge list.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        sparse (boolean) adjacency matrix
    ncols_per_row : int
        number of nonzero entries per row
    remove_duplicates : bool
        logical flag: remove duplicate rows.

    Returns
    -------
    np.ndarray
        boolean matrix given by list of column indices for each row.
    """
    if remove_duplicates:
        # The edges-vertex adjacency matrix may have duplicate rows, remove
        # them by selecting the rows that are unique:
        nrows = int(adj_matrix.nnz / ncols_per_row)
        mat = adj_matrix.indices.reshape((nrows, ncols_per_row))
        return np.unique(mat, axis=0)
    else:
        nrows = adj_matrix.shape[0]
        return adj_matrix.indices.reshape((nrows, ncols_per_row))


@typechecked
def selection_matrix(idx: np.ndarray, isize: int = 0) -> scipy.sparse.csr_matrix:
    """Create a diagonal selection matrix.

    Parameters
    ----------
    idx : np.ndarray
        integer array of indices
    isize : int
        size of (square) selection matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        diagonal matrix with ones at selected indices (idx,idx).
    """
    return scipy.sparse.csr_matrix((np.ones((len(idx))), (idx, idx)), dtype=bool, shape=(isize, isize))
