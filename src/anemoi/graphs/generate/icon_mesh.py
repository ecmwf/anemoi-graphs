# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import uuid
from functools import cached_property
from typing import Optional

import netCDF4
import numpy as np
import scipy
from typeguard import typechecked
from typing_extensions import Self

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.generate.utils import convert_adjacency_matrix_to_list
from anemoi.graphs.generate.utils import convert_list_to_adjacency_matrix
from anemoi.graphs.generate.utils import selection_matrix

LOGGER = logging.getLogger(__name__)


@typechecked
class NodeSet:
    """Stores nodes on the unit sphere.

    Attributes
    ----------
    id_iter : int
        Unique ID for each object.
    gc_vertices : np.ndarray
        Geographical (lat/lon) coordinates, in radians. Shape: (num_nodes, 2)
    ref_level : np.ndarray
        Reference level of each node. Shape: (num_nodes, )
    """

    id_iter: int = itertools.count()
    gc_vertices: np.ndarray
    ref_level: np.ndarray

    def __init__(self, lats: np.ndarray, lons: np.ndarray, ref_level: np.ndarray):
        self.latitudes = lats
        self.longitudes = lons
        self.ref_level = ref_level
        self.id = uuid.uuid4()

    @property
    def num_nodes(self) -> int:
        return self.gc_vertices.shape[0]

    @cached_property
    def gc_vertices(self):
        return np.column_stack((self.latitudes, self.longitudes))

    @cached_property
    def cc_vertices(self):
        """Cartesian coordinates [rad], shape [:,3]."""
        return latlon_rad_to_cartesian(self.gc_vertices.T)

    def __add__(self, other: Self) -> Self:
        """concatenates two node sets."""
        latitudes = np.concatenate([self.latitudes, other.latitudes])
        longitudes = np.concatenate([self.longitudes, other.longitudes])
        ref_levels = np.concatenate([self.ref_level, other.ref_level])
        return NodeSet(latitudes, longitudes, ref_levels)

    def __eq__(self, other: Self) -> bool:
        """Compares two node sets."""
        return self.id == other.id

    def get_mask_level(self, level: int) -> np.ndarray:
        return self.ref_level <= level

    def restrict_to_level(self, level: int) -> Self:
        mask = self.get_mask_level(level)
        return NodeSet(self.latitudes[mask], self.longitudes[mask], self.ref_level[mask])


@typechecked
class ICONMultiMesh:
    """Reads vertices and topology from an ICON grid file; creates multi-mesh."""

    uuidOfHGrid: str
    max_level: int
    nodeset: NodeSet  # set of ICON grid vertices
    cell_vertices: np.ndarray

    def __init__(self, icon_grid_filename: str, max_level: Optional[int] = None):
        # open file, representing the finest level
        LOGGER.debug(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:
            # read vertex coordinates
            vlon = read_coordinate_array(ncfile, "vlon", "vertex")
            vlat = read_coordinate_array(ncfile, "vlat", "vertex")
            vreflevel = get_ncfile_variable(ncfile, "refinement_level_v", expected_dimensions=("vertex",))
            self.nodeset = NodeSet(vlon, vlat, vreflevel)

            edge_vertices_fine = np.asarray(
                get_ncfile_variable(ncfile, "edge_vertices", ("nc", "edge")) - 1, dtype=np.int64
            ).transpose()
            cell_vertices_fine = np.asarray(
                get_ncfile_variable(ncfile, "vertex_of_cell", ("nv", "cell")) - 1, dtype=np.int64
            ).transpose()

            self.uuidOfHGrid = ncfile.uuidOfHGrid

        self.max_level = max_level if max_level is not None else vreflevel.max()

        # generate edge-vertex relations for coarser levels:
        (edge_vertices, cell_vertices) = self._get_hierarchy_of_icon_edge_graphs(
            edge_vertices_fine=edge_vertices_fine,
            cell_vertices_fine=cell_vertices_fine,
            reflvl_vertex=vreflevel,
        )

        if self.max_level <= vreflevel.max():  # restric multi_mesh to "max_level"
            self.edge_vertices, self.cell_vertices = self._restrict_multi_mesh_level(edge_vertices, cell_vertices)
            self.nodeset = self.nodeset.restrict_to_level(self.max_level)

    def node_coordinates(self):
        return self.nodeset.gc_vertices.astype(np.float32)

    def _restrict_multi_mesh_level(
        self,
        edge_vertices: list[np.ndarray],
        cell_vertices: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Creates a new mesh with only the vertices at the desired level.

        Parameters
        ----------
        edge_vertices : list[np.ndarray]
            Edge vertices
        cell_vertices : np.ndarray
            Cell vertices

        Returns
        -------
        edge_vertices : list[np.ndarray]
            Updated edge vertices
        cell_vertices : np.ndarray
            Updated cell vertices
        """
        vertex_mask = self.nodeset.get_mask_level(self.max_level)

        # Mapping vertices index of the edges to the new subset of vertices
        vertex_glb2loc = np.zeros(self.nodeset.num_nodes, dtype=int)
        vertex_glb2loc[vertex_mask] = np.arange(vertex_mask.sum())

        restricted_edge_vertices = [vertex_glb2loc[vertices] for vertices in edge_vertices[: self.max_level + 1]]
        restricted_cell_vertices = np.where(cell_vertices >= 0, vertex_glb2loc[cell_vertices], cell_vertices)
        # cell_vertices: preserve negative indices (incomplete cells)
        return restricted_edge_vertices, restricted_cell_vertices

    def _get_hierarchy_of_icon_edge_graphs(
        self,
        edge_vertices_fine: np.ndarray,
        cell_vertices_fine: np.ndarray,
        reflvl_vertex: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Returns a list of edge-vertex relations (coarsest to finest level)."""

        edge_vertices = [edge_vertices_fine]  # list of edge-vertex relations (coarsest to finest level).

        num_vertices = reflvl_vertex.shape[0]
        # edge-to-vertex adjacency matrix with 2 non-zero entries per row:
        edge2vertex_matrix = convert_list_to_adjacency_matrix(edge_vertices_fine, num_vertices)
        # cell-to-vertex adjacency matrix with 3 non-zero entries per row:
        cell2vertex_matrix = convert_list_to_adjacency_matrix(cell_vertices_fine, num_vertices)
        vertex2vertex_matrix = edge2vertex_matrix.transpose() * edge2vertex_matrix
        vertex2vertex_matrix.setdiag(np.ones(num_vertices))  # vertices are self-connected

        selected_vertex_coarse = scipy.sparse.diags(np.ones(num_vertices), dtype=bool)

        # coarsen edge-vertex list from level `ilevel -> ilevel - 1`:
        for ilevel in reversed(range(1, reflvl_vertex.max() + 1)):
            LOGGER.debug(f"  edges[{ilevel}] = {edge_vertices[0].shape[0] : >9}")

            # define edge selection matrix (selecting only edges of which have
            # exactly one coarse vertex):
            #
            # get a boolean mask, matching all edges where one of its vertices
            # has refinement level index `ilevel`:
            ref_level_mask = reflvl_vertex[edge_vertices[0]] == ilevel
            edges_coarse = np.logical_xor(ref_level_mask[:, 0], ref_level_mask[:, 1])  # = bisected coarse edges
            idx_edge2edge = np.argwhere(edges_coarse).flatten()
            selected_edges = selection_matrix(idx_edge2edge, edges_coarse.shape[0])

            # define vertex selection matrix selecting only vertices of
            # level `ilevel`:
            idx_v_fine = np.argwhere(reflvl_vertex == ilevel).flatten()
            selected_vertex_fine = selection_matrix(idx_v_fine, num_vertices)
            # define vertex selection matrix, selecting only vertices of
            # level < `ilevel`, by successively removing `s_fine` from an identity matrix.
            selected_vertex_coarse.data[0][idx_v_fine] = False

            # create an adjacency matrix which links each fine level
            # vertex to its two coarser neighbor vertices:
            vertex2vertex_fine2coarse = selected_vertex_fine * vertex2vertex_matrix * selected_vertex_coarse
            # remove rows that have only one non-zero entry
            # (corresponding to incomplete parent triangles in LAM grids):
            csum = vertex2vertex_fine2coarse * np.ones((vertex2vertex_fine2coarse.shape[1], 1))
            selected_vertex2vertex = selection_matrix(
                np.argwhere(csum == 2).flatten(), vertex2vertex_fine2coarse.shape[0]
            )
            vertex2vertex_fine2coarse = selected_vertex2vertex * vertex2vertex_fine2coarse

            # then construct the edges-to-parent-vertex adjacency matrix:
            parent_edge_vertices = selected_edges * edge2vertex_matrix * vertex2vertex_fine2coarse
            # again, we have might have selected edges within
            # `selected_edges` which are part of an incomplete parent edge
            # (LAM case). We filter these here:
            csum = parent_edge_vertices * np.ones((parent_edge_vertices.shape[1], 1))
            selected_edge2edge = selection_matrix(np.argwhere(csum == 2).flatten(), parent_edge_vertices.shape[0])
            parent_edge_vertices = selected_edge2edge * parent_edge_vertices

            # note: the edges-vertex adjacency matrix still has duplicate
            # rows, since two child edges have the same parent edge.
            edge_vertices_coarse = convert_adjacency_matrix_to_list(parent_edge_vertices, ncols_per_row=2)
            edge_vertices.insert(0, edge_vertices_coarse)

            # store cell-to-vert adjacency matrix
            if ilevel > self.max_level:
                cell2vertex_matrix = cell2vertex_matrix * vertex2vertex_fine2coarse
                # similar to the treatment above, we need to handle
                # coarse LAM cells which are incomplete.
                csum = cell2vertex_matrix * np.ones((cell2vertex_matrix.shape[1], 1))
                selected_cell2cell = selection_matrix(np.argwhere(csum == 3).flatten(), cell2vertex_matrix.shape[0])
                cell2vertex_matrix = selected_cell2cell * cell2vertex_matrix

            # replace edge-to-vertex and vert-to-vert adjacency matrices (for next level):
            if ilevel > 1:
                vertex2vertex_matrix = selected_vertex_coarse * vertex2vertex_matrix * vertex2vertex_fine2coarse
                edge2vertex_matrix = convert_list_to_adjacency_matrix(edge_vertices_coarse, num_vertices)

        # Fine-level cells outside of multi-mesh (LAM boundary)
        # correspond to empty rows in the adjacency matrix. We
        # substitute these by three additional, non-existent vertices:
        csum = 3 - cell2vertex_matrix * np.ones((cell2vertex_matrix.shape[1], 1))
        nvmax = cell2vertex_matrix.shape[1]
        cell2vertex_matrix = scipy.sparse.csr_matrix(scipy.sparse.hstack((cell2vertex_matrix, csum, csum, csum)))

        # build a list of cell-vertices [1..num_cells,1..3] for all
        # fine-level cells:
        cell_vertices = convert_adjacency_matrix_to_list(cell2vertex_matrix, remove_duplicates=False, ncols_per_row=3)

        # finally, translate non-existent vertices into "-1" indices:
        cell_vertices = np.where(
            cell_vertices >= nvmax,
            -np.ones(cell_vertices.shape, dtype=np.int32),
            cell_vertices,
        )

        return (edge_vertices, cell_vertices)


@typechecked
class ICONCellDataGrid:
    """Reads cell locations from an ICON grid file; builds grid-to-mesh edges based on ICON topology."""

    uuidOfHGrid: str
    nodeset: NodeSet  # set of ICON cell circumcenters
    max_level: int
    select_c: np.ndarray

    def __init__(
        self,
        icon_grid_filename: str,
        max_level: Optional[int] = None,
    ):
        LOGGER.debug(f"{type(self).__name__}: read ICON grid file '{icon_grid_filename}'")
        with netCDF4.Dataset(icon_grid_filename, "r") as ncfile:  # open file, representing the finest level
            # read cell circumcenter coordinates
            clon = read_coordinate_array(ncfile, "clon", "cell")
            clat = read_coordinate_array(ncfile, "clat", "cell")
            creflevel = get_ncfile_variable(ncfile, "refinement_level_c", expected_dimensions=("cell",))
            self.nodeset = NodeSet(clon, clat, creflevel)
            self.uuidOfHGrid = ncfile.uuidOfHGrid

        self.max_level = max_level if max_level is not None else creflevel.max()
        self.select_c = np.argwhere(creflevel <= self.max_level)  # restrict to level `max_level`:
        self.nodeset = NodeSet(clon[self.select_c], clat[self.select_c], creflevel)  # source nodes

    def node_coordinates(self):
        return self.nodeset.gc_vertices.astype(np.float32)


def get_multimesh_edges(multi_mesh: ICONMultiMesh, resolutions: list[int]) -> np.ndarray:
    return np.concatenate([multi_mesh.edge_vertices[res] for res in resolutions], axis=0)


def get_grid2mesh_edges(cell_grid: ICONCellDataGrid, multi_mesh: ICONMultiMesh) -> np.ndarray:
    """Create "grid-to-mesh" edges.

    It creates the edges from (clat,clon) to the vertices of the multi-mesh.

    Parameters
    ----------
    cell_grid : ICONCellDataGrid
        Cell grid
    multi_mesh : ICONMultimesh
        Multi mesh

    Returns
    -------
    np.ndarray
        Bipartite graph of the "grid-to-mesh" connections.
    """
    num_cells = cell_grid.select_c.shape[0]
    num_vertices_per_cell = multi_mesh.cell_vertices.shape[1]
    src_list = np.kron(np.arange(num_cells), np.ones((1, num_vertices_per_cell), dtype=np.int64)).flatten()
    dst_list = multi_mesh.cell_vertices[cell_grid.select_c[:, 0], :].flatten()
    return np.stack((src_list, dst_list), axis=1, dtype=np.int64)


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
def get_ncfile_variable(ncfile: netCDF4.Dataset, variable: str, expected_dimensions: tuple[str, ...]) -> np.ndarray:
    var_dims = ncfile.variables[variable].dimensions
    assert var_dims == expected_dimensions, f"Variable {variable} has dimesnions {var_dims} != {expected_dimensions}."
    return ncfile.variables[variable][:]
