"""
Wrapper around dolfin mesh class (originally for submesh implementation - possibly unneeded now)
"""
import dolfin as d
import numpy as np
from cached_property import cached_property
from .common import _fancy_print as fancy_print
from typing import Dict, Union
import logging
from mpi4py import MPI as _MPI
logger = logging.getLogger("smart")

__all__ = ["ParentMesh", "ChildMesh"]


def load_mesh_from_xml(mesh_filename: str) -> d.Mesh:
    """
    Read Dolfin mesh from xml file.

    .. note::
        Initializes facets and facet to cell connectivity

    Args:
        mesh_filename: Name of mesh file

    """
    mesh = d.Mesh(mesh_filename)
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1)
    mesh.init(tdim-1, tdim)
    logger.info(f'XML mesh, successfully loaded from file: {mesh_filename}!')
    return mesh


def load_mesh_from_hdf5(mesh_filename: str,
                        use_partition: bool = False,
                        comm: _MPI.Intracomm = d.MPI.comm_world) -> d.Mesh:
    """
    Read mesh from :func:`dolfin.HDF5File`.

    .. note::
        Initializes facets and facet to cell connectivity

    Args:
        mesh_fileanme: Path to mesh
        use_partition: If True use mesh partitioning from file
        comm: MPI-communicator to use for mesh

    """
    mesh = d.Mesh(comm)
    with d.HDF5File(mesh.mpi_comm(), mesh_filename, "r") as hdf5:
        hdf5.read(mesh, "/mesh", use_partition)
    comm.Barrier()

    d.MPI.comm_world.Barrier()

    tdim = mesh.topology().dim()
    mesh.init(tdim - 1)
    mesh.init(tdim-1, tdim)

    logger.info(f'HDF5 mesh successfully loaded from file: {mesh_filename}!')
    return mesh


class _Mesh:
    """
    General mesh class

    Args:
        name: Name of mesh

    """
    name: str
    _mesh: d.Mesh
    mf: Dict[str, d.MeshFunction]

    dx: d.Measure
    ds: d.Measure
    ds_uncombined: d.Measure
    dx_uncombined: d.Measure
    parent_mesh: "_Mesh"

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str, mesh: d.Mesh):
        self.name = name
        self._mesh = mesh
        self.mf = {}

    @property
    def dimensionality(self) -> int:
        return self._mesh.topology().dim()

    @property
    def mesh_view(self):
        return self._mesh.topology().mapping()

    @property
    def id(self):
        return int(self._mesh.id())

    @property
    def is_volume(self):
        return self.dimensionality == self.parent_mesh.dimensionality

    @property
    def is_surface(self):
        return self.dimensionality < self.parent_mesh.max_dim

    # Number of entities
    def get_num_entities(self, dimension):
        "Get the number of entities in this mesh with a certain topological dimension"
        return self._mesh.topology().size(dimension)

    @cached_property
    def num_cells(self):
        return self.get_num_entities(self.dimensionality)

    @cached_property
    def num_facets(self):
        return self.get_num_entities(self.dimensionality - 1)

    @cached_property
    def num_vertices(self):
        return self.get_num_entities(0)

    # Entity mapping
    def _get_entities(self, dimension):
        num_vertices_per_entity = dimension + 1  # for a simplex
        return np.reshape(
            self._mesh.topology()(dimension, 0)(),
            (self.get_num_entities(dimension), num_vertices_per_entity),
        )

    @cached_property
    def cells(self):
        return self._mesh.cells()

    @cached_property
    def facets(self):
        # By default dolfin only stores cells.
        # We must call dolfin.Mesh.init() in order to access index maps for other dimensions
        return self._get_entities(self.dimensionality - 1)

    @cached_property
    def subfacets(self):
        return self._get_entities(self.dimensionality - 2)

    @cached_property
    def vertices(self):
        return self.dolfin_mesh.coordinates()

    def get_entities(self, dimension):
        "We use this function so that values are cached and we don't need to recompute each time"
        if dimension == self.dimensionality:
            return self.cells
        elif dimension == self.dimensionality - 1:
            return self.facets
        elif dimension == self.dimensionality - 2:
            return self.subfacets
        elif dimension == 0:
            return self.vertices
        else:
            raise ValueError(f"Unknown entities for given dimension {dimension}")

    # Coordinates of entities
    @cached_property
    def cell_coordinates(self):
        return self.vertices[self.cells]

    @cached_property
    def facet_coordinates(self):
        return self.vertices[self.facets]

    # Generalized volume
    @cached_property
    def nvolume(self):
        return d.assemble(1 * self.dx)

    def get_nvolume(self, measure_type, marker=None):
        if measure_type == "dx":
            measure = self.dx
        elif measure_type == "ds":
            measure = self.ds
        return d.assemble(1 * measure(marker))

    def get_mesh_coordinate_bounds(self):
        return {"min": self.vertices.min(axis=0), "max": self.vertices.max(axis=0)}

    # Integration measures
    def get_integration_measures(self):
        # Aliases
        mesh = self._mesh

        # Regular measures
        if "cells" in self.mf:
            self.dx = d.Measure("dx", domain=mesh, subdomain_data=self.mf["cells"])
        else:
            self.dx = d.Measure("dx", domain=mesh)

        # Uncombined marker measure
        if "cells_uncombined" in self.mf:
            self.dx_uncombined = d.Measure(
                "dx", domain=mesh, subdomain_data=self.mf["cells_uncombined"]
            )


class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based
    on marker values from the .xml file.

    Args:
        mesh_filename (str): Name of mesh file
        mesh_filetype (str): Extension of mesh, either 'xml' or 'hdf5'
        name (str): Name of mesh
        use_partition (bool): If `hdf5` mesh file is loaded,
            choose if mesh should be read in with its current partition
    """
    mesh_filename: str
    mesh_filetype: str
    child_meshes: Dict[str, "ChildMesh"]
    parent_mesh: "ParentMesh"
    __slots__ = tuple(__annotations__)

    def __init__(self, mesh_filename: str, mesh_filetype, name: str, use_partition: bool = False):
        if mesh_filetype == "xml":
            mesh = load_mesh_from_xml(mesh_filename)
        elif mesh_filetype == "hdf5":
            mesh = load_mesh_from_hdf5(mesh_filename, use_partition)
        else:
            raise RuntimeError(f"Unknown {mesh_filetype=}")

        super().__init__(name, mesh)

        self.mesh_filename = mesh_filename
        self.mesh_filetype = mesh_filetype

        self.child_meshes = {}
        self.parent_mesh = self

    def get_mesh_from_id(self, id: int) -> Union["ChildMesh", "ParentMesh"]:
        # find the mesh in that has the matching id
        for mesh in self.all_meshes.values():
            if mesh.id == id:
                return mesh

    @property
    def all_meshes(self):
        return dict(list(self.child_meshes.items()) + list({self.name: self}.items()))

    def _read_parent_mesh_function_from_file(self, dim: int) -> d.MeshFunction:
        """
        Helper function to read :func:`dolfin.MeshFunction` from file for corresponding dimension
        """
        if self.mesh_filetype == "xml":
            mf = d.MeshFunction(
                "size_t", self._mesh, dim, value=self.dolfin_mesh.domains()
            )
        elif self.mesh_filetype == "hdf5":
            mf = d.MeshFunction("size_t", self._mesh, dim, value=0)
            with d.HDF5File(self._mesh.mpi_comm(), self.mesh_filename, "r") as hdf5:
                hdf5.read(mf, f"/mf{dim}")
                hdf5.close()
            self._mesh.mpi_comm().Barrier()
        return mf

    def read_parent_mesh_functions_from_file(self):
        # Aliases
        volume_dim = self.max_dim
        surface_dim = self.min_dim

        # Check validity
        if self._mesh is None:
            print(f"Mesh {self.name} has no dolfin mesh to get a mesh function from.")
            return None
        # there should be at least one child mesh
        assert len(self.child_meshes) > 0

        # Init mesh functions
        self.mf["cells"] = self._read_parent_mesh_function_from_file(volume_dim)
        if self.has_surface:
            self.mf["facets"] = self._read_parent_mesh_function_from_file(surface_dim)

        # If any cell markers are given as a list we also create mesh
        # functions to store the uncombined markers
        if any(
            [
                (child_mesh.marker_list is not None and not child_mesh.is_surface)
                for child_mesh in self.child_meshes.values()
            ]
        ):
            self.mf["cells_uncombined"] = self._read_parent_mesh_function_from_file(
                volume_dim
            )
        if any(
            [
                (child_mesh.marker_list is not None and child_mesh.is_surface)
                for child_mesh in self.child_meshes.values()
            ]
        ):
            self.mf["facets_uncombined"] = self._read_parent_mesh_function_from_file(
                surface_dim
            )

        # Combine markers in a list
        for child_mesh in self.child_meshes.values():
            if child_mesh.marker_list is None:
                continue
            for marker in child_mesh.marker_list:
                if not child_mesh.is_surface:
                    self.mf["cells"].array()[
                        self.mf["cells_uncombined"].array() == marker
                    ] = child_mesh.primary_marker
                if child_mesh.is_surface:
                    self.mf["facets"].array()[
                        self.mf["facets_uncombined"].array() == marker
                    ] = child_mesh.primary_marker

    @property
    def has_surface(self):
        return self.min_dim < self.max_dim

    @property
    def child_surface_meshes(self):
        return [cm for cm in self.child_meshes.values() if cm.is_surface]

    @property
    def child_volume_meshes(self):
        return [cm for cm in self.child_meshes.values() if cm.is_volume]


class ChildMesh(_Mesh):
    """
    Sub mesh of a parent mesh
    """

    # dimensionality, marker, name='child_mesh'):
    def __init__(self, parent_mesh, compartment):
        super().__init__(
            name=compartment.name, dimensionality=compartment.dimensionality
        )
        # Alias
        marker = compartment.cell_marker

        self.compartment = compartment

        # child mesh must be associated with a parent mesh
        self.parent_mesh = None
        assert isinstance(parent_mesh, ParentMesh)
        self.set_parent_mesh(parent_mesh)

        # markers can be either an int or a list of ints
        if isinstance(marker, list):
            assert all([isinstance(m, int) for m in marker])
            self.marker_list = marker
            self.primary_marker = marker[0]
            fancy_print(
                f"List of markers given for compartment {self.name},"
                + f"combining into single marker, {marker[0]}"
            )
        else:
            assert isinstance(marker, int)
            self.marker_list = None
            self.primary_marker = marker

        # mapping (0 or 1) of intersection with sibling mesh
        self.intersection_map = dict()
        # indices of parent mesh that correspond to the intersection
        self.intersection_map_parent = dict()
        self.intersection_submesh = dict()
        self.intersection_dx = dict()
        self.has_intersection = dict()

    def nvolume_sibling_union(self, sibling_mesh):
        # return d.assemble(1*self.intersection_dx[frozenset({sibling_mesh.id}](1))
        return d.assemble(1 * self.intersection_dx[frozenset({sibling_mesh.id})])

    @cached_property
    def map_cell_to_parent_entity(self):
        """
        We use the word 'entity' to be dimensionally-agnostic.
        If the child has topological dimension one lower than the parent,
        then the child's cell is the parent's facet.
        """
        return np.array(self.mesh_view[self.parent_mesh.id].cell_map())

    # combination maps
    @cached_property
    def map_cell_to_parent_vertex(self):
        return self.map_vertex_to_parent_vertex[self.cells]

    @cached_property
    def map_facet_to_parent_vertex(self):
        return self.map_vertex_to_parent_vertex[self.facets]

    @cached_property
    def map_vertex_to_parent_vertex(self):
        return np.array(self.mesh_view[self.parent_mesh.id].vertex_map())

    def find_surface_to_volumes_mesh_intersection(self, sibling_volume_mesh_list):
        """
        Create a mesh function over this mesh's cells with value 0 if it does not coincide
        with facets from sibling_volume_mesh_list, and value 1 if it does.

        Example:
        * child_mesh_0 is a 2d submesh adjacent to child_mesh_1 with 6 cells (e.g. triangles).
        Cells [0,1,4] of child_mesh_0 are part of cells on child_mesh_1

        * child_mesh_1 (dolfin id = 38) is a 3d submesh with 6 cells. cells [0, 1, 4] have facets on

        child_mesh_0.intersection_map[{38}] =  [1,1,0,0,1,0]
        """
        assert all(
            [
                self.dimensionality == sibling_volume_mesh.dimensionality - 1
                for sibling_volume_mesh in sibling_volume_mesh_list
            ]
        )
        assert len(sibling_volume_mesh_list) in [1, 2]
        # This is the value we set in dolfin/Mesh.cpp for non-intersecting cells
        not_intersecting_value = 18446744073709551615

        # find the intersection between this child mesh and the sibling volume meshes
        mesh_id_set = frozenset(
            [sibling_volume_mesh.id for sibling_volume_mesh in sibling_volume_mesh_list]
        )

        self.intersection_map[mesh_id_set] = d.MeshFunction(
            "size_t", self._mesh, self.dimensionality, value=0
        )
        cell_maps = [
            np.array(self.mesh_view[sibling_volume_mesh.id].cell_map())
            for sibling_volume_mesh in sibling_volume_mesh_list
        ]

        # Fill intersection_map_values with 1 where cell_maps are not equal to
        # not_intersecting_value
        if len(sibling_volume_mesh_list) == 1:
            intersection_map_values = cell_maps[0] != not_intersecting_value
        else:
            intersection_map_values = np.logical_and(
                *[cell_map != not_intersecting_value for cell_map in cell_maps]
            )
        # Set the values of intersection_map
        self.intersection_map[mesh_id_set].set_values(
            intersection_map_values.astype(np.int)
        )

        # Check if the intersection is empty
        self.has_intersection[mesh_id_set] = (
            self.intersection_map[mesh_id_set].array().any()
        )

        # Indicate which entities of the parent mesh correspond to this intersection
        self.intersection_map_parent[mesh_id_set] = d.MeshFunction(
            "size_t", self.parent_mesh._mesh, self.dimensionality, value=0
        )
        # map from our cells to parent facets
        mesh_to_parent = np.array(self.mesh_view[self.parent_mesh.id].cell_map())
        indices = np.where(self.intersection_map[mesh_id_set].array() == 1)[
            0
        ]  # indices of our cells that intersect
        # indices of parent facets that intersect
        parent_indices = mesh_to_parent[indices]
        self.intersection_map_parent[mesh_id_set].array()[parent_indices] = 1

    def get_intersection_submesh(self, mesh_id_set):
        # mesh_id_set = frozenset(
        #     [sibling_volume_mesh.id for sibling_volume_mesh in sibling_volume_mesh_list])

        self.intersection_submesh[mesh_id_set] = d.MeshView.create(
            self.intersection_map_parent[mesh_id_set], 1
        )
        self.intersection_submesh[mesh_id_set].init()
        self.intersection_dx[mesh_id_set] = d.Measure(
            "dx", self.intersection_submesh[mesh_id_set]
        )

    def set_parent_mesh(self, parent_mesh):
        # remove existing parent mesh if not None
        if self.parent_mesh:
            self.parent_mesh.child_meshes.pop(self)
        # set the parent mesh
        self.parent_mesh = parent_mesh
        # add self to parent mesh's list of children meshes
        if self.parent_mesh:
            self.parent_mesh.child_meshes.update({self.name: self})

    def extract_submesh(self):
        mf_type = "cells" if self.is_volume else "facets"
        self._mesh = d.MeshView.create(
            self.parent_mesh.mf[mf_type], self.primary_marker
        )
        # self._mesh.init()

    def init_marker_list_mesh_function(self):
        "Child mesh functions require transfering data from parent mesh functions"
        assert hasattr(self.parent_mesh, "mf")
        assert self.marker_list is not None

        # initialize
        raise NotImplementedError("Need to check this")
        self.mf["cells"] = d.MeshFunction(
            "size_t", self._mesh, self.dimensionality, value=0
        )
