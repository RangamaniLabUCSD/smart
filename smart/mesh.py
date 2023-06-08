"""
Wrapper around dolfin mesh class to define parent and child meshes for SMART simulations.
"""
from typing import Dict, FrozenSet
import logging

import dolfin as d
import numpy as np
from cached_property import cached_property


logger = logging.getLogger(__name__)


class _Mesh:
    """
    General mesh class
    """

    name: str
    dimensionality: int
    dolfin_mesh: d.Mesh
    mf: Dict[str, d.MeshFunction]
    parent_mesh: d.Mesh
    ds: d.Measure
    dx: d.Measure
    dx_uncombined: d.Measure

    def __init__(self, name="mesh_name", dimensionality=None):
        self.name = name
        self.dimensionality = dimensionality
        self.dolfin_mesh = None

        self.mf = dict()
        self.parent_mesh = None

        self.ds = None
        self.dx = None
        self.dx_uncombined = None

    @property
    def mesh_view(self):
        return self.dolfin_mesh.topology().mapping()

    @property
    def id(self):
        return int(self.dolfin_mesh.id())

    @property
    def is_volume(self):
        return self.dimensionality == self.parent_mesh.dimensionality

    @property
    def is_surface(self):
        return self.dimensionality < self.parent_mesh.max_dim

    # Number of entities
    def get_num_entities(self, dimension):
        "Get the number of entities in this mesh with a certain topological dimension"
        return self.dolfin_mesh.topology().size(dimension)

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
            self.dolfin_mesh.topology()(dimension, 0)(),
            (self.get_num_entities(dimension), num_vertices_per_entity),
        )

    @cached_property
    def cells(self):
        return self.dolfin_mesh.cells()

    @cached_property
    def facets(self):
        # By default dolfin only stores cells.
        # We must call dolfin_mesh.init() in order to access index maps for other dimensions
        return self._get_entities(self.dimensionality - 1)

    @cached_property
    def subfacets(self):
        return self._get_entities(self.dimensionality - 2)

    @cached_property
    def vertices(self):
        return self.dolfin_mesh.coordinates()

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

    # Integration measures
    def get_integration_measures(self):
        # Aliases
        mesh = self.dolfin_mesh

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
    on marker values from the .hdf5 or .xml file.

    Args:
    * mesh_filename (str): Name of mesh file
    * mesh_filetype (str): Extension of mesh, either 'xml' or 'hdf5'
    * parent_mesh (str): Name of mesh
    * use_partition (bool): If `hdf5` mesh file is loaded,
            choose if mesh should be read in with its current partition
    """

    mesh_filename: str
    mesh_filetype: str
    child_meshes: Dict[str, "ChildMesh"]
    parent_mesh: "ParentMesh"
    use_partition: bool

    def __init__(
        self,
        mesh_filename: str,
        mesh_filetype,
        name,
        use_partition=False,
        mpi_comm=d.MPI.comm_world,
    ):
        super().__init__(name)
        self.use_partition = use_partition
        self.mpi_comm = mpi_comm
        if mesh_filetype == "xml":
            self.load_mesh_from_xml(mesh_filename)
        elif mesh_filetype == "hdf5":
            self.load_mesh_from_hdf5(mesh_filename, use_partition)
        self.mesh_filename = mesh_filename
        self.mesh_filetype = mesh_filetype

        self.child_meshes = dict()
        self.parent_mesh = self

    def get_mesh_from_id(self, id):
        "Find the mesh that has the matching id."
        # find the mesh in that has the matching id
        for mesh in self.all_meshes.values():
            if mesh.id == id:
                return mesh

    @property
    def all_meshes(self):
        return dict(list(self.child_meshes.items()) + list({self.name: self}.items()))

    def load_mesh_from_xml(self, mesh_filename):
        "Load parent mesh from xml file `mesh_filename`"
        self.dolfin_mesh = d.Mesh(mesh_filename)

        self.dimensionality = self.dolfin_mesh.topology().dim()
        self.dolfin_mesh.init(self.dimensionality - 1)
        self.dolfin_mesh.init(self.dimensionality - 1, self.dimensionality)
        self.dolfin_mesh.init(self.dimensionality - 1, self.dimensionality)

        logger.info(f'XML mesh, "{self.name}", successfully loaded from file: {mesh_filename}!')

    def load_mesh_from_hdf5(self, mesh_filename, use_partition=False):
        """
        Load parent mesh from hdf5 file `mesh_filename`
        Parallelize mesh if `use_partition` is True
        """
        comm = self.mpi_comm
        # mesh, mfs = common.read_hdf5(hdf5_filename)
        self.dolfin_mesh = d.Mesh(comm)
        hdf5 = d.HDF5File(self.dolfin_mesh.mpi_comm(), mesh_filename, "r")
        hdf5.read(self.dolfin_mesh, "/mesh", use_partition)

        if comm.size > 1:
            d.MPI.comm_world.Barrier()
        hdf5.close()

        self.dimensionality = self.dolfin_mesh.topology().dim()
        self.dolfin_mesh.init(self.dimensionality - 1)
        self.dolfin_mesh.init(self.dimensionality - 1, self.dimensionality)

        logger.info(f'HDF5 mesh, "{self.name}", successfully loaded from file: {mesh_filename}!')

    def _read_parent_mesh_function_from_file(self, dim):
        if self.mesh_filetype == "xml":
            mf = d.MeshFunction("size_t", self.dolfin_mesh, dim, value=self.dolfin_mesh.domains())
        elif self.mesh_filetype == "hdf5":
            mf = d.MeshFunction("size_t", self.dolfin_mesh, dim, value=0)
            # with d.HDF5File(self.dolfin_mesh.mpi_comm(), self.mesh_filename, 'r') as hdf5:
            # hdf5.read(mf, f'/mesh/{dim}')
            hdf5 = d.HDF5File(self.dolfin_mesh.mpi_comm(), self.mesh_filename, "r")
            hdf5.read(mf, f"/mf{dim}")
            if self.dolfin_mesh.mpi_comm().size > 1:
                d.MPI.comm_world.Barrier()
            hdf5.close()
        return mf

    def read_parent_mesh_functions_from_file(self):
        """
        Read mesh function for parent mesh into :attr:`ParentMesh.mf`.
        """
        # Aliases
        volume_dim = self.max_dim
        surface_dim = self.min_dim

        # Check validity
        if self.dolfin_mesh is None:
            logger.info(f"Mesh {self.name} has no dolfin mesh to get a mesh function from.")
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
            self.mf["cells_uncombined"] = self._read_parent_mesh_function_from_file(volume_dim)
        if any(
            [
                (child_mesh.marker_list is not None and child_mesh.is_surface)
                for child_mesh in self.child_meshes.values()
            ]
        ):
            self.mf["facets_uncombined"] = self._read_parent_mesh_function_from_file(surface_dim)

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
    Sub-mesh of a parent mesh, defined by mesh markers.

    Params:
    * parent_mesh: The mesh owning the entities in the compartment
    * compartment: The compartment object
    """

    intersection_map: Dict[FrozenSet[int], d.MeshFunction]
    intersection_map_parent: Dict[FrozenSet[int], d.MeshFunction]
    intersection_submesh: Dict[FrozenSet[int], d.Mesh]
    intersection_dx: Dict[FrozenSet[int], d.Measure]
    has_intersection: Dict[FrozenSet[int], bool]
    # dimensionality, marker, name='child_mesh'):

    def __init__(self, parent_mesh, compartment):
        super().__init__(name=compartment.name, dimensionality=compartment.dimensionality)
        self.compartment = compartment

        # child mesh must be associated with a parent mesh
        self.parent_mesh = None
        assert isinstance(parent_mesh, ParentMesh)
        self.set_parent_mesh(parent_mesh)

        # markers can be either an int or a list of ints
        if isinstance(compartment.cell_marker, list):
            assert all([isinstance(m, int) for m in compartment.cell_marker])
            self.marker_list = compartment.cell_marker
            self.primary_marker = compartment.cell_marker[0]
            logger.info(
                f"List of markers given for compartment {self.name},"
                + f"combining into single marker, {compartment.cell_marker[0]}"
            )
        else:
            assert isinstance(compartment.cell_marker, int)
            self.marker_list = None
            self.primary_marker = compartment.cell_marker

        # mapping (0 or 1) of intersection with sibling mesh
        self.intersection_map = dict()
        # indices of parent mesh that correspond to the intersection
        self.intersection_map_parent = dict()
        self.intersection_submesh = dict()
        self.intersection_dx = dict()
        self.has_intersection = dict()

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
            "size_t", self.dolfin_mesh, self.dimensionality, value=0
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
        self.intersection_map[mesh_id_set].set_values(intersection_map_values.astype(np.int))

        # Check if the intersection is empty
        self.has_intersection[mesh_id_set] = self.intersection_map[mesh_id_set].array().any()

        # Indicate which entities of the parent mesh correspond to this intersection
        self.intersection_map_parent[mesh_id_set] = d.MeshFunction(
            "size_t", self.parent_mesh.dolfin_mesh, self.dimensionality, value=0
        )
        # map from our cells to parent facets
        mesh_to_parent = np.array(self.mesh_view[self.parent_mesh.id].cell_map())
        indices = np.where(self.intersection_map[mesh_id_set].array() == 1)[
            0
        ]  # indices of our cells that intersect
        # indices of parent facets that intersect
        parent_indices = mesh_to_parent[indices]
        self.intersection_map_parent[mesh_id_set].array()[parent_indices] = 1

    def get_intersection_submesh(self, mesh_id_set: FrozenSet[int]):
        self.intersection_submesh[mesh_id_set] = d.MeshView.create(
            self.intersection_map_parent[mesh_id_set], 1
        )
        self.intersection_submesh[mesh_id_set].init()
        self.intersection_dx[mesh_id_set] = d.Measure("dx", self.intersection_submesh[mesh_id_set])

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
        self.dolfin_mesh = d.MeshView.create(self.parent_mesh.mf[mf_type], self.primary_marker)
