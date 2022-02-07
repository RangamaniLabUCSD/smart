"""
Wrapper around dolfin mesh class (originally for submesh implementation - possibly unneeded now)
"""
import dolfin as d
import numpy as np
from cached_property import cached_property
from stubs.common import _fancy_print as fancy_print
import stubs.common as common

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

class _Mesh:
    """
    General mesh class
    """
    def __init__(self, name='mesh_name', dimensionality=None):
        self.name               = name
        self.dimensionality     = dimensionality
        self.dolfin_mesh        = None

        self.mf                 = dict()
        self.parent_mesh        = None

        self.ds                 = None
        self.dx                 = None
        self.ds_uncombined      = None
        self.dx_uncombined      = None

    @property
    def mesh_view(self):
        return self.dolfin_mesh.topology().mapping()
    @property
    def id(self):
        return self.dolfin_mesh.id()
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
        return self.get_num_entities(self.dimensionality-1)
    @cached_property
    def num_vertices(self):
        return self.get_num_entities(0)
    
    # Entity mapping
    def _get_entities(self, dimension):
        num_vertices_per_entity = dimension+1 # for a simplex
        return np.reshape(self.dolfin_mesh.topology()(dimension, 0)(),
                         (self.get_num_entities(dimension), num_vertices_per_entity))
    @cached_property
    def cells(self):
        return self.dolfin_mesh.cells()
    @cached_property
    def facets(self):
        # By default dolfin only stores cells.
        # We must call dolfin_mesh.init() in order to access index maps for other dimensions
        return self._get_entities(self.dimensionality-1)
    @cached_property
    def subfacets(self):
        return self._get_entities(self.dimensionality-2)
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
        return d.assemble(1*self.dx)

    def get_nvolume(self, measure_type, marker=None):
        if measure_type=='dx':
            measure = self.dx
        elif measure_type == 'ds':
            measure = self.ds
        return(d.assemble(1*measure(marker)))

    def get_mesh_coordinate_bounds(self):
        return {'min': self.vertices.min(axis=0), 'max': self.vertices.max(axis=0)}

    # Integration measures
    def get_integration_measures(self):
        # Aliases
        mesh = self.dolfin_mesh
        # if self.is_volume:
            #self.ds = d.Measure('ds', domain=mesh, subdomain_data=self.mf['facets'])
            # if 'facets_uncombined' in self.mf:
            #     self.ds_uncombined = d.Measure('ds', domain=mesh, subdomain_data=self.mf['facets_uncombined'])

        # Regular measures
        if 'cells' in self.mf:
            self.dx = d.Measure('dx', domain=mesh, subdomain_data=self.mf['cells'])
        else:
            self.dx = d.Measure('dx', domain=mesh)

        # Uncombined marker measure
        if 'cells_uncombined' in self.mf:
            self.dx_uncombined = d.Measure('dx', domain=mesh, subdomain_data=self.mf['cells_uncombined'])
        
        # intersection  maps
        if isinstance(self, ChildMesh):
            for id_set, intersection_map in self.intersection_map.items():
                self.dx_map[id_set] = d.Measure('dx', domain=mesh, subdomain_data=intersection_map)
        
class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
    on marker values from the .xml file.
    """
    def __init__(self, mesh_filename, mesh_filetype='xml', name='parent_mesh'):
        super().__init__(self, name)
        if mesh_filetype == 'xml':
            self.load_mesh_from_xml(mesh_filename)
        elif mesh_filetype == 'hdf5':
            self.load_mesh_from_hdf5(mesh_filename)
        self.mesh_filename = mesh_filename
        self.mesh_filetype = mesh_filetype
        
        self.child_meshes = dict()
        self.parent_mesh = self
        # get mesh functions
        #self.mesh_functions = self.get_mesh_functions()
    
    @property
    def all_meshes(self):
        return dict(list(self.child_meshes.items()) + list({self.name: self}.items()))

    def load_mesh_from_xml(self, mesh_filename):
        self.dolfin_mesh = d.Mesh(mesh_filename)

        self.dolfin_mesh.init()
        self.dimensionality = self.dolfin_mesh.topology().dim()
        print(f"XML mesh, \"{self.name}\", successfully loaded from file: {mesh_filename}!")

    def load_mesh_from_hdf5(self, mesh_filename):
        #mesh, mfs = common.read_hdf5(hdf5_filename)
        self.dolfin_mesh = d.Mesh()
        hdf5 = d.HDF5File(self.dolfin_mesh.mpi_comm(), mesh_filename, 'r')
        hdf5.read(self.dolfin_mesh, '/mesh', False)

        self.dolfin_mesh.init()
        self.dimensionality = self.dolfin_mesh.topology().dim()
        print(f"HDF5 mesh, \"{self.name}\", successfully loaded from file: {mesh_filename}!")

    def _read_parent_mesh_function_from_file(self, dim):
        if self.mesh_filetype == 'xml':
            mf = d.MeshFunction('size_t', self.dolfin_mesh, dim, value=self.dolfin_mesh.domains())
        elif self.mesh_filetype == 'hdf5':
            hdf5 = d.HDF5File(self.dolfin_mesh.mpi_comm(), self.mesh_filename, 'r')
            mf = d.MeshFunction('size_t', self.dolfin_mesh, dim, value=0)
            hdf5.read(mf, f"/mf{dim}")
        return mf

    def read_parent_mesh_functions_from_file(self):
        # Aliases
        mesh        = self.dolfin_mesh
        volume_dim  = self.max_dim
        surface_dim = self.min_dim

        # Check validity
        if self.dolfin_mesh is None:
            print(f"Mesh {self.name} has no dolfin mesh to get a mesh function from.")
            return None
        assert len(self.child_meshes) > 0 # there should be at least one child mesh

        # Init mesh functions
        self.mf['cells'] = self._read_parent_mesh_function_from_file(volume_dim)
        if self.has_surface:
            self.mf['facets'] = self._read_parent_mesh_function_from_file(surface_dim)
            
        # If any cell markers are given as a list we also create mesh functions to store the uncombined markers
        if any([(child_mesh.marker_list is not None and not child_mesh.is_surface) for child_mesh in self.child_meshes.values()]):
            self.mf['cells_uncombined'] = self._read_parent_mesh_function_from_file(volume_dim)
        if any([(child_mesh.marker_list is not None and child_mesh.is_surface) for child_mesh in self.child_meshes.values()]):
            self.mf['facets_uncombined'] = self._read_parent_mesh_function_from_file(surface_dim)

        # Combine markers in a list 
        for child_mesh in self.child_meshes.values(): 
            if child_mesh.marker_list is None:
                continue
            for marker in child_mesh.marker_list:
                if not child_mesh.is_surface:
                    self.mf['cells'].array()[self.mf['cells_uncombined'].array() == marker] = child_mesh.primary_marker
                if child_mesh.is_surface:
                    self.mf['facets'].array()[self.mf['facets_uncombined'].array() == marker] = child_mesh.primary_marker

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
    def __init__(self, parent_mesh, compartment):#dimensionality, marker, name='child_mesh'):
        super().__init__(name=compartment.name, dimensionality=compartment.dimensionality)
        # Alias
        marker = compartment.cell_marker

        # child mesh must be associated with a parent mesh
        self.parent_mesh = None
        assert isinstance(parent_mesh, ParentMesh)
        self.set_parent_mesh(parent_mesh)

        # markers can be either an int or a list of ints
        if isinstance(marker, list):
            assert all([isinstance(m, int) for m in marker])
            self.marker_list = marker
            self.primary_marker = marker[0]
            fancy_print(f"List of markers given for compartment {self.name}, combining into single marker, {marker[0]}")
        else:
            assert isinstance(marker, int)
            self.marker_list = None
            self.primary_marker = marker

        # mapping (0 or 1) of intersection with sibling mesh
        self.dx_map = dict()
        self.intersection_map = dict()
        self.has_intersection = dict()

    def nvolume_sibling_union(self, sibling_mesh):
        return d.assemble(1*self.dx_map[{sibling_mesh.id}](1))
    
    @cached_property
    def map_cell_to_parent_entity(self):
        """
        We use the word 'entity' to be dimensionally-agnostic.
        If the child has topological dimension one lower than the parent, then the child's cell is the parent's facet.
        """
        return np.array(self.mesh_view[self.parent_mesh.id].cell_map())

    # too slow...
    # @cached_property
    # def map_facet_to_parent_entity(self):
    #     """
    #     We use parent indices as our "grounding point" for conversion
    #     Conversion is:
    #     self.map_facet_to_parent_vertex:         child facet -> child vertex -> parent vertex
    #     loop self.parent_mesh to invert its map: parent vertex -> parent_entity
    #     """

        # mapping = []
        # # Aliases
        # pm_entities = self.parent_mesh.get_entities(self.dimensionality-1)
        # list_of_sets = [set(pm_entities[entity_idx,:]) for entity_idx in range(pm_entities.shape[0])]

        # # parent_vertex to parent entity
        # for child_facet_idx in range(self.facets.shape[0]):
        #     subset = set(self.map_facet_to_parent_vertex[child_facet_idx,:])
        #     entity_idx = list_of_sets.index(subset)
        #     mapping.append(entity_idx)
        
        # return np.array(mapping)

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

    def find_surface_to_volume_mesh_intersection(self, sibling_volume_mesh):
        """
        Create a mesh function over this mesh's cells with value 0 if it does not coincide
        with a facet from sibling_volume_mesh, and value 1 if it does.

        Example: 
        * child_mesh_0 is a 2d submesh adjacent to child_mesh_1 with 6 cells (e.g. triangles).
        Cells [0,1,4] of child_mesh_0 are part of cells on child_mesh_1

        * child_mesh_1 (dolfin id = 38) is a 3d submesh with 6 cells. cells [0, 1, 4] have facets on 

        child_mesh_0.intersection_map[{38}] =  [1,1,0,0,1,0]
        """
        assert self.dimensionality == sibling_volume_mesh.dimensionality - 1
        # This is the value we set for non-intersecting cells
        not_intersecting_value = 18446744073709551615 # 2^64 - 1
        mesh_id_set = frozenset({sibling_volume_mesh.id})

        # map from our cells to sibling facets 
        # intersection_map is 1 where this mesh intersects with the boundary of its sibling
        self.intersection_map[mesh_id_set] = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality, value=0)
        bool_array = np.array(self.mesh_view[sibling_volume_mesh.id].cell_map()) # map from our (n-1)-cells to sibling's n-cells
        bool_array[bool_array!=not_intersecting_value] = 1
        bool_array[bool_array==not_intersecting_value] = 0
        self.intersection_map[mesh_id_set].set_values(bool_array.astype(np.int))

        # Check if the intersection is empty
        self.has_intersection[mesh_id_set] = self.intersection_map[mesh_id_set].array().any()
    
    def find_surface_to_2volumes_mesh_intersection(self, sibling_volume_mesh_1, sibling_volume_mesh_2):
        """
        Create a mesh function over this mesh's cells with value 0 if it does not coincide
        with a facet from both sibling_volume_mesh_1, and sibling_volume_mesh_2, and value 1 if it does.
        """
        # can combine this with function above
        assert self.dimensionality == sibling_volume_mesh_1.dimensionality - 1
        assert self.dimensionality == sibling_volume_mesh_2.dimensionality - 1
        # This is the value we set for non-intersecting cells
        not_intersecting_value = 18446744073709551615

        # find the intersection between this child mesh, sibling_volume_mesh_1, and sibling_volume_mesh_2
        mesh_id_set = frozenset({sibling_volume_mesh_1.id, sibling_volume_mesh_2.id})

        self.intersection_map[mesh_id_set] = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality, value=0)
        cell_map_1 = np.array(self.mesh_view[list(mesh_id_set)[0]].cell_map()) # map from our (n-1)-cells to sibling's n-cells
        # bool_array_1[bool_array_1!=not_intersecting_value] = 1
        # bool_array_1[bool_array_1==not_intersecting_value] = 0
        cell_map_2 = np.array(self.mesh_view[list(mesh_id_set)[1]].cell_map()) # repeat for sibling_mesh_2

        # Fill bool_array_total with 1 where cell_map_1 and cell_map_2 are not equal to not_intersectin_value
        bool_array_total = np.logical_and(cell_map_1!=not_intersecting_value, cell_map_2!=not_intersecting_value)

        self.intersection_map[mesh_id_set].set_values(bool_array_total.astype(np.int))

        # Check if the intersection is empty
        self.has_intersection[mesh_id_set] = self.intersection_map[mesh_id_set].array().any()

        

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
        mf_type = 'cells' if self.is_volume else 'facets'
        self.dolfin_mesh = d.MeshView.create(self.parent_mesh.mf[mf_type], self.primary_marker)
        self.dolfin_mesh.init()

    def init_marker_list_mesh_function(self):
        "Child mesh functions require transfering data from parent mesh functions"
        assert hasattr(self.parent_mesh, 'mf')
        assert self.marker_list is not None

        # initialize
        raise NotImplementedError("Need to check this")
        self.mf['cells']  = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality, value=0)

        # # Easiest case, just directly transfer from parent_mesh mesh function
        # # local cell index = local cell index -> parent cell index -> parent mesh function value
        # if self.is_volume:
        #     self.mf['cells'].array()[:]  = pmf['cells'].array()[self.map_cell_to_parent_entity]
        #     # Use the mapping from local facet to parent entity
        #     # self.mf['facets'] = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality-1, value=0)
        #     # self.mf['facets'].array()[:] = pmf['facets'].array()[self.map_facet_to_parent_entity]
        #     if 'cells_uncombined' in pmf:
        #         self.mf['cells_uncombined'].array()[:]  = pmf['cells_uncombined'].array()[self.map_cell_to_parent_entity]
        #     # if 'facets_uncombined' in pmf:
        #     #     self.mf['facets_uncombined'].array()[:]  = pmf['facets_uncombined'].array()[self.map_facet_to_parent_entity]
        # else:
        #     self.mf['cells'].array()[:]  = pmf['facets'].array()[self.map_cell_to_parent_entity]
        #     # if 'facets_uncombined' in pmf:
        #     #     self.mf['cells_uncombined'].array()[:]  = pmf['facets_uncombined'].array()[self.map_facet_to_parent_entity]

