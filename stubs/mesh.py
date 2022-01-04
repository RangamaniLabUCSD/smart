"""
Wrapper around dolfin mesh class (originally for submesh implementation - possibly unneeded now)
"""
import dolfin as d
import numpy as np
from cached_property import cached_property
from stubs.common import _fancy_print as fancy_print

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
    def is_volume_mesh(self):
        return self.dimensionality == self.parent_mesh.dimensionality

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
        mf = self.mf

        self.ds = d.Measure('ds', domain=mesh, subdomain_data=mf['facets'])
        self.dx = d.Measure('dx', domain=mesh, subdomain_data=mf['cells'])
        # Markers on the boundary mesh of this child mesh
        if 'facets_uncombined' in mf:
            self.ds_uncombined = d.Measure('ds', domain=mesh, subdomain_data=mf['facets_uncombined'])
        if 'cells_uncombined' in mf:
            self.dx_uncombined = d.Measure('dx', domain=mesh, subdomain_data=mf['cells_uncombined'])
        
class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
    on marker values from the .xml file.
    """
    def __init__(self, mesh_filename, name='parent_mesh'):
        super().__init__(self, name)
        self.load_mesh_from_xml(mesh_filename)
        self.child_meshes = {}
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
        self.mesh_filename = mesh_filename
        print(f"Mesh, \"{self.name}\", successfully loaded from file: {mesh_filename}!")

    def get_mesh_functions(self):
        # Aliases
        mesh        = self.dolfin_mesh
        has_surface = self.min_dim < self.max_dim
        volume_dim  = self.max_dim
        surface_dim = self.min_dim

        # Check validity
        if self.dolfin_mesh is None:
            print(f"Mesh {self.name} has no dolfin mesh to get a mesh function from.")
            return None
        assert len(self.child_meshes) > 0 # there should be at least one child mesh

        # Init mesh functions
        mf = dict()
        mf['cells'] = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
        if has_surface:
            mf['facets'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())
            
        # If any cell markers are given as a list we also create mesh functions to store the uncombined markers
        if any([cm.marker_list is not None for cm in self.child_meshes.values()]):
            mf['cells_uncombined']  = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
            if has_surface:
                mf['facets_uncombined'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())
        # Combine markers in a list 
        for cm in self.child_meshes.values(): 
            if cm.marker_list is None:
                continue
            for marker in cm.marker_list:
                if cm.dimensionality == self.max_dim:
                    mf['cells'].array()[mf['cells_uncombined'].array() == marker] = cm.primary_marker
                if cm.dimensionality < self.max_dim:
                    mf['facets'].array()[mf['facets_uncombined'].array() == marker] = cm.primary_marker
        
        self.mf = mf

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
    
    @cached_property
    def map_cell_to_parent_entity(self):
        """
        We use the word 'entity' to be dimensionally-agnostic.
        If the child has topological dimension one lower than the parent, then the child's cell is the parent's facet.
        """
        return np.array(self.mesh_view[self.parent_mesh.id].cell_map())
    @cached_property
    def map_facet_to_parent_entity(self):
        """
        We use parent indices as our "grounding point" for conversion
        Conversion is:
        self.map_facet_to_parent_vertex:         child facet -> child vertex -> parent vertex
        loop self.parent_mesh to invert its map: parent vertex -> parent_entity
        """
        mapping = []
        # Aliases
        pm_entities = self.parent_mesh.get_entities(self.dimensionality-1)
        list_of_sets = [set(pm_entities[entity_idx,:]) for entity_idx in range(pm_entities.shape[0])]

        # parent_vertex to parent entity
        for child_facet_idx in range(self.facets.shape[0]):
            subset = set(self.map_facet_to_parent_vertex[child_facet_idx,:])
            entity_idx = list_of_sets.index(subset)
            mapping.append(entity_idx)
        
        return np.array(mapping)

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

    def map_cell_to_sibling_entity(self, child_mesh):
        our_map   = self.map_cell_to_parent_entity
        their_map = child_mesh.map_



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
        mf_type = 'cells' if self.is_volume_mesh else 'facets'
        self.dolfin_mesh = d.MeshView.create(self.parent_mesh.mf[mf_type], self.primary_marker)
            
        self.dolfin_mesh.init()
    

    def get_mesh_functions(self):
        "Child mesh functions require transfering data from parent mesh functions"
        assert hasattr(self.parent_mesh, 'mf')
        # Alias
        pmf = self.parent_mesh.mf

        # initialize
        mf           = dict()
        mf['cells']  = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality, value=0)
        mf['facets'] = d.MeshFunction('size_t', self.dolfin_mesh, self.dimensionality-1, value=0)

        # Easiest case, just directly transfer from parent_mesh mesh function
        # local cell index = local cell index -> parent cell index -> parent mesh function value
        if self.is_volume_mesh:
            mf['cells'].array()[:]  = pmf['cells'].array()[self.map_cell_to_parent_entity]
            # Use the mapping from local facet to parent entity
            mf['facets'].array()[:] = pmf['facets'].array()[self.map_facet_to_parent_entity]
            if 'cells_uncombined' in pmf:
                mf['cells_uncombined'].array()[:]  = pmf['cells_uncombined'].array()[self.map_cell_to_parent_entity]
            if 'facets_uncombined' in pmf:
                mf['facets_uncombined'].array()[:]  = pmf['facets_uncombined'].array()[self.map_facet_to_parent_entity]
        else:
            mf['cells'].array()[:]  = pmf['facets'].array()[self.map_cell_to_parent_entity]
            if 'facets_uncombined' in pmf:
                mf['cells_uncombined'].array()[:]  = pmf['facets_uncombined'].array()[self.map_facet_to_parent_entity]

        self.mf = mf
