"""
Wrapper around dolfin mesh class (originally for submesh implementation - possibly unneeded now)
"""
import dolfin as d
from stubs.common import _fancy_print as fancy_print
import numpy as np

class _Mesh:
    """
    General mesh class
    """
    def __init__(self, name='mesh_name', dimensionality=None):
        self.name               = name
        self.dimensionality     = dimensionality
        self.dolfin_mesh        = None

    @property
    def mesh_view(self):
        return self.dolfin_mesh.topology().mapping()
    @property
    def id(self):
        return self.dolfin_mesh.id()
    
    def get_num_entities(self, dimension):
        "Get the number of entities in this mesh with a certain topological dimension"
        return self.dolfin_mesh.topology().size(dimension)

    @property
    def num_cells(self):
        return self.get_num_entities(self.dimensionality)
    @property
    def num_facets(self):
        return self.get_num_entities(self.dimensionality-1)
    @property
    def num_vertices(self):
        return self.get_num_entities(0)
    
    # Index mapping
    def _get_entities(self, dimension):
        num_vertices_per_entity = dimension+1 # for a simplex
        return np.reshape(self.dolfin_mesh.topology()(dimension, 0)(), (self.get_num_entities(dimension), dimension+1))

    @property
    def cells(self):
        return self.dolfin_mesh.cells()
    @property
    def facets(self):
        # By default dolfin only stores cells - we must call dolfin_mesh.init() in order to access index maps for other dimensions
        return self._get_entities(self.dimensionality-1)
    @property
    def vertices(self):
        return self.dolfin_mesh.coordinates()

    @property
    def cell_coordinates(self):
        return self.vertices[self.cells]
    @property
    def facet_coordinates(self):
        return self.vertices[self.facets]

    def get_mesh_coordinate_bounds(self):
        return {'min': self.vertices.min(axis=0), 'max': self.vertices.max(axis=0)}

        
class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
    on marker values from the .xml file.
    """
    def __init__(self, mesh_filename, name='parent_mesh'):
        self.name = name
        self.load_mesh_from_xml(mesh_filename)
        self.child_meshes = {}
        # get mesh functions
        #self.mesh_functions = self.get_mesh_functions()

    def load_mesh_from_xml(self, mesh_filename):
        self.dolfin_mesh = d.Mesh(mesh_filename)
        self.dolfin_mesh.init()
        self.dimensionality = self.dolfin_mesh.topology().dim()
        self.mesh_filename = mesh_filename
        print(f"Mesh, \"{self.name}\", successfully loaded from file: {mesh_filename}!")

    def get_mesh_functions(self):
        # Aliases
        mesh        = self.dolfin_mesh
        has_surface = self._min_dim < self._max_dim
        volume_dim  = self._max_dim
        surface_dim = self._min_dim

        # Check validity
        if self.dolfin_mesh is None:
            print(f"Mesh {self.name} has no dolfin mesh to get a mesh function from.")
            return None
        assert len(self.child_meshes) > 0 # there should be at least one child mesh

        # Init mesh functions
        mf = dict()
        mf['vol'] = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
        if has_surface:
            mf['surf'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())
        # If any cell markers are given as a list we also create mesh functions to store the uncombined markers
        if any([cm.marker_list is not None for cm in self.child_meshes.values()]):
            mf['vol_uncombined']  = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
            if has_surface:
                mf['surf_uncombined'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())

        # Combine markers in a list 
        for cm in self.child_meshes.values(): 
            if cm.marker_list is None:
                continue
            for marker in cm.marker_list:
                if cm.dimensionality == self._max_dim:
                    mf['vol'].array()[mf['vol_uncombined'].array() == marker] = cm.primary_marker
                if cm.dimensionality < self._max_dim:
                    mf['surf'].array()[mf['surf_uncombined'].array() == marker] = cm.primary_marker
        
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
        assert type(parent_mesh) == ParentMesh
        self.set_parent_mesh(parent_mesh)

        # markers can be either an int or a list of ints
        if type(marker) == list:
            assert all([type(m) == int for m in marker])
            self.marker_list = marker
            self.primary_marker = marker[0]
            fancy_print(f"List of markers given for compartment {self.name}, combining into single marker, {marker[0]}")
        else:
            assert type(marker) == int
            self.marker_list = None
            self.primary_marker = marker
    
    @property
    def mapping_child_cell_to_parent_entity(self):
        return np.array(self.mesh_view[self.parent_mesh.id].cell_map())
    @property
    def mapping_child_vertex_to_parent_vertex(self):
        return np.array(self.mesh_view[self.parent_mesh.id].vertex_map())
    @property
    def mapping_child_cell_to_parent_vertex(self):
        return self.mapping_child_vertex_to_parent_vertex[self.cells]
    @property
    def is_surface_mesh(self):
        return self.dimensionality < self.parent_mesh.dimensionality


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
        if self.dimensionality == self.parent_mesh._max_dim:
            self.mf = self.parent_mesh.mf['vol']
        elif self.dimensionality < self.parent_mesh._max_dim:
            self.mf = self.parent_mesh.mf['surf']

        self.dolfin_mesh = d.MeshView.create(self.mf, self.primary_marker)
        self.dolfin_mesh.init()
    
    # def get_integration_measures(self):
    #     # Aliases
    #     comp = self.compartment
    #     mesh = self.dolfin_mesh
    #     parent_mesh = self.parent_mesh.dolfin_mesh

    #     # Markers on the boundary mesh of this child mesh
    #     if self.dimensionality == self.parent_mesh._max_dim:
    #         self.ds            = d.Measure('ds', domain=mesh, subdomain_data=vmf)
    #         self.ds_uncombined = d.Measure('ds', domain=mesh, subdomain_data=vmf)
    #         #comp.dP = None
    #     elif comp.dimensionality < self.parent_mesh._max_dim:
    #         #comp.dP = d.Measure('dP', domain=comp.mesh)
    #         comp.ds = None

    #     comp.dx = d.Measure('dx', domain=mesh)

        
        

    #         # integration measures
    #         if comp.dimensionality==main_mesh.dimensionality:
    #             comp.ds = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf_combined)
    #             comp.ds_uncombined = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf)
    #             comp.dP = None
    #         elif comp.dimensionality<main_mesh.dimensionality:
    #             comp.dP = d.Measure('dP', domain=comp.mesh)
    #             comp.ds = None
    #         else:
    #             raise Exception("main_mesh is not a maximum dimension compartment")
    #         comp.dx = d.Measure('dx', domain=comp.mesh)




