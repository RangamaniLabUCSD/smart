"""
Wrapper around dolfin mesh class (originally for submesh implementation - possibly unneeded now)
"""
import dolfin as d

# class _Mesh:
#     """
#     General mesh class
#     """
#     def __init__(self, name='mesh_name', _is_parent_mesh=False, dimensionality=None, mesh_filename=None):
#         self.name               = name
#         self._is_parent_mesh    = _is_parent_mesh
#         self.dimensionality     = dimensionality

#         if mesh_filename is not None:
#             self.mesh_filename = mesh_filename.__str__()
#             self.load_mesh_from_xml()
#     def load_mesh_from_xml(self):
#         if self._is_parent_mesh is not True:
#             raise ValueError("Mesh must be a parent mesh in order to load from xml.")
#         self.dolfin_mesh = d.Mesh(self.mesh_filename)
#         print(f"Mesh, \"{self.name}\", successfully loaded from file: {self.mesh_filename}!")
#     def get_mesh_coordinate_bounds(self):
#         return {'min': self.dolfin_mesh.coordinates().min(axis=0), 'max': self.dolfin_mesh.coordinates().max(axis=0)}
        

# class ParentMesh(_Mesh):
#     """
#     Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
#     on marker values from the .xml file.
#     """
#     def __init__(self, name='parent_mesh', mesh_filename=None):
#         self.mesh_filename = mesh_filename
#         super().__init__(name=name, _is_parent_mesh=True, mesh_filename=mesh_filename)
#         self.dimensionality = self.dolfin_mesh.geometric_dimension()

# class ChildMesh(_Mesh):
#     """
#     Sub mesh of a parent mesh
#     """
#     def __init__(self, parent_mesh, dimensionality, marker_value, name='child_mesh'):
#         super().__init__(name=name, _is_parent_mesh=False,
#                          dimensionality=dimensionality)
#         self.marker_value = marker_value


# list of things to do
# define mesh functions
# extract submeshes ()


class _Mesh:
    """
    General mesh class
    """
    def __init__(self, name='mesh_name', dimensionality=None, dolfin_mesh=None):
        self.name               = name
        self.dimensionality     = dimensionality
        self.dolfin_mesh        = dolfin_mesh

    def get_mesh_coordinate_bounds(self):
        if self.dolfin_mesh is None:
            raise ValueError("Mesh must have an associated dolfin_mesh to compute coordinate bounds")
        return {'min': self.dolfin_mesh.coordinates().min(axis=0), 'max': self.dolfin_mesh.coordinates().max(axis=0)}
    
    @property
    def mesh_view(self):
        return self.dolfin_mesh.topology().mapping()
    @property
    def id(self):
        return self.dolfin_mesh.id()
        

class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
    on marker values from the .xml file.
    """
    def __init__(self, mesh_filename, name='parent_mesh'):
        self.name = name
        self.load_mesh_from_xml(mesh_filename)
        self.child_meshes = []
        # get mesh functions
        #self.mesh_functions = self.get_mesh_functions()

    def load_mesh_from_xml(self, mesh_filename):
        self.dolfin_mesh = d.Mesh(mesh_filename)
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

        # Init MeshFunctions
        mf = dict()
        mf['vol'] = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
        if has_surface:
            mf['surf'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())
        # If any cell markers are given as a list we also create mesh functions to store the uncombined markers
        if any([cm.marker_list is not None for cm in self.child_meshes]):
            mf['vol_uncombined']  = d.MeshFunction('size_t', mesh, volume_dim, value=mesh.domains())
            if has_surface:
                mf['surf_uncombined'] = d.MeshFunction('size_t', mesh, surface_dim, value=mesh.domains())

        # Combine markers in a list 
        for cm in self.child_meshes: 
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
    def cell_mapping_to_parent(self):
        return self.mesh_view[self.parent_mesh.id].cell_map()
    @property
    def vertex_mapping_to_parent(self):
        return self.mesh_view[self.parentmesh.id].vertex_map()

    def set_parent_mesh(self, parent_mesh):
        # remove existing parent mesh if not None
        if self.parent_mesh:
            self.parent_mesh.child_meshes.remove(self)
        # set the parent mesh
        self.parent_mesh = parent_mesh
        # add self to parent mesh's list of children meshes
        if self.parent_mesh:
            self.parent_mesh.child_meshes.append(self)

    def extract_submesh(self):
        if self.dimensionality == self.parent_mesh._max_dim:
            self.mf = self.parent_mesh.mf['vol']
        elif self.dimensionality < self.parent_mesh._max_dim:
            self.mf = self.parent_mesh.mf['surf']

        self.dolfin_mesh = d.MeshView.create(self.mf, self.primary_marker)
    
#    def get_integration_measures(self):
#        # Aliases
#        comp = self.compartment
#        mesh = self.dolfin_mesh
#        parent_mesh = self.parent_mesh.dolfin_mesh
#
#        if self.dimensionality == self.parent_mesh._max_dim:
#            self.ds            = d.Measure('ds', domain=mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
#            self.ds_uncombined = d.Measure('ds', domain=mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
#            #comp.dP = None
#        elif comp.dimensionality < self.parent_mesh._max_dim:
#            #comp.dP = d.Measure('dP', domain=comp.mesh)
#            comp.ds = None
#
#        comp.dx = d.Measure('dx', domain=mesh, metadata={'quadrature_degree': 3})




