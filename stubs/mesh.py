"""

"""
import dolfin as d

class _Mesh(object):
    """
    General mesh class
    """
    def __init__(self, name='mesh_name', _is_parent_mesh=False, dimensionality=None, mesh_filename=None):
        self.name               = name
        self._is_parent_mesh    = _is_parent_mesh
        self.dimensionality     = dimensionality

        if mesh_filename is not None:
            self.mesh_filename = mesh_filename
            self.load_mesh_from_xml()
    def load_mesh_from_xml(self):
        if self._is_parent_mesh is not True:
            raise ValueError("Mesh must be a parent mesh in order to load from xml.")
        self.dolfin_mesh = d.Mesh(self.mesh_filename)
        print(f"Mesh, \"{self.name}\", successfully loaded from file: {self.mesh_filename}!")

class ParentMesh(_Mesh):
    """
    Mesh loaded in from data. Submeshes are extracted from the ParentMesh based 
    on marker values from the .xml file.

                                    !!! WARNING !!!
    !!! If using a mesh with multiple sub-volumes then a "primary cell marker" must be specified !!!
                                    !!! WARNING !!!

    All unmarked surfaces are assumed to have a zero-flux (Neumann) boundary condition.
    All marked surfaces (indicating there are either non-zero boundary conditions or equations)
    must be in contact with the primary cell marker. This is due to the behavior of
    dolfin.BoundaryMesh and dolfin.SubMesh - a more robust solution may be implemented in the future.

    EXAMPLES: 

    EXAMPLE 1 (sub-volume surrounded by primary cell marker):
    * There are two sub-volumes with marker values (1) and (2).
    * There is a marked surface on the bottom edge of sub-volume (1) with value 10
    * There is a marked surface between sub-volumes (1) and (2) with value 12
    ┌────────────────────────┐
    │                        │
    │                        │
    │             ┌──12───┐  │
    │             │       │  │
    │             │       │  │
    │   (1)       │  (2)  │  │
    │             │       │  │
    │             └───────┘  │
    │                        │
    └──────────10────────────┘
    This is a VALID mesh. In this scenario (1) MUST be the primary cell marker 
    because all marked surfaces are in contact with that sub-volume.
    
    EXAMPLE 2 (sub-volume with non-manifold surface):
    * There are two sub-volumes with marker values (1) and (2).
    * There is a marked surface on the bottom edge of sub-volume (1) with value 10
    * There is a marked surface between sub-volumes (1) and (2) with value 12
    ┌───────────────┬────────┐
    │               │        │
    │               │   (2)  │
    │               │        │
    │               └───12───┤
    │                        │
    │           (1)          │
    │                        │
    │                        │
    │                        │
    └────────────10──────────┘
    This is a VALID mesh. In this scenario (1) MUST be the primary cell marker 
    because all marked surfaces are in contact with that sub-volume.
    
    EXAMPLE 3 (sub-volume with non-manifold surface and marked boundary):
    * There are two sub-volumes with marker values (1) and (2).
    * There is a marked surface on the bottom edge of sub-volume (1) with value 10
    * There is a marked surface between sub-volumes (1) and (2) with value 12
    * There is a marked surface on the top edge of sub-volume (2) with value 20
    ┌───────────────┬───20───┐
    │               │        │
    │               │   (2)  │
    │               │        │
    │               └───12───┤
    │                        │
    │           (1)          │
    │                        │
    │                        │
    │                        │
    └────────────10──────────┘
    This is an INVALID mesh. No sub-volume is in contact with all marked surfaces.
    """
    def __init__(self, name='parent_mesh', mesh_filename=None):
        self.mesh_filename = mesh_filename
        super().__init__(name=name, _is_parent_mesh=True, dimensionality=None,
                         mesh_filename=mesh_filename)
        self.dimensionality = self.dolfin_mesh.geometric_dimension

#     def get_full_boundary_mesh(self):
# m       = self.dolfin_mesh
# dim     = self.dimensionality
# # Mesh function for sub-volumes
# mf_vv   = d.MeshFunction('size_t', m, dim, m.domains())
# # Get unique sub-volume markers
# unique_markers = list(set(mf_vv.array()))
# # Get BoundaryMesh for each sub-volume
# bm = dict()
# bm_union = d.Mesh()
# for marker in unique_markers:
#     bm[marker] = d.BoundaryMesh(m, 'exterior')

# me = d.MeshEditor()
# b = bm[11]
# btype = d.CellType.type2string(b.type().cell_type())
# me.open(b, btype, b.topology().dim(), b.geometric_dimension())


        # Combine BoundaryMesh while eliminating 

class ChildMesh(_Mesh):
    """
    Sub mesh of a parent mesh
    """
    def __init__(self, parent_mesh, dimensionality, marker_value, name='child_mesh'):
        super().__init__(name=name, _is_parent_mesh=False,
                         dimensionality=dimensionality)
        self.marker_value = marker_value


