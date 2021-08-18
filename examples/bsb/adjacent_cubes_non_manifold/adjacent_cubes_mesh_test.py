#adjacent_cubes_mesh_test.py


import dolfin as d

#def get_full_boundary_mesh(self):
#m       = self.dolfin_mesh
#dim     = self.dimensionality
m = d.Mesh('adjacent_cubes.xml')
dim = m.geometric_dimension()

# Mesh function for sub-volumes
mf_vv   = d.MeshFunction('size_t', m, dim, m.domains())
# Get unique sub-volume markers
unique_markers = list(set(mf_vv.array()))
# Get BoundaryMesh for each sub-volume
sm = dict() # submeshes for each sub-volume
bm = dict() # boundary mesh for each submesh
# Entity maps
submesh_to_parent_mesh_0    = dict()
submesh_to_parent_mesh_n    = dict()
bmesh_to_submesh_0          = dict()
bmesh_to_submesh_n          = dict()
# Union of boundary meshes
bm_union = d.Mesh()

# for testing
# make sure mapping from bm -> submesh -> mesh works
coord_idx = 12 WIP# x=[1,1,0]
cell_idx = 12  WIP# x=[1,1,0]


for marker in unique_markers:
    sm[marker]                          = d.SubMesh(m, mf_vv, marker)
    # Get entity maps (submesh -> parent mesh)
    submesh_to_parent_mesh_0[marker]    = sm[marker].data().array('parent_vertex_indices', 0)
    submesh_to_parent_mesh_n[marker]    = sm[marker].data().array('parent_cell_indices', dim)

    bm[marker]  = d.BoundaryMesh(sm[marker], 'exterior')
    print(f"BoundaryMesh of SubMesh (marker {marker}) has {bm[marker].num_vertices()} vertices and {bm[marker].num_cells()} cells.")
    print(f"BoundaryMesh of SubMesh (marker {marker}) has min,max z-value "
          + f"({bm[marker].coordinates()[:,2].min()}, {bm[marker].coordinates()[:,2].max()})\n")

    # Get entity maps (boundary -> submesh)
    temp_emap_0 = bm[marker].entity_map(0)
    temp_emap_n = bm[marker].entity_map(dim-1)
    bmesh_to_submesh_0[marker] = deepcopy(temp_emap_0.array())
    bmesh_to_submesh_n[marker] = deepcopy(temp_emap_n.array())

    # Combine entity maps

#np_unique_array = lambda 
#https://stackoverflow.com/questions/49950412/merge-two-numpy-arrays-and-delete-duplicates

# https://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh/

# Get entity maps
temp_emap_0 = 

# Find overlapping vertices and cells


# Combine meshes
me = d.MeshEditor()
b = bm[11]
btype = d.CellType.type2string(b.type().cell_type())
me.open(b, btype, b.topology().dim(), b.geometric_dimension())

