# embedded_balls mesh demonstration
# showing how fenics mesh functions work with the embedded_balls mesh file

from dolfin import *

m       = Mesh("embedded_balls.xml")
mf33    = MeshFunction('size_t', m, 3, m.domains())
minner  = SubMesh(m,mf33,3)
mouter  = SubMesh(m,mf33,1)

mf32    = MeshFunction('size_t', m, 2, m.domains())



bmesh        = BoundaryMesh(m, "exterior")
temp_emap_0  = bmesh.entity_map(0)
bmesh_emap_0 = deepcopy(temp_emap_0.array())
temp_emap_n  = bmesh.entity_map(surfaceDim)
bmesh_emap_n = deepcopy(temp_emap_n.array())

mf32         = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
mf22         = d.MeshFunction("size_t", bmesh, surfaceDim)
# iterate through facets of bmesh (transfer markers from volume mesh function to boundary mesh function)
for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): 
    vmesh_idx            = bmesh_emap_n[idx]        # get the index of the face on vmesh corresponding to this face on bmesh
    vmesh_boundarynumber = mf32.array()[vmesh_idx]  # get the value of the mesh function at this face
    mf22.array()[idx]    = vmesh_boundarynumber     # set the value of the boundary mesh function to be the same value



mf2_out = MeshFunction('size_t', m_out, 2, m_out.domains())


mb     = BoundaryMesh(m_out, 'exterior')
mb
mb2    = SubMesh(mtot,mf2,2)
mb4    = SubMesh(mb,mfb,4)

temp_emap_0  = bmesh.entity_map(0)
bmesh_emap_0 = deepcopy(temp_emap_0.array())
temp_emap_n  = bmesh.entity_map(surfaceDim)
bmesh_emap_n = deepcopy(temp_emap_n.array())


# iterate through facets of bmesh (transfer markers from volume mesh function to boundary mesh function)
for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): 
    vmesh_idx = bmesh_emap_n[idx] # get the index of the face on vmesh corresponding to this face on bmesh
    vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
    bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value