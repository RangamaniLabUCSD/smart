# testing vertex map


#V = model.V['cyto']
Vsubmesh = model.V['pm']
#mesh = model.CD.meshes['cyto']
submesh = model.CD.meshes['pm']
bmesh = model.CD.bmesh
Vdof = range(640)



idx=2

submesh_dof_to_vertex = dof_to_vertex_map(Vsubmesh)
submesh_dof_to_vertex = submesh_dof_to_vertex[range(0, len(submesh_dof_to_vertex), idx+1)]/(idx+1)
submesh_dof_to_vertex = submesh_dof_to_vertex.astype(int)


def submesh_dof_to_vertex(Vsubmesh, index):
    return dof_to_vertex_map(Vsubmesh)[index]

def submesh_to_bmesh(submesh, index):
    submesh_to_bmesh_vertex = submesh.data().array("parent_vertex_indices", 0)
    return submesh_to_bmesh_vertex[index]

def bmesh_to_parent(bmesh, index):
    return bmesh.entity_map(0).array()[index]

def mesh_vertex_to_dof(V, index):
    return vertex_to_dof_map(V)[index]

def submesh_dof_to_mesh_dof(Vsubmesh, submesh, bmesh, V, index):
    idx = submesh_dof_to_vertex(Vsubmesh, index)
    idx = submesh_to_bmesh(submesh, idx)
    idx = bmesh_to_parent(bmesh, idx)
    idx = mesh_vertex_to_dof(V, idx)
    return idx


submesh_dof_to_mesh_dof(Vs,submesh,bmesh,V,range(3))


bmesh_to_parent




submesh_vertices = dof_to_vertex_map(Vsubmesh)

bmesh_vertices = [int(x) for x in submesh.data().array("parent_vertex_indices", 0)]
mesh_vertices = [bmesh.entity_map(0)[x] for x in bmesh_vertices]
return d.vertex_to_dof_map(Vdof)[mesh_vertices]



model.CD.create_vertex_mapping(meshS, Vs, bmesh, Vdof)